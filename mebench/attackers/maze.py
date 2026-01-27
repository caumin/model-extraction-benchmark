"""MAZE (Model Stealing via Zeroth-Order Gradient Estimation) attack."""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from mebench.attackers.base import BaseAttack
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.models.gan import DCGANGenerator
from mebench.models.substitute_factory import create_substitute


class MAZE(BaseAttack):
    """MAZE attack with zeroth-order generator updates and replay buffer."""

    def __init__(self, config: dict, state: BenchmarkState):
        super().__init__(config, state)

        self.batch_size = int(config.get("batch_size", 128))
        self.clone_lr = float(config.get("clone_lr", 0.1))
        self.generator_lr = float(config.get("generator_lr", 1e-4))
        self.n_g_steps = int(config.get("n_g_steps", 1))
        self.n_c_steps = int(config.get("n_c_steps", 5))
        self.n_r_steps = int(config.get("n_r_steps", 10))
        self.grad_approx_m = int(config.get("grad_approx_m", 10))
        self.grad_approx_epsilon = float(config.get("grad_approx_epsilon", 1e-3))
        self.noise_dim = int(config.get("noise_dim", 100))
        self.replay_max = int(config.get("replay_max", 5000))
        self.num_classes = int(
            self.state.metadata.get("num_classes")
            or self.state.metadata.get("dataset_config", {}).get("num_classes", 10)
        )
        self.base_channels = int(config.get("base_channels", 64))

        self.generator: nn.Module | None = None
        self.clone: nn.Module | None = None
        self.generator_optimizer: optim.Optimizer | None = None
        self.clone_optimizer: optim.Optimizer | None = None
        self.clone_scheduler: optim.lr_scheduler.CosineAnnealingLR | None = None
        self.generator_scheduler: optim.lr_scheduler.CosineAnnealingLR | None = None

        self._initialize_state(state)

    def _initialize_state(self, state: BenchmarkState) -> None:
        state.attack_state["step"] = 0
        state.attack_state["replay_x"] = []
        state.attack_state["replay_y"] = []

    def _create_generator(self) -> nn.Module:
        input_shape = self.state.metadata.get("input_shape", (3, 32, 32))
        return DCGANGenerator(
            noise_dim=self.noise_dim,
            output_channels=int(self.config.get("output_channels", input_shape[0])),
            base_channels=self.base_channels,
            num_classes=None,
            output_size=int(input_shape[1]),
        )

    def _create_clone(self, input_shape: tuple) -> nn.Module:
        sub_config = self.state.metadata.get("substitute_config", {})
        arch = sub_config.get("arch", "resnet18-8x")
        input_channels = int(input_shape[0])
        return create_substitute(
            arch=arch,
            num_classes=self.num_classes,
            input_channels=input_channels,
        )

    def propose(self, k: int, state: BenchmarkState) -> QueryBatch:
        device = state.metadata.get("device", "cpu")
        max_budget = state.metadata.get("max_budget", 1)
        per_iter = (self.n_c_steps - 1) + (1 + self.grad_approx_m) * self.n_g_steps
        queries_per_step = max(1, int(k * max(1, per_iter)))
        t_max = max(1, int(max_budget / queries_per_step))
        
        sub_config = state.metadata.get("substitute_config", {})
        opt_params = sub_config.get("optimizer", {})

        if self.generator is None:
            self.generator = self._create_generator().to(device)
            self.generator_optimizer = optim.Adam(
                self.generator.parameters(), lr=self.generator_lr
            )
            self.generator_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.generator_optimizer, t_max
            )
        if self.clone is None:
            self.clone = self._create_clone(state.metadata.get("input_shape", (3, 32, 32))).to(device)
            self.clone_optimizer = optim.SGD(
                self.clone.parameters(),
                lr=float(opt_params.get("lr", self.clone_lr)),
                momentum=float(opt_params.get("momentum", 0.9)),
                weight_decay=float(opt_params.get("weight_decay", 5e-4))
            )
            self.clone_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.clone_optimizer, t_max
            )

        remaining = int(k)
        z_list = []
        m_list = []
        x_parts = []
        directions_list = []

        with torch.no_grad():
            x_shape = self.generator(torch.randn(1, self.noise_dim, device=device)).shape[1:]

        while remaining > 0:
            m_eff = 0
            if remaining > 1:
                m_eff = min(int(self.grad_approx_m), remaining - 1)
            z = torch.randn(1, self.noise_dim, device=device)
            with torch.no_grad():
                x_base = self.generator(z)
            x_parts.append(x_base)
            z_list.append(z)
            m_list.append(m_eff)

            if m_eff > 0:
                dirs = torch.randn(1, m_eff, *x_shape, device=device)
                norm = dirs.view(1, m_eff, -1).norm(dim=2, keepdim=True) + 1e-12
                dirs = dirs / norm.view(1, m_eff, 1, 1, 1)
                x_perturbed = x_base.unsqueeze(1) + self.grad_approx_epsilon * dirs
                x_perturbed = x_perturbed.view(m_eff, *x_shape)
                x_parts.append(x_perturbed)
                directions_list.append(dirs)
            else:
                directions_list.append(torch.zeros(1, 0, *x_shape, device=device))

            remaining -= 1 + m_eff

        x_all = torch.cat(x_parts, dim=0)
        max_m = int(self.grad_approx_m)
        n_bases = len(z_list)
        directions = torch.zeros(n_bases, max_m, *x_shape, device=device)
        m_per_base = torch.zeros(n_bases, dtype=torch.long, device=device)
        for i, m_eff in enumerate(m_list):
            m_per_base[i] = int(m_eff)
            if m_eff > 0:
                directions[i, :m_eff] = directions_list[i][0]

        meta = {
            "generator_step": state.attack_state["step"],
            "synthetic": True,
            "z": torch.cat(z_list, dim=0).detach().cpu(),
            "directions": directions.detach().cpu(),
            "m_per_base": m_per_base.detach().cpu(),
        }
        return QueryBatch(x=x_all, meta=meta)

    def observe(
        self,
        query_batch: QueryBatch,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        if self.generator is None or self.clone is None:
            return

        x_all = query_batch.x
        device = x_all.device
        z_cpu = query_batch.meta.get("z")
        directions_cpu = query_batch.meta.get("directions")
        m_per_base_cpu = query_batch.meta.get("m_per_base")

        if z_cpu is None or directions_cpu is None or m_per_base_cpu is None:
            return

        z = z_cpu.to(device)
        directions = directions_cpu.to(device)
        m_per_base = m_per_base_cpu.to(device)

        victim_config = state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization")
        if normalization is None:
            normalization = {"mean": [0.0], "std": [1.0]}
        norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(device)
        norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(device)

        def _normalize_maze(x):
            x_01 = x * 0.5 + 0.5
            return (x_01 - norm_mean) / norm_std

        bases = []
        victim_bases = []
        perturbed = []
        victim_perturbed = []

        if oracle_output.kind == "soft_prob":
            victim_probs = oracle_output.y
        else:
            victim_probs = F.one_hot(oracle_output.y, num_classes=self.num_classes).float()

        victim_probs = victim_probs.to(device)

        cursor = 0
        for i in range(int(m_per_base.numel())):
            m_eff = int(m_per_base[i].item())
            bases.append(x_all[cursor : cursor + 1])
            victim_bases.append(victim_probs[cursor : cursor + 1])
            cursor += 1
            if m_eff > 0:
                perturbed.append(x_all[cursor : cursor + m_eff])
                victim_perturbed.append(victim_probs[cursor : cursor + m_eff])
                cursor += m_eff
        
        x_base = torch.cat(bases, dim=0)
        victim_base = torch.cat(victim_bases, dim=0)
        x_perturbed = torch.cat(perturbed, dim=0) if perturbed else torch.empty(0, *x_all.shape[1:], device=device)
        victim_pert = torch.cat(victim_perturbed, dim=0) if victim_perturbed else torch.empty(0, self.num_classes, device=device)

        self._append_replay(x_base.detach(), victim_base.detach(), state)

        for _ in range(self.n_g_steps):
            self.generator_optimizer.zero_grad()
            x_base_grad = self.generator(z)
            clone_base = self.clone(_normalize_maze(x_base_grad))
            loss_base = F.kl_div(
                F.log_softmax(clone_base, dim=1),
                victim_base,
                reduction="none",
            ).sum(dim=1)

            if x_perturbed.numel() > 0:
                clone_perturbed = self.clone(_normalize_maze(x_perturbed.detach()))
                loss_pert_all = F.kl_div(
                    F.log_softmax(clone_perturbed, dim=1),
                    victim_pert,
                    reduction="none",
                ).sum(dim=1)
            else:
                loss_pert_all = torch.empty(0, device=device)

            grad_est = torch.zeros_like(x_base_grad)
            cursor_pert = 0
            for i in range(x_base_grad.size(0)):
                m_eff = int(m_per_base[i].item())
                if m_eff <= 0:
                    continue
                loss_i = loss_base[i]
                loss_p = loss_pert_all[cursor_pert : cursor_pert + m_eff]
                dirs = directions[i, :m_eff].view(m_eff, -1)
                loss_diff = (loss_p - loss_i) / float(self.grad_approx_epsilon)
                g = torch.sum(loss_diff.view(m_eff, 1) * dirs, dim=0)
                grad_est[i] = g.view_as(grad_est[i])
                cursor_pert += m_eff

            x_base_grad.backward(grad_est)
            self.generator_optimizer.step()

        for _ in range(self.n_c_steps):
            self.clone_optimizer.zero_grad()
            clone_logits = self.clone(_normalize_maze(x_base))
            loss = F.kl_div(
                F.log_softmax(clone_logits, dim=1),
                victim_base,
                reduction="batchmean",
            )
            loss.backward()
            self.clone_optimizer.step()

        self._replay_train(state, device, _normalize_maze)

        if self.clone_scheduler is not None:
            self.clone_scheduler.step()
        if self.generator_scheduler is not None:
            self.generator_scheduler.step()

        state.attack_state["step"] += 1
        state.attack_state["substitute"] = self.clone

    def _append_replay(
        self, x: torch.Tensor, victim_probs: torch.Tensor, state: BenchmarkState
    ) -> None:
        state.attack_state["replay_x"].append(x.cpu())
        state.attack_state["replay_y"].append(victim_probs.cpu())

        if len(state.attack_state["replay_x"]) > self.replay_max:
            state.attack_state["replay_x"] = state.attack_state["replay_x"][-self.replay_max :]
            state.attack_state["replay_y"] = state.attack_state["replay_y"][-self.replay_max :]

    def _replay_train(self, state: BenchmarkState, device: str, norm_fn) -> None:
        replay_x = state.attack_state["replay_x"]
        replay_y = state.attack_state["replay_y"]
        if len(replay_x) == 0:
            return

        for _ in range(self.n_r_steps):
            idx = np.random.randint(0, len(replay_x))
            x = replay_x[idx].to(device)
            y = replay_y[idx].to(device)
            self.clone_optimizer.zero_grad()
            clone_logits = self.clone(norm_fn(x))
            loss = F.kl_div(
                F.log_softmax(clone_logits, dim=1),
                y,
                reduction="batchmean",
            )
            loss.backward()
            self.clone_optimizer.step()
