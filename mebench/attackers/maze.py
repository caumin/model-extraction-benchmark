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
        # Replay buffer as individual samples
        state.attack_state["replay_x"] = torch.empty(0) 
        state.attack_state["replay_y"] = torch.empty(0)
        state.attack_state["replay_ptr"] = 0
        state.attack_state["replay_count"] = 0
        
        # Track position in Algorithm 1 cycle
        # Cycle consists of n_g_steps (each with 1 base + m pert) and n_c_steps (each with 1 base)
        state.attack_state["cycle_step"] = 0 
        state.attack_state["sub_cycle_step"] = 0 # 0 for base, 1..m for pert
        state.attack_state["current_z"] = None # Current base z for perturbations

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
        
        # Consistent with Eq. 11 and Algorithm 1
        # Generator iteration uses B(m+1) queries
        # Clone iteration uses B queries
        # Total cycle: n_g_steps * B * (m+1) + n_c_steps * B
        
        if self.generator is None:
            self.generator = self._create_generator().to(device)
            self.generator_optimizer = optim.SGD(
                self.generator.parameters(), 
                lr=self.generator_lr, 
                momentum=0.5
            )
            # t_max for scheduler: total iterations
            # queries_per_cycle = batch_size * (n_g_steps * (grad_approx_m + 1) + n_c_steps)
            q_per_cycle = self.batch_size * (self.n_g_steps * (self.grad_approx_m + 1) + self.n_c_steps)
            t_max = max(1, int(max_budget / q_per_cycle))
            self.generator_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.generator_optimizer, t_max
            )
            
        if self.clone is None:
            self.clone = self._create_clone(state.metadata.get("input_shape", (3, 32, 32))).to(device)
            sub_config = state.metadata.get("substitute_config", {})
            opt_params = sub_config.get("optimizer", {})
            self.clone_optimizer = optim.SGD(
                self.clone.parameters(),
                lr=float(opt_params.get("lr", self.clone_lr)),
                momentum=float(opt_params.get("momentum", 0.9)),
                weight_decay=float(opt_params.get("weight_decay", 5e-4))
            )
            q_per_cycle = self.batch_size * (self.n_g_steps * (self.grad_approx_m + 1) + self.n_c_steps)
            t_max = max(1, int(max_budget / q_per_cycle))
            self.clone_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.clone_optimizer, t_max
            )

        remaining = int(k)
        x_all = []
        batch_meta = []
        
        ng = self.n_g_steps
        nc = self.n_c_steps
        m = self.grad_approx_m
        batch_size = self.batch_size
        
        # We work in blocks of batch_size
        while remaining >= batch_size:
            b_in_cycle = state.attack_state.get("batch_in_cycle", 0)
            total_batches_in_cycle = ng * (m + 1) + nc
            
            if b_in_cycle < ng * (m + 1):
                # Generator training phase
                g_iter = b_in_cycle // (m + 1)
                g_sub = b_in_cycle % (m + 1)
                
                if g_sub == 0:
                    # Base batch for G update
                    z = torch.randn(batch_size, self.noise_dim, device=device)
                    state.attack_state["current_z"] = z.cpu()
                    with torch.no_grad():
                        x = self.generator(z)
                    type_str = "G_BASE"
                else:
                    # Perturbation batch
                    z = state.attack_state["current_z"].to(device)
                    with torch.no_grad():
                        x_base = self.generator(z)
                    
                    dirs = torch.randn(batch_size, *x_base.shape[1:], device=device)
                    # Normalize directions
                    norm = dirs.view(batch_size, -1).norm(dim=1, keepdim=True) + 1e-12
                    dirs = dirs / norm.view(batch_size, 1, 1, 1)
                    
                    x = x_base + self.grad_approx_epsilon * dirs
                    type_str = "G_PERT"
                    
                meta = {
                    "type": type_str,
                    "iter": g_iter,
                    "sub": g_sub,
                    "z": z.cpu() if g_sub == 0 else None,
                    "dirs": dirs.cpu() if g_sub > 0 else None
                }
            else:
                # Clone training phase
                c_iter = b_in_cycle - ng * (m + 1)
                z = torch.randn(batch_size, self.noise_dim, device=device)
                with torch.no_grad():
                    x = self.generator(z)
                
                meta = {
                    "type": "C_BASE",
                    "iter": c_iter,
                    "sub": 0,
                    "z": z.cpu(),
                    "dirs": None
                }
            
            x_all.append(x)
            batch_meta.append(meta)
            
            remaining -= batch_size
            b_in_cycle = (b_in_cycle + 1) % total_batches_in_cycle
            state.attack_state["batch_in_cycle"] = b_in_cycle
            
        if len(x_all) == 0:
            # Budget k too small for a full batch?
            # Fallback: return noise to fill budget and avoid hanging
            input_shape = state.metadata.get("input_shape", (3, 32, 32))
            x = torch.randn(int(k), *input_shape)
            return QueryBatch(x=x, meta={"type": "NOISE"})

        x_final = torch.cat(x_all, dim=0)
        # If engine requested more than we produced (due to batching), fill with noise
        if x_final.size(0) < int(k):
            pad = int(k) - x_final.size(0)
            x_pad = torch.randn(pad, *x_final.shape[1:], device=device)
            x_final = torch.cat([x_final, x_pad], dim=0)
            batch_meta.append({"type": "NOISE", "pad": pad})

        return QueryBatch(x=x_final, meta={"blocks": batch_meta})

    def observe(
        self,
        query_batch: QueryBatch,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        if self.generator is None or self.clone is None:
            return

        if oracle_output.kind != "soft_prob":
            # Paper strictly uses soft labels. Hard labels violate the MAZE gradient estimation logic.
            raise ValueError("MAZE attack requires soft_prob (soft labels) output mode.")

        device = query_batch.x.device
        blocks = query_batch.meta.get("blocks", [])
        if not blocks:
            return

        batch_size = self.batch_size
        cursor = 0
        
        def _normalize_maze(x):
            victim_config = state.metadata.get("victim_config", {})
            normalization = victim_config.get("normalization")
            if normalization is None:
                normalization = {"mean": [0.0], "std": [1.0]}
            norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(device)
            norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(device)
            x_01 = x * 0.5 + 0.5
            return (x_01 - norm_mean) / norm_std

        for block in blocks:
            if block["type"] == "NOISE":
                cursor += block.get("pad", 0)
                continue
                
            x_batch = query_batch.x[cursor : cursor + batch_size]
            y_batch = oracle_output.y[cursor : cursor + batch_size]
            cursor += batch_size
            
            if block["type"] == "G_BASE":
                state.attack_state["pending_g"] = {
                    "z": block["z"].to(device),
                    "x_base": x_batch,
                    "y_base": y_batch,
                    "pert_y": [],
                    "pert_dirs": []
                }
            elif block["type"] == "G_PERT":
                pending = state.attack_state.get("pending_g")
                if pending:
                    pending["pert_y"].append(y_batch)
                    pending["pert_dirs"].append(block["dirs"].to(device))
                    
                    if len(pending["pert_y"]) == self.grad_approx_m:
                        self._update_generator(state, pending, _normalize_maze)
                        state.attack_state["pending_g"] = None
                        
            elif block["type"] == "C_BASE":
                # Train clone (Algorithm 1 Step 13)
                self.clone_optimizer.zero_grad()
                clone_logits = self.clone(_normalize_maze(x_batch))
                loss = F.kl_div(
                    F.log_softmax(clone_logits, dim=1),
                    y_batch,
                    reduction="batchmean",
                )
                loss.backward()
                self.clone_optimizer.step()
                
                # Experience Replay storage (Step 14)
                self._append_replay(x_batch.detach(), y_batch.detach(), state)
                
                # Algorithm 1: Replay training (Steps 15-17)
                # Paper implies this happens after NC loop, but here we do it per C_BASE block 
                # to maintain the NR ratio if multiple C_BASE blocks are in one observe.
                self._replay_train(state, device, _normalize_maze)

        if self.clone_scheduler is not None:
            self.clone_scheduler.step()
        if self.generator_scheduler is not None:
            self.generator_scheduler.step()

        state.attack_state["step"] += 1
        state.attack_state["substitute"] = self.clone

    def _update_generator(self, state, pending, norm_fn):
        device = pending["z"].device
        self.generator_optimizer.zero_grad()
        
        # Fresh G(z) for the gradient step (Algorithm 1 Step 3)
        # Note: we use the saved z. 
        # Requirement: "Regenerate x (or at least recompute the ZO estimates) each time"
        x_base_recon = self.generator(pending["z"])
        clone_base = self.clone(norm_fn(x_base_recon))
        
        # Loss for G: minimize similarity (maximize disagreement)
        # Paper Eq. 6/7/8 uses KL divergence as loss L
        loss_base = F.kl_div(
            F.log_softmax(clone_base, dim=1),
            pending["y_base"],
            reduction="none",
        ).sum(dim=1)
        
        grad_est = torch.zeros_like(x_base_recon)
        m = self.grad_approx_m
        d = x_base_recon[0].numel()
        
        # We need the victim's outputs for perturbed samples
        for i in range(m):
            y_pert = pending["pert_y"][i]
            dirs = pending["pert_dirs"][i]
            
            # Note: directions were added to x_base (from generator(z) at proposal time)
            # Re-evaluating clone on x_pert = x_base_recon + eps * dirs
            x_pert_recon = x_base_recon + self.grad_approx_epsilon * dirs
            clone_pert = self.clone(norm_fn(x_pert_recon))
            
            loss_pert = F.kl_div(
                F.log_softmax(clone_pert, dim=1),
                y_pert,
                reduction="none",
            ).sum(dim=1)
            
            loss_diff = (loss_pert - loss_base) / self.grad_approx_epsilon
            
            # g = (d/m) * sum( (L(x+eu)-L(x))/e * u )
            # We accumulate over m directions
            for b in range(self.batch_size):
                grad_est[b] += (loss_diff[b] * dirs[b]) * (d / m)

        # Gradient Ascent: backward with negative gradient
        x_base_recon.backward(-grad_est)
        self.generator_optimizer.step()

    def _append_replay(
        self, x: torch.Tensor, y: torch.Tensor, state: BenchmarkState
    ) -> None:
        # Append individual samples (Step 14)
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        
        if state.attack_state["replay_x"].numel() == 0:
            state.attack_state["replay_x"] = x_cpu
            state.attack_state["replay_y"] = y_cpu
        else:
            state.attack_state["replay_x"] = torch.cat([state.attack_state["replay_x"], x_cpu], dim=0)
            state.attack_state["replay_y"] = torch.cat([state.attack_state["replay_y"], y_cpu], dim=0)
            
        # Manage buffer size
        if state.attack_state["replay_x"].size(0) > self.replay_max:
            state.attack_state["replay_x"] = state.attack_state["replay_x"][-self.replay_max:]
            state.attack_state["replay_y"] = state.attack_state["replay_y"][-self.replay_max:]
        
        state.attack_state["replay_count"] = state.attack_state["replay_x"].size(0)

    def _replay_train(self, state: BenchmarkState, device: str, norm_fn) -> None:
        count = state.attack_state.get("replay_count", 0)
        if count < self.batch_size:
            return
            
        replay_x = state.attack_state["replay_x"]
        replay_y = state.attack_state["replay_y"]

        for _ in range(self.n_r_steps):
            indices = np.random.choice(count, self.batch_size, replace=False)
            x = replay_x[indices].to(device)
            y = replay_y[indices].to(device)
            
            self.clone_optimizer.zero_grad()
            clone_logits = self.clone(norm_fn(x))
            loss = F.kl_div(
                F.log_softmax(clone_logits, dim=1),
                y,
                reduction="batchmean",
            )
            loss.backward()
            self.clone_optimizer.step()
