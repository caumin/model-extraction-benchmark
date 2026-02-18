"""MAZE (Model Stealing via Zeroth-Order Gradient Estimation) attack implementation."""

from typing import Dict, Any, List, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.models.gan import DCGANGenerator
from mebench.models.substitute_factory import create_substitute


class MAZE(AttackRunner):
    """MAZE (data-free) implementation aligned with Kariyappa et al. (2021).

    Implements Algorithm 1 (data-free MAZE) with KL objectives and zeroth-order
    gradient estimation. MAZE-PD (partial-data; Section 6 / Algorithm 2) is not
    implemented in this benchmark.
    """

    def __init__(self, config: dict, state: BenchmarkState):
        super().__init__(config, state)

        self.batch_size = int(config.get("batch_size", 128))
        self.grad_approx_m = int(config.get("grad_approx_m", 10))
        self.epsilon = float(config.get("grad_approx_epsilon", 1e-3))
        self.n_g = int(config.get("n_g_steps", 1))
        self.n_c = int(config.get("n_c_steps", 5))
        self.n_r = int(config.get("n_r_steps", 10))
        self.noise_dim = int(config.get("noise_dim", 100))
        self.lr_schedule = str(config.get("lr_schedule", "multistep")).lower()
        if self.lr_schedule not in {"multistep", "cosine"}:
            raise ValueError(f"MAZE lr_schedule must be 'multistep' or 'cosine', got {self.lr_schedule!r}")
        
        self.generator = None
        self.clone = None
        self.g_scheduler = None
        self.c_scheduler = None
        self.replay_x = []
        self.replay_y = []
        
        self._initialize_models(state)

    def _initialize_models(self, state: BenchmarkState):
        device = state.metadata.get("device", "cpu")
        input_shape = state.metadata.get("input_shape", (3, 32, 32))
        num_classes = state.metadata.get("num_classes", 10)
        
        # Generator: Unconditional DCGAN per Algorithm 1
        self.generator = DCGANGenerator(
            noise_dim=self.noise_dim, 
            output_channels=input_shape[0], 
            output_size=input_shape[1]
        ).to(device)
        
        # Paper (Section 5.1): SGD optimizer with lr=1e-4 for G.
        g_lr = float(self.config.get("generator_lr", 1e-4))
        g_momentum = float(self.config.get("generator_momentum", 0.0))
        self.g_opt = optim.SGD(self.generator.parameters(), lr=g_lr, momentum=g_momentum)
        
        # Clone: honor substitute config if provided
        sub_config = state.metadata.get("substitute_config", {})
        arch = sub_config.get("arch") or self.config.get("student_arch", "resnet18-8x")
        width_mult = int(sub_config.get("width_mult", 1))
        dropout_prob = float(sub_config.get("dropout_prob", 0.0))
        self.clone = create_substitute(
            arch=arch,
            num_classes=num_classes,
            input_channels=input_shape[0],
            width_mult=width_mult,
            dropout_prob=dropout_prob,
        ).to(device)
        
        # [UNIFIED] Standard Student Optimizer (SGD)
        # Config-driven LR to support SET-A (0.01) vs SET-B (0.1)
        # Default to 0.1 if not specified (Data-Free Standard)
        opt_config = sub_config.get("optimizer", {})
        lr = float(opt_config.get("lr", 0.1))
        self.c_opt = optim.SGD(
            self.clone.parameters(),
            lr=lr,
            momentum=float(opt_config.get("momentum", 0.9)),
            weight_decay=float(opt_config.get("weight_decay", 5e-4))
        )

        if self.lr_schedule == "cosine":
            max_budget = int(self.state.metadata.get("max_budget", 20_000_000))
            queries_per_iter = max(
                1,
                int(self.batch_size) * int(self.n_g * (self.grad_approx_m + 1) + self.n_c),
            )
            approx_iters = max(1, int(math.ceil(max_budget / float(queries_per_iter))))
            self.g_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.g_opt, T_max=approx_iters)
            self.c_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.c_opt, T_max=approx_iters)
        
    def run(self, ctx: BenchmarkContext) -> None:
        device = self.state.metadata.get("device", "cpu")
        # [FEATURE] Clean progress bar for Data-Free (Query Progress Only)
        pbar = self._create_progress_bar(ctx.budget_remaining, "[MAZE] Extracting")
        
        # Default benchmark scheduler: query milestones.
        if self.lr_schedule == "multistep":
            max_budget = int(self.state.metadata.get("max_budget", 20_000_000))
            milestones = [int(max_budget * p) for p in [0.1, 0.3, 0.5]]
            gamma = 0.3 # Standard decay factor
            self.milestones = sorted(milestones)
            self.current_milestone_idx = 0

        while ctx.budget_remaining > 0:
            # Check for LR decay
            if self.lr_schedule == "multistep":
                current_queries = self.state.query_count
                if self.current_milestone_idx < len(self.milestones):
                    if current_queries >= self.milestones[self.current_milestone_idx]:
                        for param_group in self.g_opt.param_groups:
                            param_group['lr'] *= gamma
                        for param_group in self.c_opt.param_groups:
                            param_group['lr'] *= gamma
                        self.logger.info(f"[MAZE] Decayed LR at {current_queries} queries (Milestone {self.milestones[self.current_milestone_idx]})")
                        self.current_milestone_idx += 1

            total_queries = 1 + self.grad_approx_m
            max_g_batch = min(self.batch_size, ctx.budget_remaining // total_queries)
            max_s_batch = min(self.batch_size, ctx.budget_remaining)
            if max_g_batch <= 0 and max_s_batch <= 0:
                break
            # 1. Generator Update Phase (Disagreement Maximization)
            for _ in range(self.n_g):
                total_queries = 1 + self.grad_approx_m
                batch = min(self.batch_size, ctx.budget_remaining // total_queries)
                if batch <= 0:
                    break
                self.generator.train(); self.clone.eval()
                z = torch.randn(batch, self.noise_dim, device=device)
                
                # Base output for estimation
                x_base = self.generator(z)
                x_base_01 = torch.clamp((x_base + 1.0) / 2.0, 0.0, 1.0)
                y_c_base = F.softmax(self.clone(self._normalize(x_base)), dim=1)
                
                # Objective LG: -KL(yT || yC) to maximize disagreement
                # loss_base depends on victim output; computed after batching oracle queries below.
                
                # Zeroth-Order Gradient Estimation (Eq. 11)
                grad_est_x = torch.zeros_like(x_base)
                d = x_base[0].numel()

                x_pert_01_list = []
                u_list = []
                for _ in range(self.grad_approx_m):
                    u = torch.randn_like(x_base)
                    u /= (torch.norm(u.view(batch, -1), dim=1).view(-1, 1, 1, 1) + 1e-8)
                    x_pert = torch.clamp(x_base + self.epsilon * u, -1.0, 1.0)
                    x_pert_01 = torch.clamp((x_pert + 1.0) / 2.0, 0.0, 1.0)

                    x_pert_01_list.append(x_pert_01)
                    u_list.append(u)

                # Batch oracle queries: base + perturbed (same total images queried).
                x_query = torch.cat([x_base_01] + x_pert_01_list, dim=0)
                y_t_all = ctx.query(x_query).y.to(device)
                y_t_base = y_t_all[:batch]
                y_t_pert_all = y_t_all[batch:].view(self.grad_approx_m, batch, -1)

                loss_base = -F.kl_div(torch.log(y_c_base + 1e-10), y_t_base, reduction='none').sum(dim=1)

                for j in range(self.grad_approx_m):
                    u = u_list[j]
                    x_pert = torch.clamp(x_base + self.epsilon * u, -1.0, 1.0)
                    y_c_pert = F.softmax(self.clone(self._normalize(x_pert)), dim=1)
                    y_t_pert = y_t_pert_all[j]
                    loss_pert = -F.kl_div(torch.log(y_c_pert + 1e-10), y_t_pert, reduction='none').sum(dim=1)
                    grad_est_x += (d / self.grad_approx_m) * ((loss_pert - loss_base).view(-1, 1, 1, 1) / self.epsilon) * u
                
                self.g_opt.zero_grad()
                # Chain rule: dLG/dThetaG = dLG/dx * dx/dThetaG
                x_base.backward(grad_est_x)
                self.g_opt.step()
                pbar.update(batch * (1 + self.grad_approx_m))

            # 2. Clone Update Phase (Disagreement Minimization)
            for _ in range(self.n_c):
                batch = min(self.batch_size, ctx.budget_remaining)
                if batch <= 0:
                    break
                self.generator.eval(); self.clone.train()
                z = torch.randn(batch, self.noise_dim, device=device)
                x_gen = self.generator(z).detach()
                y_t = ctx.query(torch.clamp((x_gen + 1.0) / 2.0, 0.0, 1.0)).y.to(device)

                # Avoid BatchNorm crashes on tiny final batches (e.g., batch=1)
                # while still consuming the remaining query budget.
                clone_was_training = self.clone.training
                if batch < 2:
                    self.clone.eval()
                
                # Minimize KL Divergence (Eq. 4)
                self.c_opt.zero_grad()
                y_c = F.log_softmax(self.clone(self._normalize(x_gen)), dim=1)
                loss = F.kl_div(y_c, y_t, reduction='batchmean')
                loss.backward(); self.c_opt.step()

                if clone_was_training:
                    self.clone.train()
                
                # Experience Replay Storage
                self._append_replay(x_gen, y_t)
                pbar.update(batch)

            # 3. Experience Replay Phase
            if self.replay_x:
                self.clone.train()
                for _ in range(self.n_r):
                    x_r, y_r = self._sample_replay()
                    self.c_opt.zero_grad()
                    y_c_r = F.log_softmax(self.clone(self._normalize(x_r.to(device))), dim=1)
                    F.kl_div(y_c_r, y_r.to(device), reduction='batchmean').backward()
                    self.c_opt.step()

            if self.lr_schedule == "cosine":
                if self.g_scheduler is not None:
                    self.g_scheduler.step()
                if self.c_scheduler is not None:
                    self.c_scheduler.step()

        self.state.attack_state["substitute"] = self.clone
        pbar.close()

    def observe(
        self,
        query_batch: QueryBatch,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        # MAZE is implemented as a self-contained AttackRunner (Track B) with all
        # logic inside run(). This method exists for interface consistency.
        return None

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map [-1, 1] to [0, 1] then apply standard normalization."""
        x_01 = (x + 1.0) / 2.0
        # In this benchmark, standard normalization is removed/identity as per AGENTS.md
        return x_01

    def _append_replay(self, x: torch.Tensor, y: torch.Tensor):
        self.replay_x.append(x.cpu())
        self.replay_y.append(y.cpu())

    def _sample_replay(self) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = np.random.choice(len(self.replay_x))
        return self.replay_x[idx], self.replay_y[idx]
