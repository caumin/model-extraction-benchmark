"""MAZE (Model Stealing via Zeroth-Order Gradient Estimation) attack implementation."""

from typing import Dict, Any, List, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.models.substitute_factory import create_substitute


class _OfficialMAZEConv3Generator(nn.Module):
    """Official MAZE conv3_gen architecture.

    Mirrors `official_repo_clones/maze/src/models/conv3_gen.py` while keeping
    the `return_pre_tanh` interface used by mebench MAZE runner.
    """

    def __init__(
        self,
        z_dim: int,
        out_channels: int = 3,
        start_dim: int = 8,
        output_size: int = 32,
    ) -> None:
        super().__init__()
        self.output_size = int(output_size)
        self.linear = nn.Linear(int(z_dim), 128 * int(start_dim) ** 2)
        self.bn0 = nn.BatchNorm2d(128)

        self.up1 = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.up2 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(64, int(out_channels), 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(int(out_channels), affine=True)
        self.tanh = nn.Tanh()
        self._start_dim = int(start_dim)

    def forward(
        self, z: torch.Tensor, return_pre_tanh: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.linear(z)
        x = x.view(-1, 128, self._start_dim, self._start_dim)
        x = self.bn0(x)

        x = self.up1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.up2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        pre_tanh = self.bn3(x)
        out = self.tanh(pre_tanh)

        if out.shape[-1] != self.output_size:
            out = F.interpolate(
                out,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )
            pre_tanh = F.interpolate(
                pre_tanh,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )

        if return_pre_tanh:
            return pre_tanh, out
        return out


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
        # Official MAZE implementation uses cosine annealing with SGD optimizers.
        self.lr_schedule = str(config.get("lr_schedule", "cosine")).lower()
        if self.lr_schedule not in {"multistep", "cosine"}:
            raise ValueError(f"MAZE lr_schedule must be 'multistep' or 'cosine', got {self.lr_schedule!r}")
        eval_interval_raw = int(config.get("eval_interval_queries", 100_000))
        self.eval_interval_queries = eval_interval_raw if eval_interval_raw > 0 else 0
        self._next_eval_query = self.eval_interval_queries
        self._periodic_eval_done: set[int] = set()
        
        self.generator = None
        self.clone = None
        self._eval_substitute = None
        self.g_scheduler = None
        self.c_scheduler = None
        self.replay_buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._cached_clone_x: Optional[torch.Tensor] = None
        self._cached_clone_y: Optional[torch.Tensor] = None
        
        self._initialize_models(state)

    def _initialize_models(self, state: BenchmarkState):
        device = state.metadata.get("device", "cpu")
        input_shape = state.metadata.get("input_shape", (3, 32, 32))
        num_classes = state.metadata.get("num_classes", 10)
        
        # Generator: official MAZE conv3_gen architecture.
        self.generator = _OfficialMAZEConv3Generator(
            z_dim=self.noise_dim,
            out_channels=int(input_shape[0]),
            start_dim=max(1, int(input_shape[1]) // 4),
            output_size=int(input_shape[1]),
        ).to(device)
        
        # Paper (Section 5.1): SGD optimizer with lr=1e-4 for G.
        g_lr = float(self.config.get("generator_lr", 1e-4))
        g_momentum = float(self.config.get("generator_momentum", 0.9))
        g_weight_decay = float(self.config.get("generator_weight_decay", 5e-4))
        self.g_opt = optim.SGD(
            self.generator.parameters(),
            lr=g_lr,
            momentum=g_momentum,
            weight_decay=g_weight_decay,
        )
        
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
            # Official MAZE query accounting per outer iteration:
            # batch_size * ((iter_clone - 1) + (1 + ndirs) * iter_gen)
            queries_per_iter = max(
                1,
                int(self.batch_size)
                * int(self.n_g * (self.grad_approx_m + 1) + max(0, self.n_c - 1)),
            )
            approx_iters = max(1, int(math.ceil(max_budget / float(queries_per_iter))))
            self.g_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.g_opt, T_max=approx_iters)
            self.c_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.c_opt, T_max=approx_iters)

    def _get_substitute_for_eval(self) -> nn.Module:
        return self.clone

    def _publish_substitute(self) -> None:
        if self.clone is None:
            return
        self.state.attack_state["substitute"] = self._get_substitute_for_eval()

    def _maybe_periodic_eval(self, device: str) -> None:
        if self.eval_interval_queries <= 0:
            return
        if self.victim is None:
            return
        current_queries = int(self.state.query_count)
        if current_queries < int(self._next_eval_query):
            return
        if current_queries in self._periodic_eval_done:
            return
        substitute = self.state.attack_state.get("substitute")
        if substitute is None:
            return

        self._evaluate_current_substitute(
            substitute,
            device,
            track="track_b",
            query_count=current_queries,
        )
        self._periodic_eval_done.add(current_queries)
        while self._next_eval_query <= current_queries:
            self._next_eval_query += self.eval_interval_queries
        
    def run(self, ctx: BenchmarkContext) -> None:
        self.victim = ctx.oracle.model
        device = self.state.metadata.get("device", "cpu")
        # [FEATURE] Clean progress bar for Data-Free (Query Progress Only)
        pbar = self._create_progress_bar(ctx.budget_remaining, "[MAZE] Extracting")
        self._publish_substitute()
        
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
                # Official MAZE keeps both G and S in train mode during the loop.
                self.generator.train(); self.clone.train()
                z = torch.randn(batch, self.noise_dim, device=device)

                # Official MAZE perturbs generator pre-tanh activations.
                pre_tanh, x_base = self.generator(z, return_pre_tanh=True)
                x_base_query = self._query_scale(x_base)
                y_c_base = self._clone_probs_eval(x_base)

                # Objective LG: -KL(yT || yC) to maximize disagreement
                # loss_base depends on victim output; computed after batching oracle queries below.

                # Zeroth-Order Gradient Estimation (Eq. 11)
                grad_est_x = torch.zeros_like(pre_tanh)
                d = int(pre_tanh[0].numel())

                x_pert_01_list = []
                u_list = []
                for _ in range(self.grad_approx_m):
                    u = torch.randn_like(pre_tanh)
                    u /= (torch.norm(u.view(batch, -1), dim=1).view(-1, 1, 1, 1) + 1e-8)
                    x_pert = torch.tanh(pre_tanh + self.epsilon * u)
                    x_pert_query = self._query_scale(x_pert)

                    x_pert_01_list.append(x_pert_query)
                    u_list.append(u)

                # Batch oracle queries: base + perturbed (same total images queried).
                x_query = torch.cat([x_base_query] + x_pert_01_list, dim=0).detach()
                y_t_all = ctx.query(x_query).y
                if y_t_all.device != x_base.device:
                    y_t_all = y_t_all.to(x_base.device)
                y_t_base = y_t_all[:batch]
                y_t_pert_all = y_t_all[batch:].view(self.grad_approx_m, batch, -1)

                # Reuse base generator samples for the first clone update step
                # (aligns MAZE paper budget accounting intent with iter_clone-1 fresh queries).
                self._cached_clone_x = x_base.detach()
                self._cached_clone_y = y_t_base.detach()

                loss_base = -F.kl_div(torch.log(y_c_base + 1e-10), y_t_base, reduction='none').sum(dim=1)

                for j in range(self.grad_approx_m):
                    u = u_list[j]
                    x_pert = torch.tanh(pre_tanh + self.epsilon * u)
                    y_c_pert = self._clone_probs_eval(x_pert)
                    y_t_pert = y_t_pert_all[j]
                    loss_pert = -F.kl_div(torch.log(y_c_pert + 1e-10), y_t_pert, reduction='none').sum(dim=1)
                    grad_est_x += (d / self.grad_approx_m) * ((loss_pert - loss_base).view(-1, 1, 1, 1) / self.epsilon) * u

                # Official implementation averages zeroth-order estimate over batch.
                grad_est_x /= float(max(1, batch))

                self.g_opt.zero_grad()
                # Chain rule: dLG/dThetaG = dLG/dx * dx/dThetaG
                pre_tanh.backward(grad_est_x)
                self.g_opt.step()
                pbar.update(batch * (1 + self.grad_approx_m))

            # 2. Clone Update Phase (Disagreement Minimization)
            replay_x: Optional[torch.Tensor] = None
            replay_y: Optional[torch.Tensor] = None
            for c_idx in range(self.n_c):
                use_cached = bool(c_idx == 0 and self._cached_clone_x is not None and self._cached_clone_y is not None)
                if use_cached:
                    x_gen = self._cached_clone_x
                    y_t = self._cached_clone_y
                    batch = int(x_gen.size(0))
                else:
                    batch = min(self.batch_size, ctx.budget_remaining)
                    if batch <= 0:
                        break
                    z = torch.randn(batch, self.noise_dim, device=device)
                    x_gen = self.generator(z).detach()
                    y_t = ctx.query(self._query_scale(x_gen)).y
                    if y_t.device != x_gen.device:
                        y_t = y_t.to(x_gen.device)

                # Official MAZE samples clone-phase queries with G still in train mode.
                self.generator.train(); self.clone.train()

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

                # Official MAZE stores one replay batch per outer iteration
                # (after clone loop), using the latest (x, T(x)).
                replay_x = x_gen
                replay_y = y_t
                if not use_cached:
                    pbar.update(batch)

            if replay_x is not None and replay_y is not None:
                self._append_replay(replay_x, replay_y)

            self._cached_clone_x = None
            self._cached_clone_y = None

            # 3. Experience Replay Phase
            if self.replay_buffer:
                self.clone.train()
                replay_loader = torch.utils.data.DataLoader(
                    self.replay_buffer,
                    batch_size=self.batch_size,
                    shuffle=True,
                )
                replay_iter = iter(replay_loader)
                for _ in range(self.n_r):
                    try:
                        x_r, y_r = next(replay_iter)
                    except StopIteration:
                        replay_iter = iter(replay_loader)
                        x_r, y_r = next(replay_iter)

                    if int(x_r.size(0)) < int(self.batch_size):
                        break

                    self.c_opt.zero_grad()
                    y_c_r = F.log_softmax(self.clone(self._normalize(x_r.to(device))), dim=1)
                    F.kl_div(y_c_r, y_r.to(device), reduction='batchmean').backward()
                    self.c_opt.step()

            if self.lr_schedule == "cosine":
                if self.g_scheduler is not None:
                    self.g_scheduler.step()
                if self.c_scheduler is not None:
                    self.c_scheduler.step()

            self._publish_substitute()
            self._maybe_periodic_eval(device)

        self._publish_substitute()
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
        """Prepare clone input in fixed tanh scale."""
        return torch.clamp(x, -1.0, 1.0)

    def _clone_probs_eval(self, x: torch.Tensor) -> torch.Tensor:
        """Clone inference for generator-phase zeroth-order estimation."""
        x_in = self._normalize(x).contiguous()
        with torch.no_grad():
            logits = self.clone(x_in)
        return F.softmax(logits.float(), dim=1)

    def _query_scale(self, x_tanh: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x_tanh, -1.0, 1.0)

    def _append_replay(self, x: torch.Tensor, y: torch.Tensor):
        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu()
        self.replay_buffer.extend(
            (x_cpu[i], y_cpu[i]) for i in range(int(x_cpu.size(0)))
        )
