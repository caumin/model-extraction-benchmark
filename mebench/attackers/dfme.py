"""DFME: Data-Free Model Extraction (CVPR 2021)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.models.gan import DFMEGenerator
from mebench.models.substitute_factory import create_substitute
from mebench.utils.scaling import normalize_input_scale


class _ScaledSubstituteWrapper(nn.Module):
    """Apply input scaling before substitute forward for evaluation."""

    def __init__(self, model: nn.Module, input_scale_mode: str) -> None:
        super().__init__()
        self.model = model
        self.input_scale_mode = str(input_scale_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scaled = normalize_input_scale(x, self.input_scale_mode)
        return self.model(x_scaled)

class DFME(AttackRunner):
    """DFME implementation strictly aligned with Truong et al. (2021).
    
    Ref Logic:
    - Loss: L1 distance on Logits (Eq. 5).
    - Logit Recovery: mean correction log(p) - mean(log(p)).
    - G-Update: Zeroth-order gradient estimate with random directions (Eq. 6).
    """

    def __init__(self, config: dict, state: BenchmarkState):
        super().__init__(config, state)
        self.batch_size = int(config.get("batch_size", 256))
        self.n_g = int(config.get("n_g_steps", 1))
        self.n_s = int(config.get("n_s_steps", 5))
        self.epsilon = float(config.get("grad_approx_epsilon", 1e-3))
        self.m = int(config.get("grad_approx_m", 1))
        self.noise_dim = int(config.get("noise_dim", 256))
        self.generator_lr = float(config.get("generator_lr", config.get("lr_G", 1e-4)))
        self.internal_input_scale_mode = str(
            config.get("internal_input_scale_mode", "unit")
        ).strip().lower()
        if self.internal_input_scale_mode not in {"unit", "tanh"}:
            raise ValueError(
                "DFME internal_input_scale_mode must be 'unit' or 'tanh', "
                f"got {self.internal_input_scale_mode!r}"
            )
        self.query_input_scale_mode = str(
            config.get("query_input_scale_mode", "tanh")
        ).strip().lower()
        if self.query_input_scale_mode not in {"unit", "tanh"}:
            raise ValueError(
                "DFME query_input_scale_mode must be 'unit' or 'tanh', "
                f"got {self.query_input_scale_mode!r}"
            )
        eval_interval_raw = int(config.get("eval_interval_queries", 100_000))
        self.eval_interval_queries = eval_interval_raw if eval_interval_raw > 0 else 0
        self._next_eval_query = self.eval_interval_queries
        self._periodic_eval_done: set[int] = set()

        self.student = None
        self.generator = None
        self._eval_substitute = None
        self._initialize_models(state)

    def _initialize_models(self, state: BenchmarkState):
        device = state.metadata.get("device", "cpu")
        input_shape = state.metadata.get("input_shape", (3, 32, 32))
        
        # Generator: Transposed Conv based architecture (GeneratorA in official repo)
        self.generator = DFMEGenerator(
            noise_dim=self.noise_dim,
            output_channels=input_shape[0],
            output_size=input_shape[1],
        ).to(device)
        self.g_opt = optim.Adam(self.generator.parameters(), lr=self.generator_lr)
        
        # Student: honor substitute config if provided
        sub_config = state.metadata.get("substitute_config", {})
        arch = sub_config.get("arch") or self.config.get("student_arch", "resnet18")
        width_mult = int(sub_config.get("width_mult", 1))
        dropout_prob = float(sub_config.get("dropout_prob", 0.0))
        self.student = create_substitute(
            arch=arch,
            num_classes=state.metadata.get("num_classes", 10),
            input_channels=int(input_shape[0]),
            width_mult=width_mult,
            dropout_prob=dropout_prob,
        ).to(device)
        opt_config = dict(sub_config.get("optimizer", {}))
        # Official DFME defaults (train.py): SGD lr=0.1, momentum=0.9, weight_decay=5e-4.
        opt_config.setdefault("name", "sgd")
        opt_config.setdefault("lr", float(self.config.get("student_lr", 0.1)))
        opt_config.setdefault("momentum", 0.9)
        opt_config.setdefault("weight_decay", 5e-4)
        self.s_opt = self._build_optimizer(self.student.parameters(), opt_config)
        
        # [ADDED] LR Schedulers as per DFME paper
        # Generator: MultiStepLR at 10%, 30%, 50% of budget
        # Student: MultiStepLR at 10%, 30%, 50% of budget
        # Since we run in a query loop, we'll step the scheduler based on query count.
        # But PyTorch schedulers expect step() to be called per epoch or iteration.
        # We will wrap them and call step() manually in run().
        
        # NOTE: Scheduler initialization is deferred to run() where we know max_budget.

    def _recover_logits(self, probs: torch.Tensor) -> torch.Tensor:
        """Mean Correction for Logit Recovery (Section 3.2).
        Approximate logit l_i = log(p_i) - 1/K * sum(log(p_j)).
        """
        log_p = torch.log(probs + 1e-10)
        return log_p - log_p.mean(dim=1, keepdim=True)

    def _student_scale(self, x_tanh: torch.Tensor) -> torch.Tensor:
        if self.internal_input_scale_mode == "tanh":
            return torch.clamp(x_tanh, -1.0, 1.0)
        return torch.clamp(x_tanh * 0.5 + 0.5, 0.0, 1.0)

    def _query_scale(self, x_tanh: torch.Tensor) -> torch.Tensor:
        if self.query_input_scale_mode == "tanh":
            return torch.clamp(x_tanh, -1.0, 1.0)
        return torch.clamp(x_tanh * 0.5 + 0.5, 0.0, 1.0)

    def _get_substitute_for_eval(self) -> nn.Module:
        return self.student

    def _publish_substitute(self) -> None:
        if self.student is None:
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
        device = self.state.metadata.get("device", "cpu")
        self.victim = ctx.oracle.model
        # [FEATURE] Clean progress bar for Data-Free (Query Progress Only)
        pbar = self._create_progress_bar(ctx.budget_remaining, "[DFME] Extracting")
        self._publish_substitute()
        
        # Initialize schedulers
        # Milestones: 0.1, 0.3, 0.5 of TOTAL budget.
        # We need total budget.
        max_budget = int(self.state.metadata.get("max_budget", 20_000_000))
        # We calculate iterations approximately.
        # Queries per iteration = (1 + m) * n_g + n_s * 1
        # Approx: (1+1)*1 + 5*1 = 7 queries per loop (if batch=1).
        # Actually it's batch-based.
        # We can't easily map queries to epochs for standard schedulers.
        # But we can just check query count and decay LR manually or use LambdaLR.
        
        # DFME paper: "learning rate is decayed by 0.3 at 10%, 30% and 50% of the training process"
        milestones = [int(max_budget * p) for p in [0.1, 0.3, 0.5]]
        gamma = 0.3
        
        # We will manually decay LR
        self.milestones = sorted(milestones)
        self.current_milestone_idx = 0

        while ctx.budget_remaining > 0:
            # Check for LR decay
            current_queries = self.state.query_count
            if self.current_milestone_idx < len(self.milestones):
                if current_queries >= self.milestones[self.current_milestone_idx]:
                    # Decay LR
                    for param_group in self.g_opt.param_groups:
                        param_group['lr'] *= gamma
                    for param_group in self.s_opt.param_groups:
                        param_group['lr'] *= gamma
                    self.logger.info(f"[DFME] Decayed LR at {current_queries} queries (Milestone {self.milestones[self.current_milestone_idx]})")
                    self.current_milestone_idx += 1

            total_queries = 1 + self.m
            max_g_batch = min(self.batch_size, ctx.budget_remaining // total_queries)
            max_s_batch = min(self.batch_size, ctx.budget_remaining)
            if max_g_batch <= 0 and max_s_batch <= 0:
                break
            # 1. Generator Update (Disagreement Maximization)
            for _ in range(self.n_g):
                total_queries = 1 + self.m
                batch = min(self.batch_size, ctx.budget_remaining // total_queries)
                if batch <= 0:
                    break
                self.generator.train(); self.student.eval()
                z = torch.randn(batch, self.noise_dim, device=device)
                
                # Forward Difference for Gradient Estimation (Eq. 6)
                pre_tanh, x_tanh = self.generator(z, return_pre_tanh=True)
                x_student = self._student_scale(x_tanh)
                x_query = self._query_scale(x_tanh)
                with torch.no_grad():
                    s_out = self.student(x_student)
                # loss_base depends on victim output; computed after querying v_out below.
                
                # Estimating Gradient
                d = pre_tanh.view(pre_tanh.size(0), -1).size(1)
                grad_est = torch.zeros_like(pre_tanh)
                x_pert_query_list = []
                x_pert_student_list = []
                u_list = []
                for _ in range(self.m):
                    u = torch.randn_like(pre_tanh)
                    u /= (torch.norm(u.view(batch, -1), dim=1).view(-1, 1, 1, 1) + 1e-8)
                    x_pert_tanh = torch.tanh(pre_tanh + self.epsilon * u)

                    x_pert_query_list.append(self._query_scale(x_pert_tanh))
                    x_pert_student_list.append(self._student_scale(x_pert_tanh))
                    u_list.append(u)

                # Query victim once for base + perturbed batches (same total images queried).
                x_query_all = torch.cat([x_query] + x_pert_query_list, dim=0)
                v_all = self._recover_logits(ctx.query(x_query_all).y.to(device))
                v_out = v_all[:batch]
                v_pert_all = v_all[batch:].view(self.m, batch, -1)

                self._publish_substitute()
                self._maybe_periodic_eval(device)
                num_classes = int(v_out.size(1))

                # Official path (approximate_gradients.py):
                # loss_values = -F.l1_loss(..., reduction='none').mean(dim=1)
                loss_base = -F.l1_loss(s_out, v_out, reduction="none").mean(dim=1)

                for j in range(self.m):
                    x_pert_j = x_pert_student_list[j]
                    u_j = u_list[j]
                    with torch.no_grad():
                        s_pert = self.student(x_pert_j)
                    v_pert = v_pert_all[j]
                    loss_pert = -F.l1_loss(s_pert, v_pert, reduction="none").mean(dim=1)
                    grad_est += (loss_pert - loss_base).view(-1, 1, 1, 1) * u_j
                
                self.g_opt.zero_grad()
                # Official estimate scaling:
                # (1/eps) * differences * u * dim, averaged over m,
                # then divided by (num_classes * batch_size).
                grad_est_scaled = grad_est * (d / (self.m * self.epsilon))
                grad_est_scaled = grad_est_scaled / float(max(1, num_classes * batch))
                pre_tanh.backward(grad_est_scaled)
                self.g_opt.step()
                pbar.update(batch * (1 + self.m))

            # 2. Student Update (Disagreement Minimization)
            for _ in range(self.n_s):
                batch = min(self.batch_size, ctx.budget_remaining)
                if batch <= 0:
                    break
                self.generator.eval(); self.student.train()
                z = torch.randn(batch, self.noise_dim, device=device)
                x_tanh = self.generator(z).detach()
                x_student = self._student_scale(x_tanh)
                x_query = self._query_scale(x_tanh)
                v_out = self._recover_logits(ctx.query(x_query).y.to(device))

                self._publish_substitute()
                self._maybe_periodic_eval(device)
                
                self.s_opt.zero_grad()
                # Avoid BatchNorm crashes on tiny final batches (e.g., batch=1)
                # while still consuming the remaining query budget.
                student_was_training = self.student.training
                if batch < 2:
                    self.student.eval()

                # [FIX] Align with Official Code (TEMP_DFME/dfme/train.py line 34)
                # Official implementation uses default reduction='mean' for student update.
                loss = F.l1_loss(self.student(x_student), v_out)
                loss.backward(); self.s_opt.step()

                if student_was_training:
                    self.student.train()
                pbar.update(batch)

            self._publish_substitute()
            self._maybe_periodic_eval(device)

        self._publish_substitute()
        pbar.close()

    def observe(
        self,
        query_batch,
        oracle_output,
        state,
    ) -> None:
        # DFME is implemented as a self-contained AttackRunner (Track B) with
        # all query/train logic inside run(). This method exists for interface
        # consistency with the benchmark test suite.
        return None
