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
        
        self.student = None
        self.generator = None
        self._initialize_models(state)

    def _initialize_models(self, state: BenchmarkState):
        device = state.metadata.get("device", "cpu")
        input_shape = state.metadata.get("input_shape", (3, 32, 32))
        
        # Generator: Transposed Conv based architecture (GeneratorA in official repo)
        self.generator = DFMEGenerator(noise_dim=100, output_channels=input_shape[0], output_size=input_shape[1]).to(device)
        self.g_opt = optim.Adam(self.generator.parameters(), lr=5e-4)
        
        # Student: honor substitute config if provided
        sub_config = state.metadata.get("substitute_config", {})
        arch = sub_config.get("arch") or self.config.get("student_arch", "resnet18-8x")
        width_mult = int(sub_config.get("width_mult", 1))
        dropout_prob = float(sub_config.get("dropout_prob", 0.0))
        self.student = create_substitute(
            arch=arch,
            num_classes=state.metadata.get("num_classes", 10),
            input_channels=int(input_shape[0]),
            width_mult=width_mult,
            dropout_prob=dropout_prob,
        ).to(device)
        opt_config = sub_config.get("optimizer", {})
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

    def run(self, ctx: BenchmarkContext) -> None:
        device = self.state.metadata.get("device", "cpu")
        pbar = self._create_progress_bar(ctx.budget_remaining, "[DFME] Extracting")
        
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
                z = torch.randn(batch, 100, device=device)
                
                # Forward Difference for Gradient Estimation (Eq. 6)
                pre_tanh, x_raw = self.generator(z, return_pre_tanh=True)
                # Benchmark contract: oracle inputs are in [0, 1] (no mean/std normalization).
                x = torch.clamp(x_raw * 0.5 + 0.5, 0.0, 1.0)
                v_out = self._recover_logits(ctx.query(x).y.to(device))
                s_out = self.student(x)
                loss_base = torch.norm(v_out - s_out, p=1, dim=1) # L1 Loss (Eq. 5)
                
                # Estimating Gradient
                d = pre_tanh.view(pre_tanh.size(0), -1).size(1)
                grad_est = torch.zeros_like(pre_tanh)
                for _ in range(self.m):
                    u = torch.randn_like(pre_tanh)
                    u /= (torch.norm(u.view(batch, -1), dim=1).view(-1, 1, 1, 1) + 1e-8)
                    x_pert_raw = torch.tanh(pre_tanh + self.epsilon * u)
                    x_pert = torch.clamp(x_pert_raw * 0.5 + 0.5, 0.0, 1.0)
                    v_pert = self._recover_logits(ctx.query(x_pert).y.to(device))
                    s_pert = self.student(x_pert)
                    loss_pert = torch.norm(v_pert - s_pert, p=1, dim=1)
                    grad_est += (loss_pert - loss_base).view(-1, 1, 1, 1) * u
                
                self.g_opt.zero_grad()
                # Maximize L1 Disagreement (Gradient Ascent)
                pre_tanh.backward(- (grad_est * d / (self.m * self.epsilon)))
                self.g_opt.step()
                pbar.update(batch * (1 + self.m))

            # 2. Student Update (Disagreement Minimization)
            for _ in range(self.n_s):
                batch = min(self.batch_size, ctx.budget_remaining)
                if batch <= 0:
                    break
                self.generator.eval(); self.student.train()
                z = torch.randn(batch, 100, device=device)
                x_raw = self.generator(z).detach()
                x = torch.clamp(x_raw * 0.5 + 0.5, 0.0, 1.0)
                v_out = self._recover_logits(ctx.query(x).y.to(device))
                
                self.s_opt.zero_grad()
                # Avoid BatchNorm crashes on tiny final batches (e.g., batch=1)
                # while still consuming the remaining query budget.
                student_was_training = self.student.training
                if batch < 2:
                    self.student.eval()

                loss = F.l1_loss(self.student(x), v_out)
                loss.backward(); self.s_opt.step()

                if student_was_training:
                    self.student.train()
                pbar.update(batch)
        
        self.state.attack_state["substitute"] = self.student
        pbar.close()
