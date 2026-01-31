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
        
        # Student: ResNet-18-8x per Section 4.1
        self.student = create_substitute(arch="resnet18-8x", num_classes=state.metadata.get("num_classes", 10)).to(device)
        self.s_opt = optim.SGD(self.student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    def _recover_logits(self, probs: torch.Tensor) -> torch.Tensor:
        """Mean Correction for Logit Recovery (Section 3.2).
        Approximate logit l_i = log(p_i) - 1/K * sum(log(p_j)).
        """
        log_p = torch.log(probs + 1e-10)
        return log_p - log_p.mean(dim=1, keepdim=True)

    def run(self, ctx: BenchmarkContext) -> None:
        device = self.state.metadata.get("device", "cpu")
        pbar = tqdm(total=ctx.budget_remaining, desc="[DFME] Extracting")

        while ctx.budget_remaining > 0:
            # 1. Generator Update (Disagreement Maximization)
            for _ in range(self.n_g):
                if ctx.budget_remaining < self.batch_size * (1 + self.m): break
                self.generator.train(); self.student.eval()
                z = torch.randn(self.batch_size, 100, device=device)
                
                # Forward Difference for Gradient Estimation (Eq. 6)
                pre_tanh, x = self.generator(z, return_pre_tanh=True)
                v_out = self._recover_logits(ctx.oracle.query(x).y.to(device))
                s_out = self.student(x)
                loss_base = torch.norm(v_out - s_out, p=1, dim=1) # L1 Loss (Eq. 5)
                
                # Estimating Gradient
                grad_est = torch.zeros_like(pre_tanh)
                for _ in range(self.m):
                    u = torch.randn_like(pre_tanh); u /= (torch.norm(u.view(self.batch_size, -1), dim=1).view(-1, 1, 1, 1) + 1e-8)
                    x_pert = torch.tanh(pre_tanh + self.epsilon * u)
                    v_pert = self._recover_logits(ctx.oracle.query(x_pert).y.to(device))
                    s_pert = self.student(x_pert)
                    loss_pert = torch.norm(v_pert - s_pert, p=1, dim=1)
                    grad_est += (loss_pert - loss_base).view(-1, 1, 1, 1) * u
                
                self.g_opt.zero_grad()
                # Maximize L1 Disagreement (Gradient Ascent)
                pre_tanh.backward(- (grad_est / (self.m * self.epsilon)))
                self.g_opt.step()
                pbar.update(self.batch_size * (1 + self.m))

            # 2. Student Update (Disagreement Minimization)
            for _ in range(self.n_s):
                if ctx.budget_remaining < self.batch_size: break
                self.generator.eval(); self.student.train()
                z = torch.randn(self.batch_size, 100, device=device)
                x = self.generator(z).detach()
                v_out = self._recover_logits(ctx.oracle.query(x).y.to(device))
                
                self.s_opt.zero_grad()
                loss = F.l1_loss(self.student(x), v_out)
                loss.backward(); self.s_opt.step()
                pbar.update(self.batch_size)
        
        self.state.attack_state["substitute"] = self.student
        pbar.close()
