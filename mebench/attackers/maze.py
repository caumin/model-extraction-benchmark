"""MAZE (Model Stealing via Zeroth-Order Gradient Estimation) attack implementation."""

from typing import Dict, Any, List, Tuple, Optional
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
    """MAZE implementation strictly aligned with Kariyappa et al. (2021).
    
    Ref Logic:
    - Loss: KL Divergence (Eq. 4/5).
    - Generator Gradient: Zeroth-order estimation on unit sphere with dimension scaling (Eq. 11).
    - Training Schedule: 1 Generator iteration -> 5 Clone iterations -> 10 Replay iterations.
    """

    def __init__(self, config: dict, state: BenchmarkState):
        super().__init__(config, state)

        self.batch_size = int(config.get("batch_size", 128))
        self.grad_approx_m = int(config.get("grad_approx_m", 10))
        self.epsilon = float(config.get("grad_approx_epsilon", 1e-3))
        self.n_g = int(config.get("n_g_steps", 1))
        self.n_c = int(config.get("n_c_steps", 5))
        self.n_r = int(config.get("n_r_steps", 10))
        self.replay_max = int(config.get("replay_max", 10000))
        self.noise_dim = int(config.get("noise_dim", 100))
        
        self.generator = None
        self.clone = None
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
        self.g_opt = optim.SGD(self.generator.parameters(), lr=1e-4, momentum=0.5)
        
        # Clone: ResNet-18 or similar
        self.clone = create_substitute(
            arch="resnet18-8x", 
            num_classes=num_classes,
            input_channels=input_shape[0]
        ).to(device)
        self.c_opt = optim.SGD(self.clone.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    def run(self, ctx: BenchmarkContext) -> None:
        device = self.state.metadata.get("device", "cpu")
        pbar = tqdm(total=ctx.budget_remaining, desc="[MAZE] Extracting")

        while ctx.budget_remaining > 0:
            # 1. Generator Update Phase (Disagreement Maximization)
            for _ in range(self.n_g):
                if ctx.budget_remaining < self.batch_size * (1 + self.grad_approx_m): break
                self.generator.train(); self.clone.eval()
                z = torch.randn(self.batch_size, self.noise_dim, device=device)
                
                # Base output for estimation
                x_base = self.generator(z)
                y_t_base = ctx.oracle.query((x_base + 1.0) / 2.0).y.to(device) # Victim input in [0, 1]
                y_c_base = F.softmax(self.clone(self._normalize(x_base)), dim=1)
                
                # Objective LG: -KL(yT || yC) to maximize disagreement
                loss_base = -F.kl_div(torch.log(y_c_base + 1e-10), y_t_base, reduction='none').sum(dim=1)
                
                # Zeroth-Order Gradient Estimation (Eq. 11)
                grad_est_x = torch.zeros_like(x_base)
                d = x_base[0].numel()
                for _ in range(self.grad_approx_m):
                    u = torch.randn_like(x_base)
                    u /= (torch.norm(u.view(self.batch_size, -1), dim=1).view(-1, 1, 1, 1) + 1e-8)
                    x_pert = x_base + self.epsilon * u
                    y_t_pert = ctx.oracle.query((x_pert + 1.0) / 2.0).y.to(device)
                    y_c_pert = F.softmax(self.clone(self._normalize(x_pert)), dim=1)
                    loss_pert = -F.kl_div(torch.log(y_c_pert + 1e-10), y_t_pert, reduction='none').sum(dim=1)
                    
                    grad_est_x += (d / self.grad_approx_m) * ((loss_pert - loss_base).view(-1, 1, 1, 1) / self.epsilon) * u
                
                self.g_opt.zero_grad()
                # Chain rule: dLG/dThetaG = dLG/dx * dx/dThetaG
                x_base.backward(grad_est_x)
                self.g_opt.step()
                pbar.update(self.batch_size * (1 + self.grad_approx_m))

            # 2. Clone Update Phase (Disagreement Minimization)
            for _ in range(self.n_c):
                if ctx.budget_remaining < self.batch_size: break
                self.generator.eval(); self.clone.train()
                z = torch.randn(self.batch_size, self.noise_dim, device=device)
                x_gen = self.generator(z).detach()
                y_t = ctx.oracle.query((x_gen + 1.0) / 2.0).y.to(device)
                
                # Minimize KL Divergence (Eq. 4)
                self.c_opt.zero_grad()
                y_c = F.log_softmax(self.clone(self._normalize(x_gen)), dim=1)
                loss = F.kl_div(y_c, y_t, reduction='batchmean')
                loss.backward(); self.c_opt.step()
                
                # Experience Replay Storage
                self._append_replay(x_gen, y_t)
                pbar.update(self.batch_size)

            # 3. Experience Replay Phase
            if self.replay_x:
                self.clone.train()
                for _ in range(self.n_r):
                    x_r, y_r = self._sample_replay()
                    self.c_opt.zero_grad()
                    y_c_r = F.log_softmax(self.clone(self._normalize(x_r.to(device))), dim=1)
                    F.kl_div(y_c_r, y_r.to(device), reduction='batchmean').backward()
                    self.c_opt.step()

        self.state.attack_state["substitute"] = self.clone
        pbar.close()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map [-1, 1] to [0, 1] then apply standard normalization."""
        x_01 = (x + 1.0) / 2.0
        # In this benchmark, standard normalization is removed/identity as per AGENTS.md
        return x_01

    def _append_replay(self, x: torch.Tensor, y: torch.Tensor):
        self.replay_x.append(x.cpu()); self.replay_y.append(y.cpu())
        if len(self.replay_x) * self.batch_size > self.replay_max:
            self.replay_x.pop(0); self.replay_y.pop(0)

    def _sample_replay(self) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = np.random.choice(len(self.replay_x))
        return self.replay_x[idx], self.replay_y[idx]
