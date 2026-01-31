"""ES Attack (Estimate & Synthesize) implementation."""

from typing import Dict, Any, List, Tuple, Optional
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm

from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.models.gan import DCGANGenerator, ACGANGenerator
from mebench.models.substitute_factory import create_substitute


class ESAttack(AttackRunner):
    """Estimate & Synthesize attack with DNN-SYN or OPT-SYN."""

    def __init__(self, config: dict, state: BenchmarkState):
        super().__init__(config, state)

        self.batch_size = int(config.get("batch_size", 128))
        self.student_lr = float(config.get("student_lr", 0.01))
        self.generator_lr = float(config.get("generator_lr", 1e-3))
        self.noise_dim = int(config.get("noise_dim", 100))
        self.num_classes = int(
            state.metadata.get("num_classes")
            or config.get("num_classes")
            or state.metadata.get("dataset_config", {}).get("num_classes", 10)
        )
        self.base_channels = int(config.get("base_channels", 64))
        self.synthesis_mode = config.get("synthesis_mode", "dnn_syn")
        self.opt_steps = int(config.get("opt_steps", 30))
        self.opt_lr = float(config.get("opt_lr", 0.01))
        self.syn_size = int(config.get("syn_size", 256))
        self.student_epochs = int(config.get("student_epochs", 10))
        self.synthesis_steps = int(config.get("synthesis_steps", 1))
        self.mode_seeking_weight = float(config.get("mode_seeking_weight", 1.0))
        self.use_opt_augment = bool(config.get("use_opt_augment", True))
        self.use_class_conditional = bool(config.get("use_class_conditional", True))
        self.acgan_weight = float(config.get("acgan_weight", 1.0))

        self.student: nn.Module | None = None
        self.generator: nn.Module | None = None
        self.student_optimizer: optim.Optimizer | None = None
        self.generator_optimizer: optim.Optimizer | None = None
        self.syn_data: torch.Tensor | None = None

        self._initialize_state(state)

    def run(self, ctx: BenchmarkContext) -> None:
        self.victim = ctx.oracle.model
        device = self.state.metadata.get("device", "cpu")
        total_budget = self.state.budget_remaining
        pbar = tqdm(total=total_budget, desc="[ESAttack] Extracting")
        
        while ctx.budget_remaining > 0:
            step_size = self._default_step_size(ctx)
            x_query, meta = self._select_query_batch(step_size, self.state)
            oracle_output = ctx.query(x_query, meta=meta)
            self._handle_oracle_output(x_query, meta, oracle_output, self.state)
            pbar.update(x_query.size(0))
        pbar.close()

    def _select_query_batch(self, k: int, state: BenchmarkState) -> tuple[torch.Tensor, dict]:
        self._init_models(state)
        device = state.metadata.get("device", "cpu")

        if self.synthesis_mode == "opt_syn":
            x, indices = self._sample_syn_batch(k)
            self._optimize_syn_batch(indices, device)
            x = self.syn_data[indices].to(device)
            x_query = x * 0.5 + 0.5
            meta = {"synthetic": True, "mode": "opt_syn", "indices": indices.cpu()}
            return x_query, meta

        z = torch.randn(k, self.noise_dim, device=device)
        with torch.no_grad():
            if self.use_class_conditional:
                y_g = torch.randint(0, self.num_classes, (k,), device=device)
                x_raw = self.generator(z, y_g)
                meta = {
                    "synthetic": True,
                    "mode": "dnn_syn",
                    "z": z.cpu(),
                    "y_g": y_g.cpu(),
                }
            else:
                x_raw = self.generator(z)
                meta = {"synthetic": True, "mode": "dnn_syn", "z": z.cpu()}

        x_query = x_raw * 0.5 + 0.5
        return x_query, meta

    def _handle_oracle_output(
        self,
        x_query: torch.Tensor,
        meta: dict,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        if self.student is None:
            return

        device = x_query.device
        if oracle_output.kind == "soft_prob":
            victim_probs = oracle_output.y.to(device)
        else:
            victim_probs = F.one_hot(oracle_output.y, num_classes=self.num_classes).float().to(device)

        self._train_student(x_query, victim_probs)

        if self.synthesis_mode == "dnn_syn" and self.generator is not None:
            self._train_generator(meta.get("z"), meta.get("y_g"), device)
        elif self.synthesis_mode == "opt_syn":
            pass

        state.attack_state["step"] += 1
        state.attack_state["substitute"] = self.student

    def _initialize_state(self, state: BenchmarkState) -> None:
        state.attack_state["step"] = 0
        state.attack_state["syn_index"] = 0

    def _init_models(self, state: BenchmarkState) -> None:
        device = state.metadata.get("device", "cpu")
        input_shape = state.metadata.get("input_shape", (3, 32, 32))
        if self.student is None:
            input_shape = state.metadata.get("input_shape", (3, 32, 32))
            arch = self.config.get("student_arch", "resnet18-8x")
            sub_config = state.metadata.get("substitute_config", {})
            opt_params = sub_config.get("optimizer", {})
            
            self.student = create_substitute(
                arch=arch,
                num_classes=self.num_classes,
                input_channels=int(input_shape[0]),
            ).to(device)
            self.student_optimizer = optim.SGD(
                self.student.parameters(),
                lr=float(opt_params.get("lr", self.student_lr)),
                momentum=float(opt_params.get("momentum", 0.9)),
                weight_decay=float(opt_params.get("weight_decay", 5e-4))
            )

        if self.generator is None and self.synthesis_mode == "dnn_syn":
            input_shape = state.metadata.get("input_shape", (3, 32, 32))
            
            # [P0 ARCHITECTURE ENFORCEMENT] ES-Attack requires ACGAN
            if self.use_class_conditional:
                if not hasattr(self, 'generator') or self.generator is None:
                    self.generator = ACGANGenerator(
                        noise_dim=self.noise_dim,
                        output_channels=int(self.config.get("output_channels", input_shape[0])),
                        base_channels=self.base_channels,
                        num_classes=self.num_classes,
                        output_size=int(input_shape[1]),
                        dropout_prob=0.25,  # Paper-mandated dropout
                    ).to(device)
            else:
                if not hasattr(self, 'generator') or self.generator is None:
                    self.generator = DCGANGenerator(
                        noise_dim=self.noise_dim,
                        output_channels=int(self.config.get("output_channels", input_shape[0])),
                        base_channels=self.base_channels,
                        num_classes=None,
                        output_size=int(input_shape[1]),
                    ).to(device)
            self.generator_optimizer = optim.Adam(
                self.generator.parameters(), lr=self.generator_lr
            )

        if self.syn_data is None:
            c, h, w = int(input_shape[0]), int(input_shape[1]), int(input_shape[2])
            self.syn_data = torch.randn(self.syn_size, c, h, w, device=device)
            # Init syn_data in [-1, 1] for optimization, but we will clamp/normalize when using
            self.syn_data = self.syn_data.clamp(-1.0, 1.0)

    def _train_student(self, x: torch.Tensor, victim_probs: torch.Tensor) -> None:
        victim_config = self.state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization")
        if normalization is None:
            normalization = {"mean": [0.0], "std": [1.0]}
        norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(x.device)
        norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(x.device)
        
        # Augmentation logic moved here (Step 4)
        augmenter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=15, translate=(0.05, 0.05)),
            ]
        )
        
        # [FIX] Removed heuristic check. Input x is guaranteed to be [0, 1] by selection.
        def _norm(img):
            return (img - norm_mean) / norm_std

        sub_pbar = tqdm(range(self.student_epochs), desc="[ESAttack] Training Student", leave=False)
        for _ in sub_pbar:
            epoch_loss = 0.0
            
            # Create a dataloader for the query data to support proper batching
            dataset = torch.utils.data.TensorDataset(x, victim_probs)
            # Ensure strict batch size from config, or default 128
            train_bs = min(len(x), 128)
            loader = torch.utils.data.DataLoader(dataset, batch_size=train_bs, shuffle=True)
            
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(x.device), batch_y.to(x.device)
                self.student_optimizer.zero_grad()
                
                if self.use_opt_augment and self.synthesis_mode == "opt_syn":
                     # Apply augmentation to input x which is in [0, 1]
                     x_aug = augmenter(batch_x)
                     # Add noise
                     x_aug = x_aug + torch.randn_like(x_aug) * 0.01
                     x_input = torch.clamp(x_aug, 0.0, 1.0)
                else:
                     x_input = batch_x

                logits = self.student(_norm(x_input))
                loss = F.kl_div(
                    F.log_softmax(logits, dim=1),
                    batch_y,
                    reduction="batchmean",
                )
                loss.backward()
                self.student_optimizer.step()
                epoch_loss += loss.item()
            
            sub_pbar.set_postfix({"Loss": f"{epoch_loss/len(loader):.4f}"})

        self.logger.info(f"ESAttack substitute trained at step {self.state.attack_state['step']}")
        self._evaluate_current_substitute(self.student, x.device)

    def _train_generator(
        self,
        z_cpu: torch.Tensor | None,
        y_g_cpu: torch.Tensor | None,
        device: str,
    ) -> None:
        if self.generator is None:
            return

        for _ in range(self.synthesis_steps):
            if z_cpu is None:
                # Need batch_size pairs of (z, z2) for mode seeking
                # But here we are given one batch 'z_cpu' from selection usually.
                # If dnn_syn, we sample fresh z.
                z1 = torch.randn(self.batch_size, self.noise_dim, device=device)
            else:
                z1 = z_cpu.to(device)
            
            # Sample z2 for mode seeking
            z2 = torch.randn_like(z1)
            
            self.generator_optimizer.zero_grad()
            
            # Generate labels if needed
            if self.use_class_conditional:
                if y_g_cpu is None:
                    y = torch.randint(0, self.num_classes, (z1.size(0),), device=device)
                else:
                    y = y_g_cpu.to(device)
            else:
                y = None
                
            # Forward G
            if y is not None:
                x_gen_raw_1 = self.generator(z1, y)
                x_gen_raw_2 = self.generator(z2, y)
            else:
                x_gen_raw_1 = self.generator(z1)
                x_gen_raw_2 = self.generator(z2)
                
            # Convert [-1, 1] -> [0, 1] for student query
            x_gen_1 = x_gen_raw_1 * 0.5 + 0.5
            x_gen_2 = x_gen_raw_2 * 0.5 + 0.5

            # Prepare student input
            victim_config = self.state.metadata.get("victim_config", {})
            normalization = victim_config.get("normalization")
            if normalization is None:
                normalization = {"mean": [0.0], "std": [1.0]}
            norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(device)
            norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(device)
            
            def _norm(img):
                return (img - norm_mean) / norm_std
            
            # 1. Classification Loss (L_img)
            # Maximize probability of target class 'y'
            # L_img = CE(S(G(z)), y)
            logits_1 = self.student(_norm(x_gen_1))
            
            if y is not None:
                # [P0 FIX] Paper Equation 6 mandates Cross-Entropy, NOT KL Divergence
                # ACGAN style: maximize prob of specific class y
                # We want to MINIMIZE CrossEntropy(S(G(z)), y)
                l_img = F.cross_entropy(logits_1, y)
            else:
                # Unconditional: Maximize confidence of ANY class (entropy minimization)
                # Or just maximize max-prob?
                # Paper says "maximize confidence". So minimize entropy or minimize -max(prob).
                probs = F.softmax(logits_1, dim=1)
                l_img = -probs.max(dim=1).values.mean()

            # 2. Mode Seeking Loss (L_ms)
            # L_ms = d(z1, z2) / d(G(z1), G(z2))
            # We maximize this ratio -> Minimize reciprocal?
            # Paper Eq 5: L_DNN = L_img + lambda * L_ms
            # where L_ms is defined as sum_i [ d(z1, z2) / d(G1, G2) ]
            # Wait, maximizing the ratio means maximizing d(G1, G2) for small d(z1, z2).
            # The formula in paper is likely a maximizing objective or minimizing negative.
            # Usually Mode Seeking Loss is: Minimize L_ms = d(G1, G2) / d(z1, z2) ??
            # Let's check common implementation (e.g. DSGAN).
            # DSGAN: Maximize d(G1, G2) / d(z1, z2).
            # So Loss = - d(G1, G2) / d(z1, z2).
            
            # L_ms = dz / lz
            # We want to MINIMIZE L_ms = dz / lz (according to correction plan)
            # Paper Eq 5: L_DNN = L_img + lambda * L_ms
            
            # Let's use L1 distance for images and z
            lz = torch.mean(torch.abs(x_gen_raw_1 - x_gen_raw_2).view(z1.size(0), -1), dim=1)
            dz = torch.mean(torch.abs(z1 - z2).view(z1.size(0), -1), dim=1)
            
            # Add epsilon to lz to prevent division by zero
            epsilon = 1e-8
            l_ms = torch.mean(dz / (lz + epsilon))
            
            # Total Loss
            loss = l_img + self.mode_seeking_weight * l_ms
            
            loss.backward()
            self.generator_optimizer.step()

    def _sample_syn_batch(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.syn_data is None:
            raise ValueError("Synthetic dataset not initialized")
        idx = torch.randint(0, self.syn_data.size(0), (k,), device=self.syn_data.device)
        return self.syn_data[idx], idx

    def _optimize_syn_batch(self, indices: torch.Tensor, device: str) -> None:
        if self.syn_data is None:
            return
        # x is in [-1, 1] here
        x = self.syn_data[indices].clone().detach().to(device)
        x.requires_grad_(True)
        optimizer = optim.Adam([x], lr=self.opt_lr)
        
        # [FIX] OPT-SYN Objective: Match a Dirichlet distribution
        # Paper: "sample alpha from Gaussian N(0,1), then y ~ Dirichlet(alpha)"
        # Since Dirichlet requires alpha > 0, we use abs(alpha).
        # We sample ONE alpha per batch or per sample? Paper implies per sample variability or per batch.
        # "randomly sample the parameter alpha from a Gaussian distribution"
        
        # We sample a different alpha vector for each sample in the batch to encourage diversity
        # alpha_vec = torch.randn(x.size(0), self.num_classes, device=device).abs()
        # But commonly symmetric Dirichlet is used. Let's assume symmetric alpha sampled from Gaussian.
        
        # Implementation: Sample alpha_val ~ |N(0,1)|, then y ~ Dirichlet(alpha_val * ones)
        alpha_val = torch.randn(x.size(0), 1, device=device).abs()
        dist = torch.distributions.Dirichlet(torch.ones(x.size(0), self.num_classes, device=device) * alpha_val)
        y_target = dist.sample()

        # Student normalization
        victim_config = self.state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization")
        if normalization is None:
            normalization = {"mean": [0.0], "std": [1.0]}
        norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(device)
        norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(device)

        for _ in range(self.opt_steps):
            optimizer.zero_grad()
            
            # Forward pass: [-1, 1] -> [0, 1] -> Normalized
            # Note: No augmentation here as per Correction Plan Step 4
            x_input_clamped = torch.clamp(x, -1.0, 1.0)
            logits = self.student((x_input_clamped * 0.5 + 0.5 - norm_mean) / norm_std)
            
            # [FIX] Loss: KL Divergence with Dirichlet Targets
            loss = F.kl_div(F.log_softmax(logits, dim=1), y_target, reduction='batchmean')
            
            loss.backward()
            optimizer.step()
            
            # Project back to valid range [-1, 1]
            with torch.no_grad():
                x.data.clamp_(-1.0, 1.0)
                
        self.syn_data[indices] = x.detach()

    def _refresh_syn_data(self, device: str) -> None:
        if self.syn_data is None or self.generator is None:
            return
        z = torch.randn(self.syn_size, self.noise_dim, device=device)
        with torch.no_grad():
            if self.use_class_conditional:
                y_g = torch.randint(0, self.num_classes, (self.syn_size,), device=device)
                self.syn_data = self.generator(z, y_g).clamp(-1.0, 1.0)
            else:
                self.syn_data = self.generator(z).clamp(-1.0, 1.0)
