"""ES Attack (Estimate & Synthesize) implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from mebench.attackers.base import BaseAttack
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.models.gan import DCGANGenerator
from mebench.models.substitute_factory import create_substitute


class ESAttack(BaseAttack):
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
        self.mode_seeking_weight = float(config.get("mode_seeking_weight", 0.1))
        self.use_opt_augment = bool(config.get("use_opt_augment", True))
        self.use_class_conditional = bool(config.get("use_class_conditional", True))
        self.acgan_weight = float(config.get("acgan_weight", 1.0))

        self.student: nn.Module | None = None
        self.generator: nn.Module | None = None
        self.student_optimizer: optim.Optimizer | None = None
        self.generator_optimizer: optim.Optimizer | None = None
        self.syn_data: torch.Tensor | None = None

        self._initialize_state(state)

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
            self.generator = DCGANGenerator(
                noise_dim=self.noise_dim,
                output_channels=int(self.config.get("output_channels", input_shape[0])),
                base_channels=self.base_channels,
                num_classes=self.num_classes if self.use_class_conditional else None,
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

    def propose(self, k: int, state: BenchmarkState) -> QueryBatch:
        self._init_models(state)
        device = state.metadata.get("device", "cpu")
        
        # [FIX] ESAttack Normalization: Generator outputs [-1, 1], but Benchmark expects [0, 1].
        # We explicitly convert before returning query batch.
        
        if self.synthesis_mode == "opt_syn":
            x, indices = self._sample_syn_batch(k)
            # Synthesis-before-query: optimize selected synthetic batch before querying.
            self._optimize_syn_batch(indices, device)
            x = self.syn_data[indices].to(device)
            
            # Convert [-1, 1] -> [0, 1]
            x_query = x * 0.5 + 0.5
            
            meta = {"synthetic": True, "mode": "opt_syn", "indices": indices.cpu()}
            return QueryBatch(x=x_query, meta=meta)

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
        
        # Convert [-1, 1] -> [0, 1]
        x_query = x_raw * 0.5 + 0.5
        return QueryBatch(x=x_query, meta=meta)

    def observe(
        self,
        query_batch: QueryBatch,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        if self.student is None:
            return

        device = query_batch.x.device
        if oracle_output.kind == "soft_prob":
            victim_probs = oracle_output.y.to(device)
        else:
            victim_probs = F.one_hot(oracle_output.y, num_classes=self.num_classes).float().to(device)

        self._train_student(query_batch.x, victim_probs)

        if self.synthesis_mode == "dnn_syn" and self.generator is not None:
            self._train_generator(query_batch.meta.get("z"), query_batch.meta.get("y_g"), device)
            # Refresh pool if needed (omitted for dnn_syn as it samples fresh z)
        elif self.synthesis_mode == "opt_syn":
            pass

        state.attack_state["step"] += 1
        state.attack_state["substitute"] = self.student

    def _train_student(self, x: torch.Tensor, victim_probs: torch.Tensor) -> None:
        victim_config = self.state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization")
        if normalization is None:
            normalization = {"mean": [0.0], "std": [1.0]}
        norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(x.device)
        norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(x.device)
        
        # [FIX] Removed heuristic check. Input x is guaranteed to be [0, 1] by propose()
        def _norm(img):
            return (img - norm_mean) / norm_std

        for _ in range(self.student_epochs):
            self.student_optimizer.zero_grad()
            logits = self.student(_norm(x))
            loss = F.kl_div(
                F.log_softmax(logits, dim=1),
                victim_probs,
                reduction="batchmean",
            )
            loss.backward()
            self.student_optimizer.step()

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
                # But here we are given one batch 'z_cpu' from propose() usually.
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
                # ACGAN style: maximize prob of specific class y
                # We want to MINIMIZE CrossEntropy(S(G(z)), y) 
                # wait, if S is fixed, and we want G(z) to be classified as y by S.
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
            
            # Let's use L1 distance for images and z
            lz = torch.mean(torch.abs(x_gen_raw_1 - x_gen_raw_2).view(z1.size(0), -1), dim=1)
            dz = torch.mean(torch.abs(z1 - z2).view(z1.size(0), -1), dim=1) + 1e-8
            
            # We want to MAXIMIZE lz / dz. So MINIMIZE - (lz / dz)
            l_ms = - torch.mean(lz / dz)
            
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
        
        # [FIX] OPT-SYN Objective: Match a random hard label
        # Paper: "randomly assign a label y to each data... optimize x to minimize the cross-entropy loss"
        target_labels = torch.randint(0, self.num_classes, (x.size(0),), device=device)
        
        augmenter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=15, translate=(0.05, 0.05)),
            ]
        )
        
        # Student normalization
        victim_config = self.state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization")
        if normalization is None:
            normalization = {"mean": [0.0], "std": [1.0]}
        norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(device)
        norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(device)

        for _ in range(self.opt_steps):
            optimizer.zero_grad()
            
            # Ensure computation graph flows from 'x'
            if self.use_opt_augment:
                x_01 = x * 0.5 + 0.5
                x_aug = augmenter(x_01)
                x_input_raw = (x_aug - 0.5) / 0.5 # back to [-1, 1]
                
                noise = torch.randn_like(x_input_raw) * 0.01
                x_input_raw = x_input_raw + noise
            else:
                x_input_raw = x

            # Forward pass: [-1, 1] -> [0, 1] -> Normalized
            x_input_clamped = torch.clamp(x_input_raw, -1.0, 1.0)
            logits = self.student((x_input_clamped * 0.5 + 0.5 - norm_mean) / norm_std)
            
            # [FIX] Loss: Cross Entropy with Hard Targets
            loss = F.cross_entropy(logits, target_labels)
            
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
