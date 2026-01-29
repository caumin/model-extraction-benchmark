"""DFMS-HL (Data-Free Model Stealing with Hard Labels) attack."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mebench.attackers.base import BaseAttack
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.models.gan import DCGANGenerator, DCGANDiscriminator
from mebench.models.substitute_factory import create_substitute
from mebench.data.loaders import create_dataloader


class DFMSHL(BaseAttack):
    """DFMS-HL with proxy data, GAN training, and hard-label cloning."""

    def __init__(self, config: dict, state: BenchmarkState):
        super().__init__(config, state)

        self.batch_size = int(config.get("batch_size", 128))
        self.clone_lr = float(config.get("clone_lr", 0.1))
        self.generator_lr = float(config.get("generator_lr", 2e-4))
        self.discriminator_lr = float(config.get("discriminator_lr", 2e-4))
        self.noise_dim = int(config.get("noise_dim", 100))
        self.num_classes = int(
            state.metadata.get("num_classes")
            or config.get("num_classes")
            or state.metadata.get("dataset_config", {}).get("num_classes", 10)
        )
        self.base_channels = int(config.get("base_channels", 64))
        
        # [STRICT] Paper specifies lambda_div = 500.0 for CIFAR-10.
        self.diversity_weight = float(config.get("diversity_weight", 500.0))
        self.certainty_weight = float(config.get("certainty_weight", 1.0))
        
        self.init_nc = int(config.get("init_nc", 1000))
        self.init_ng = int(config.get("init_ng", 100))
        
        self.pretrain_steps = int(config.get("pretrain_steps", 200))
        self.use_clone_cosine = bool(config.get("use_clone_cosine", True))

        self.generator: nn.Module | None = None
        self.discriminator: nn.Module | None = None
        self.clone: nn.Module | None = None
        self.generator_optimizer: optim.Optimizer | None = None
        self.discriminator_optimizer: optim.Optimizer | None = None
        self.clone_optimizer: optim.Optimizer | None = None
        self.clone_scheduler: optim.lr_scheduler.CosineAnnealingLR | None = None
        self.proxy_loader = None
        self.proxy_iter = None
        self.pretrained = False

        self._initialize_state(state)

    @property
    def phase(self) -> str:
        return self.state.attack_state.get("phase", "unknown")

    def _initialize_state(self, state: BenchmarkState) -> None:
        state.attack_state["step"] = 0
        state.attack_state["phase"] = "init_collect"
        state.attack_state["collected_count"] = 0
        state.attack_state["init_x"] = []
        state.attack_state["init_y"] = []

    def _init_models(self, state: BenchmarkState) -> None:
        device = state.metadata.get("device", "cpu")
        if self.generator is None:
            input_shape = state.metadata.get("input_shape", (3, 32, 32))
            self.generator = DCGANGenerator(
                noise_dim=self.noise_dim,
                output_channels=int(self.config.get("output_channels", input_shape[0])),
                base_channels=self.base_channels,
                num_classes=None,
                output_size=int(input_shape[1]),
                num_upsamples=self.config.get("generator_upsamples"),
            ).to(device)
            self.generator_optimizer = optim.Adam(
                self.generator.parameters(), lr=self.generator_lr, betas=(0.5, 0.999)
            )

        if self.discriminator is None:
            input_shape = state.metadata.get("input_shape", (3, 32, 32))
            self.discriminator = DCGANDiscriminator(
                input_channels=int(self.config.get("output_channels", input_shape[0])),
                base_channels=self.base_channels,
                num_classes=None,
                input_size=int(input_shape[1]),
            ).to(device)
            self.discriminator_optimizer = optim.Adam(
                self.discriminator.parameters(), lr=self.discriminator_lr, betas=(0.5, 0.999)
            )

        if self.clone is None:
            input_shape = state.metadata.get("input_shape", (3, 32, 32))
            arch = self.config.get("clone_arch", "resnet18-8x")
            sub_config = state.metadata.get("substitute_config", {})
            opt_params = sub_config.get("optimizer", {})
            
            self.clone = create_substitute(
                arch=arch,
                num_classes=self.num_classes,
                input_channels=int(input_shape[0]),
            ).to(device)
            self.clone_optimizer = optim.SGD(
                self.clone.parameters(),
                lr=float(opt_params.get("lr", self.clone_lr)),
                momentum=float(opt_params.get("momentum", 0.9)),
                weight_decay=float(opt_params.get("weight_decay", 5e-4))
            )
            if self.use_clone_cosine:
                max_budget = state.metadata.get("max_budget", 1000)
                t_max = max(1, int(max_budget / self.batch_size))
                self.clone_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.clone_optimizer, t_max
                )

        if self.proxy_loader is None:
            proxy_config = self.config.get("attack", {}).get("proxy_dataset")
            if proxy_config is None:
                proxy_config = self.config.get("proxy_dataset")
            if proxy_config is None:
                raise ValueError("DFMS-HL requires proxy_dataset configuration")
            self.proxy_loader = create_dataloader(
                proxy_config,
                batch_size=self.batch_size,
                shuffle=True,
            )
            self.proxy_iter = iter(self.proxy_loader)

        if not self.pretrained and self.pretrain_steps > 0:
            self._pretrain_gan(device)
            self.pretrained = True

    def propose(self, k: int, state: BenchmarkState) -> QueryBatch:
        self._init_models(state)
        device = state.metadata.get("device", "cpu")
        phase = state.attack_state["phase"]

        if phase == "init_collect":
            # 50/50 mixture of proxy images and synthetic images
            k_proxy = k // 2
            k_synth = k - k_proxy
            
            x_proxy = self._next_proxy_batch(device, batch_size=k_proxy)
            
            z = torch.randn(k_synth, self.noise_dim, device=device)
            with torch.no_grad():
                x_synth = self.generator(z) * 0.5 + 0.5
            
            x = torch.cat([x_proxy, x_synth], dim=0)
            meta = {"phase": phase, "k_proxy": k_proxy, "k_synth": k_synth}
            return QueryBatch(x=x, meta=meta)
        
        elif phase == "init_retrain_collect":
            # 100% synthetic from fine-tuned G
            z = torch.randn(k, self.noise_dim, device=device)
            with torch.no_grad():
                x = self.generator(z) * 0.5 + 0.5
            meta = {"phase": phase, "synthetic": True}
            return QueryBatch(x=x, meta=meta)
        
        else:
            # Standard alternating training: 100% synthetic
            z = torch.randn(k, self.noise_dim, device=device)
            with torch.no_grad():
                x_raw = self.generator(z)
            x = x_raw * 0.5 + 0.5
            meta = {"generator_step": state.attack_state["step"], "synthetic": True, "phase": phase}
            return QueryBatch(x=x, meta=meta)

    def observe(
        self,
        query_batch: QueryBatch,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        if self.generator is None or self.discriminator is None or self.clone is None:
            return

        device = query_batch.x.device

        if oracle_output.kind == "hard_top1":
            hard_labels = oracle_output.y
        else:
            hard_labels = torch.argmax(oracle_output.y, dim=1)

        hard_labels = hard_labels.to(device)
        x_query = query_batch.x
        phase = state.attack_state["phase"]

        if phase == "init_collect":
            state.attack_state["init_x"].append(x_query.cpu())
            state.attack_state["init_y"].append(hard_labels.cpu())
            state.attack_state["collected_count"] += x_query.size(0)

            if state.attack_state["collected_count"] >= self.init_nc:
                # Transition to fine-tuning
                x_all = torch.cat(state.attack_state["init_x"], dim=0)[:self.init_nc].to(device)
                y_all = torch.cat(state.attack_state["init_y"], dim=0)[:self.init_nc].to(device)
                
                self._fine_tune_generator(x_all, y_all, self.init_ng)
                
                self._reset_clone()
                state.attack_state["phase"] = "init_retrain_collect"
                state.attack_state["collected_count"] = 0
                # Clear buffers to save memory
                state.attack_state["init_x"] = []
                state.attack_state["init_y"] = []

        elif phase == "init_retrain_collect":
            self._train_clone(x_query, hard_labels)
            state.attack_state["collected_count"] += x_query.size(0)
            
            if state.attack_state["collected_count"] >= self.init_nc:
                state.attack_state["phase"] = "alternate"

        elif phase == "alternate":
            real_x = self._next_proxy_batch(device)
            self._train_discriminator(real_x, x_query)
            self._train_generator(x_query)
            self._train_clone(x_query, hard_labels)

            if self.clone_scheduler is not None:
                self.clone_scheduler.step()

        state.attack_state["step"] += 1
        state.attack_state["substitute"] = self.clone

    def _next_proxy_batch(self, device: str, batch_size: int | None = None) -> torch.Tensor:
        bs = batch_size or self.batch_size
        batches = []
        collected = 0
        while collected < bs:
            try:
                x_real, _ = next(self.proxy_iter)
            except StopIteration:
                self.proxy_iter = iter(self.proxy_loader)
                x_real, _ = next(self.proxy_iter)
            
            needed = bs - collected
            if x_real.size(0) > needed:
                batches.append(x_real[:needed])
                collected += needed
            else:
                batches.append(x_real)
                collected += x_real.size(0)
        
        return torch.cat(batches, dim=0).to(device)

    def _train_discriminator(self, real_x: torch.Tensor, fake_x: torch.Tensor) -> None:
        self.discriminator_optimizer.zero_grad()
        real_logits = self.discriminator(real_x)
        fake_logits = self.discriminator(fake_x.detach())
        real_labels = torch.ones_like(real_logits)
        fake_labels = torch.zeros_like(fake_logits)
        loss_real = F.binary_cross_entropy_with_logits(real_logits, real_labels)
        loss_fake = F.binary_cross_entropy_with_logits(fake_logits, fake_labels)
        loss = loss_real + loss_fake
        loss.backward()
        self.discriminator_optimizer.step()

    def _train_generator(self, fake_x: torch.Tensor) -> None:
        # Regenerate to maintain gradient graph from z to G(z)
        z = torch.randn(self.batch_size, self.noise_dim, device=fake_x.device)
        fake_x_gen_raw = self.generator(z)
        fake_x_gen = fake_x_gen_raw * 0.5 + 0.5  # [-1, 1] -> [0, 1]

        self.generator_optimizer.zero_grad()
        fake_logits = self.discriminator(fake_x_gen)
        real_labels = torch.ones_like(fake_logits)
        adv_loss = F.binary_cross_entropy_with_logits(fake_logits, real_labels)

        victim_config = self.state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization")
        if normalization is None:
            normalization = {"mean": [0.0], "std": [1.0]}
        norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(fake_x.device)
        norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(fake_x.device)
        
        def _norm(x):
            return (x - norm_mean) / norm_std
        
        clone_logits = self.clone(_norm(fake_x_gen))
        probs = F.softmax(clone_logits, dim=1)
        
        # Diversity: entropy of batch-mean distribution alpha.
        alpha = probs.mean(dim=0)
        class_div = torch.sum(alpha * torch.log(alpha + 1e-6))
        
        # Certainty: mean entropy of individual distributions.
        certainty_loss = -torch.sum(probs * torch.log(probs + 1e-6), dim=1).mean()
        
        loss = adv_loss + self.diversity_weight * class_div + self.certainty_weight * certainty_loss
        loss.backward()
        self.generator_optimizer.step()

    def _pretrain_gan(self, device: str) -> None:
        for _ in range(self.pretrain_steps):
            real_x = self._next_proxy_batch(device)
            z = torch.randn(real_x.size(0), self.noise_dim, device=device)
            fake_x = self.generator(z) * 0.5 + 0.5
            
            self._train_discriminator(real_x, fake_x)
            
            z2 = torch.randn(real_x.size(0), self.noise_dim, device=device)
            fake_x_2 = self.generator(z2) * 0.5 + 0.5
            
            self.generator_optimizer.zero_grad()
            fake_logits = self.discriminator(fake_x_2)
            real_labels = torch.ones_like(fake_logits)
            loss_g = F.binary_cross_entropy_with_logits(fake_logits, real_labels)
            loss_g.backward()
            self.generator_optimizer.step()

    def _train_clone(self, x_fake: torch.Tensor, hard_labels: torch.Tensor) -> None:
        victim_config = self.state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization")
        if normalization is None:
            normalization = {"mean": [0.0], "std": [1.0]}
        norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(x_fake.device)
        norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(x_fake.device)
        
        def _norm(x):
            return (x - norm_mean) / norm_std

        # Batching to avoid OOM for large buffers during init
        dataset = torch.utils.data.TensorDataset(x_fake, hard_labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for batch_x, batch_y in loader:
            self.clone_optimizer.zero_grad()
            logits = self.clone(_norm(batch_x))
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            self.clone_optimizer.step()

    def _reset_clone(self) -> None:
        device = self.state.metadata.get("device", "cpu")
        input_shape = self.state.metadata.get("input_shape", (3, 32, 32))
        arch = self.config.get("clone_arch", "resnet18-8x")
        sub_config = self.state.metadata.get("substitute_config", {})
        opt_params = sub_config.get("optimizer", {})

        self.clone = create_substitute(
            arch=arch,
            num_classes=self.num_classes,
            input_channels=int(input_shape[0]),
        ).to(device)
        self.clone_optimizer = optim.SGD(
            self.clone.parameters(),
            lr=float(opt_params.get("lr", self.clone_lr)),
            momentum=float(opt_params.get("momentum", 0.9)),
            weight_decay=float(opt_params.get("weight_decay", 5e-4))
        )
        if self.use_clone_cosine:
            max_budget = self.state.metadata.get("max_budget", 1000)
            t_max = max(1, int(max_budget / self.batch_size))
            self.clone_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.clone_optimizer, t_max
            )

    def _fine_tune_generator(self, x_collected: torch.Tensor, y_collected: torch.Tensor, epochs: int) -> None:
        device = x_collected.device
        # Train clone first on collected data
        print(f"Fine-tuning: Training clone for 50 epochs on {x_collected.size(0)} samples...")
        for _ in range(50): 
            self._train_clone(x_collected, y_collected)
        
        # Then tune G
        print(f"Fine-tuning: Training generator for {epochs} epochs...")
        self.generator.train()
        for i in range(epochs):
            real_x = self._next_proxy_batch(device)
            z = torch.randn(self.batch_size, self.noise_dim, device=device)
            fake_x = self.generator(z) * 0.5 + 0.5
            
            self._train_discriminator(real_x, fake_x)
            self._train_generator(fake_x)
            if (i+1) % 10 == 0 or i == epochs - 1:
                print(f"  Epoch {i+1}/{epochs} done")
