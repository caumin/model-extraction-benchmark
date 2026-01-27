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
        # Official settings commonly use clone lr=0.1 (SGD).
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
        default_diversity = 500.0 if self.num_classes == 10 else 100.0
        self.diversity_weight = float(config.get("diversity_weight", default_diversity))
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

    def _initialize_state(self, state: BenchmarkState) -> None:
        state.attack_state["step"] = 0

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
        z = torch.randn(k, self.noise_dim, device=device)
        with torch.no_grad():
            x = self.generator(z)
        meta = {"generator_step": state.attack_state["step"], "synthetic": True}
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
        x_fake = query_batch.x

        real_x = self._next_proxy_batch(device)

        self._train_discriminator(real_x, x_fake)
        self._train_generator(x_fake)
        self._train_clone(x_fake, hard_labels)

        if self.clone_scheduler is not None:
            self.clone_scheduler.step()

        state.attack_state["step"] += 1
        state.attack_state["substitute"] = self.clone

    def _next_proxy_batch(self, device: str) -> torch.Tensor:
        try:
            x_real, _ = next(self.proxy_iter)
        except StopIteration:
            self.proxy_iter = iter(self.proxy_loader)
            x_real, _ = next(self.proxy_iter)
        return x_real.to(device)

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
        self.generator_optimizer.zero_grad()
        fake_logits = self.discriminator(fake_x)
        real_labels = torch.ones_like(fake_logits)
        adv_loss = F.binary_cross_entropy_with_logits(fake_logits, real_labels)

        with torch.no_grad():
            victim_config = self.state.metadata.get("victim_config", {})
            normalization = victim_config.get("normalization")
            if normalization is None:
                normalization = {"mean": [0.0], "std": [1.0]}
            norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(fake_x.device)
            norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(fake_x.device)
            def _norm(x):
                if x.min() < -0.1:
                    x = x * 0.5 + 0.5
                return (x - norm_mean) / norm_std
            
            clone_logits = self.clone(_norm(fake_x))
        probs = F.softmax(clone_logits, dim=1)
        # Paper DFMS-HL: class-diversity loss uses entropy of the batch-mean
        # class distribution alpha (not mean of per-sample entropies).
        alpha = probs.mean(dim=0)
        class_div = torch.sum(alpha * torch.log(alpha + 1e-6))
        loss = adv_loss + self.diversity_weight * class_div
        loss.backward()
        self.generator_optimizer.step()

    def _pretrain_gan(self, device: str) -> None:
        for _ in range(self.pretrain_steps):
            real_x = self._next_proxy_batch(device)
            z = torch.randn(real_x.size(0), self.noise_dim, device=device)
            fake_x = self.generator(z)
            self._train_discriminator(real_x, fake_x)
            self._train_generator(fake_x)

    def _train_clone(self, x_fake: torch.Tensor, hard_labels: torch.Tensor) -> None:
        victim_config = self.state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization")
        if normalization is None:
            normalization = {"mean": [0.0], "std": [1.0]}
        norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(x_fake.device)
        norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(x_fake.device)
        
        # DFMS generates in [-1, 1] usually (check GAN implementation)
        # Assuming [0, 1] for proxy but [-1, 1] for generator.
        # Let's normalize consistent with Oracle fix.
        def _norm(x):
            if x.min() < -0.1:
                x = x * 0.5 + 0.5
            return (x - norm_mean) / norm_std

        self.clone_optimizer.zero_grad()
        logits = self.clone(_norm(x_fake))
        loss = F.cross_entropy(logits, hard_labels)
        loss.backward()
        self.clone_optimizer.step()
