"""GAME (Generative-Based Adaptive Model Extraction) attack."""

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


class GAME(BaseAttack):
    """GAME with shared generator/discriminator and adaptive losses."""

    def __init__(self, config: dict, state: BenchmarkState):
        super().__init__(config, state)

        self.batch_size = int(config.get("batch_size", 128))
        self.student_lr = float(config.get("student_lr", 0.01))
        self.generator_lr = float(config.get("generator_lr", 2e-4))
        self.discriminator_lr = float(config.get("discriminator_lr", 2e-4))
        self.noise_dim = int(config.get("noise_dim", 100))
        self.num_classes = int(
            state.metadata.get("num_classes")
            or config.get("num_classes")
            or state.metadata.get("dataset_config", {}).get("num_classes", 10)
        )
        self.base_channels = int(config.get("base_channels", 64))
        self.acs_strategy = config.get("acs_strategy", "uncertainty")

        self.beta1 = float(config.get("beta1", 1.0))
        self.beta2 = float(config.get("beta2", 1.0))
        self.beta3 = float(config.get("beta3", 1.0))
        self.beta4 = float(config.get("beta4", 1.0))

        self.tdl_steps = int(config.get("tdl_steps", 1))
        self.agu_steps = int(config.get("agu_steps", 1))
        self.gmd_steps = int(config.get("gmd_steps", 1))
        self.use_acgan = bool(config.get("use_acgan", True))

        self.generator: nn.Module | None = None
        self.discriminator: nn.Module | None = None
        self.student: nn.Module | None = None
        self.generator_optimizer: optim.Optimizer | None = None
        self.discriminator_optimizer: optim.Optimizer | None = None
        self.student_optimizer: optim.Optimizer | None = None
        self.proxy_loader = None
        self.proxy_iter = None
        self.tdl_done = False

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
                num_classes=self.num_classes,
                output_size=int(input_shape[1]),
            ).to(device)
            self.generator_optimizer = optim.Adam(
                self.generator.parameters(), lr=self.generator_lr, betas=(0.5, 0.999)
            )

        if self.discriminator is None:
            input_shape = state.metadata.get("input_shape", (3, 32, 32))
            self.discriminator = DCGANDiscriminator(
                input_channels=int(self.config.get("output_channels", input_shape[0])),
                base_channels=self.base_channels,
                num_classes=self.num_classes if self.use_acgan else None,
                input_size=int(input_shape[1]),
            ).to(device)
            self.discriminator_optimizer = optim.Adam(
                self.discriminator.parameters(), lr=self.discriminator_lr, betas=(0.5, 0.999)
            )

        if self.student is None:
            input_shape = state.metadata.get("input_shape", (3, 32, 32))
            sub_config = state.metadata.get("substitute_config", {})
            opt_params = sub_config.get("optimizer", {})
            
            arch = self.config.get("student_arch", "resnet18-8x")
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

        if self.proxy_loader is None:
            proxy_config = self.config.get("attack", {}).get("proxy_dataset")
            if proxy_config is None:
                proxy_config = self.config.get("proxy_dataset")
            if proxy_config is None:
                raise ValueError("GAME requires proxy_dataset configuration")
            self.proxy_loader = create_dataloader(
                proxy_config,
                batch_size=self.batch_size,
                shuffle=True,
            )
            self.proxy_iter = iter(self.proxy_loader)

        if not self.tdl_done and self.tdl_steps > 0:
            self._tdl_pretrain(device)
            self.tdl_done = True

    def propose(self, k: int, state: BenchmarkState) -> QueryBatch:
        self._init_models(state)
        device = state.metadata.get("device", "cpu")
        z = torch.randn(k, self.noise_dim, device=device)
        class_probs = self._compute_class_distribution(state, device)
        # Ensure class_probs is valid for multinomial: remove NaNs and clamp
        class_probs = torch.nan_to_num(class_probs, nan=1.0/self.num_classes)
        class_probs = torch.clamp(class_probs, min=1e-9)
        class_probs = class_probs / class_probs.sum()
        
        y_g = torch.multinomial(class_probs, k, replacement=True)
        with torch.no_grad():
            x = self.generator(z, y_g)
        meta = {
            "generator_step": state.attack_state["step"],
            "synthetic": True,
            "z": z.cpu(),
            "y_g": y_g.cpu(),
            "acs_probs": class_probs.cpu(),
        }
        return QueryBatch(x=x, meta=meta)

    def observe(
        self,
        query_batch: QueryBatch,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        if self.generator is None or self.discriminator is None or self.student is None:
            return

        device = query_batch.x.device
        if oracle_output.kind == "soft_prob":
            victim_probs = oracle_output.y.to(device)
        else:
            victim_probs = F.one_hot(oracle_output.y, num_classes=self.num_classes).float().to(device)

        state.attack_state["last_victim_probs"] = victim_probs.detach().cpu()
        self._agu_phase(query_batch.x, victim_probs, device, query_batch.meta.get("z"), query_batch.meta.get("y_g"))
        self._gmd_phase(query_batch.x, victim_probs)

        state.attack_state["step"] += 1
        state.attack_state["substitute"] = self.student

    def _next_proxy_batch(self, device: str) -> torch.Tensor:
        try:
            x_real, _ = next(self.proxy_iter)
        except StopIteration:
            self.proxy_iter = iter(self.proxy_loader)
            x_real, _ = next(self.proxy_iter)
        return x_real.to(device), _.to(device)

    def _train_discriminator(
        self,
        fake_x: torch.Tensor,
        device: str,
        fake_labels: torch.Tensor | None = None,
    ) -> None:
        real_x, real_labels = self._next_proxy_batch(device)
        self.discriminator_optimizer.zero_grad()
        real_out = self.discriminator(real_x)
        fake_out = self.discriminator(fake_x.detach())

        if isinstance(real_out, tuple):
            real_source, real_class = real_out
            fake_source, fake_class = fake_out
        else:
            real_source, real_class = real_out, None
            fake_source, fake_class = fake_out, None

        real_source_labels = torch.ones_like(real_source)
        fake_source_labels = torch.zeros_like(fake_source)
        loss_real = F.binary_cross_entropy_with_logits(real_source, real_source_labels)
        loss_fake = F.binary_cross_entropy_with_logits(fake_source, fake_source_labels)
        loss = loss_real + loss_fake

        if real_class is not None:
            loss += F.cross_entropy(real_class, real_labels.long())
            if fake_labels is not None:
                loss += F.cross_entropy(fake_class, fake_labels.long())
        loss.backward()
        self.discriminator_optimizer.step()

    def _train_student(self, x: torch.Tensor, victim_probs: torch.Tensor) -> None:
        victim_config = self.state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization")
        if normalization is None:
            normalization = {"mean": [0.0], "std": [1.0]}
        norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(x.device)
        norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(x.device)
        
        def _norm(img):
            if img.min() < -0.1:
                img = img * 0.5 + 0.5
            return (img - norm_mean) / norm_std

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
        y_cpu: torch.Tensor | None,
        victim_probs: torch.Tensor,
        device: str,
    ) -> None:
        if z_cpu is None:
            z = torch.randn(victim_probs.size(0), self.noise_dim, device=device)
        else:
            z = z_cpu.to(device)
        if y_cpu is None:
            y_g = torch.randint(0, self.num_classes, (victim_probs.size(0),), device=device)
        else:
            y_g = y_cpu.to(device)
        victim_config = self.state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization")
        if normalization is None:
            normalization = {"mean": [0.0], "std": [1.0]}
        norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(device)
        norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(device)
        def _norm(img):
            return (img * 0.5 + 0.5 - norm_mean) / norm_std

        self.generator_optimizer.zero_grad()
        x_gen = self.generator(z, y_g)
        student_logits = self.student(_norm(x_gen))
        student_probs = F.softmax(student_logits, dim=1)

        l_res = -torch.mean(torch.relu(student_logits))
        top2 = torch.topk(student_probs, k=2, dim=1).values
        l_bou = torch.mean(top2[:, 0] - top2[:, 1])
        pseudo_labels = torch.argmax(student_probs, dim=1)
        l_adv = -F.cross_entropy(student_logits, pseudo_labels)
        l_dif = -F.kl_div(
            F.log_softmax(student_logits, dim=1),
            victim_probs,
            reduction="batchmean",
        )

        total = (
            self.beta1 * l_res
            + self.beta2 * l_bou
            + self.beta3 * l_adv
            + self.beta4 * l_dif
        )
        total.backward()
        self.generator_optimizer.step()

    def _tdl_pretrain(self, device: str) -> None:
        for _ in range(self.tdl_steps):
            real_x, _ = self._next_proxy_batch(device)
            z = torch.randn(real_x.size(0), self.noise_dim, device=device)
            y_g = torch.randint(0, self.num_classes, (real_x.size(0),), device=device)
            fake_x = self.generator(z, y_g)
            self._train_discriminator(fake_x, device, fake_labels=y_g)
            self.generator_optimizer.zero_grad()
            fake_logits = self.discriminator(fake_x)
            if isinstance(fake_logits, tuple):
                fake_source, fake_class = fake_logits
            else:
                fake_source, fake_class = fake_logits, None
            real_labels = torch.ones_like(fake_source)
            loss = F.binary_cross_entropy_with_logits(fake_source, real_labels)
            if fake_class is not None:
                loss += F.cross_entropy(fake_class, y_g)
            loss.backward()
            self.generator_optimizer.step()

    def _agu_phase(
        self,
        x: torch.Tensor,
        victim_probs: torch.Tensor,
        device: str,
        z_cpu: torch.Tensor | None,
        y_cpu: torch.Tensor | None,
    ) -> None:
        for _ in range(self.agu_steps):
            self._train_discriminator(x, device, fake_labels=y_cpu.to(device) if y_cpu is not None else None)
            self._train_generator(z_cpu, y_cpu, victim_probs, device)

    def _gmd_phase(self, x: torch.Tensor, victim_probs: torch.Tensor) -> None:
        for _ in range(self.gmd_steps):
            self._train_student(x, victim_probs)

    def _compute_class_distribution(self, state: BenchmarkState, device: str) -> torch.Tensor:
        if self.student is None:
            return torch.full((self.num_classes,), 1.0 / self.num_classes, device=device)

        z = torch.randn(self.num_classes, self.noise_dim, device=device)
        class_ids = torch.arange(self.num_classes, device=device)
        with torch.no_grad():
            x = self.generator(z, class_ids)
            student_logits = self.student(x)
            student_probs = F.softmax(student_logits, dim=1)

        if self.acs_strategy == "deviation" and "last_victim_probs" in state.attack_state:
            victim_probs = state.attack_state["last_victim_probs"].to(device)
            victim_mean = victim_probs.mean(dim=0)
            score = F.kl_div(
                torch.log(student_probs + 1e-6),
                victim_mean.unsqueeze(0).expand_as(student_probs),
                reduction="none",
            ).sum(dim=1)
        else:
            score = 1.0 - student_probs.max(dim=1).values

        score = score - score.min()
        score = score + 1e-6
        return score / score.sum()
