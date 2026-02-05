"""GAME (Generative-Based Adaptive Model Extraction) attack."""

from typing import Dict, Any, List, Tuple, Optional, Callable
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.models.gan import DCGANGenerator, DCGANDiscriminator, ACGANGenerator, ACGANDiscriminator
from mebench.models.substitute_factory import create_substitute
from mebench.data.loaders import create_dataloader


class GAME(AttackRunner):
    """GAME with shared generator/discriminator and adaptive losses."""

    def __init__(self, config: dict, state: BenchmarkState):
        super().__init__(config, state)

        self.batch_size = int(config.get("batch_size", 128))
        self.student_lr = float(config.get("student_lr", 0.1))
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
        self.acs_probe_size = int(config.get("acs_probe_size", 0))

        self.beta1 = float(config.get("beta1", 1.0)) # l_res
        self.beta2 = float(config.get("beta2", 1.0)) # l_bou
        self.beta3 = float(config.get("beta3", 0.0)) # l_adv
        self.beta4 = float(config.get("beta4", 1.0)) # l_dif

        # TDL: Training Discriminator and Generator with proxy data.
        # Paper implies iterative training. default to 20 epochs/steps.
        self.tdl_steps = int(config.get("tdl_steps", 20))
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
        self._ctx: Optional[BenchmarkContext] = None
        self._query_fn: Optional[Callable[..., OracleOutput]] = None
        self._pending_query_k: Optional[int] = None

        self._initialize_state(state)

    def run(self, ctx: BenchmarkContext) -> None:
        self.victim = ctx.oracle.model
        self._ctx = ctx
        self._query_fn = ctx.query
        device = self.state.metadata.get("device", "cpu")
        total_budget = self.state.budget_remaining
        pbar = self._create_progress_bar(total_budget, "[GAME] Extracting")
        
        last_eval_queries = 0
        eval_interval = total_budget // 10

        while ctx.budget_remaining > 0:
            step_size = self._default_step_size(ctx)
            x_query, meta = self._select_query_batch(step_size, self.state)
            if x_query.size(0) == 0:
                break
            oracle_output = ctx.query(x_query, meta=meta)
            self._handle_oracle_output(x_query, meta, oracle_output, self.state)
            pbar.update(x_query.size(0))
            
            # Periodic evaluation
            queries_done = total_budget - ctx.budget_remaining
            if queries_done - last_eval_queries >= eval_interval:
                self._evaluate_current_substitute(self.student, device)
                last_eval_queries = queries_done
                
        pbar.close()
        self._ctx = None
        self._query_fn = None

    def _select_query_batch(self, k: int, state: BenchmarkState) -> tuple[torch.Tensor, dict]:
        self._init_models(state)
        device = state.metadata.get("device", "cpu")
        self._pending_query_k = k
        class_probs = self._compute_class_distribution(state, device)
        self._pending_query_k = None
        class_probs = torch.nan_to_num(class_probs, nan=1.0 / self.num_classes)
        class_probs = torch.clamp(class_probs, min=1e-9)
        class_probs = class_probs / class_probs.sum()

        if self._ctx is not None:
            k = min(k, self._ctx.budget_remaining)
            if k <= 0:
                input_shape = state.metadata.get("input_shape", (3, 32, 32))
                return torch.empty((0, *input_shape), device=device), {"acs_probs": class_probs.cpu()}

        z = torch.randn(k, self.noise_dim, device=device)
        y_g = torch.multinomial(class_probs, k, replacement=True)
        with torch.no_grad():
            x = self.generator(z, y_g)

        x_query = x * 0.5 + 0.5

        meta = {
            "generator_step": state.attack_state["step"],
            "synthetic": True,
            "z": z.cpu(),
            "y_g": y_g.cpu(),
            "acs_probs": class_probs.cpu(),
        }
        return x_query, meta

    def _handle_oracle_output(
        self,
        x_query: torch.Tensor,
        meta: dict,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        if self.generator is None or self.discriminator is None or self.student is None:
            return

        device = x_query.device
        if oracle_output.kind == "soft_prob":
            victim_probs = oracle_output.y.to(device)
        else:
            victim_probs = F.one_hot(oracle_output.y, num_classes=self.num_classes).float().to(device)

        self._update_victim_stats(state, victim_probs, meta.get("y_g"))

        state.attack_state["last_victim_probs"] = victim_probs.detach().cpu()
        self._agu_phase(x_query, victim_probs, device, meta.get("z"), meta.get("y_g"))
        self._gmd_phase(x_query, victim_probs)

        state.attack_state["step"] += 1
        state.attack_state["substitute"] = self.student

    def _initialize_state(self, state: BenchmarkState) -> None:
        state.attack_state["step"] = 0
        state.attack_state["victim_class_avg_prob"] = torch.full(
            (self.num_classes, self.num_classes),
            1.0 / self.num_classes,
        )
        state.attack_state["victim_class_counts"] = torch.zeros(self.num_classes)

    def _init_models(self, state: BenchmarkState) -> None:
        device = state.metadata.get("device", "cpu")
        if self.generator is None:
            input_shape = state.metadata.get("input_shape", (3, 32, 32))
            
            # [P0 ARCHITECTURE ENFORCEMENT] GAME requires ACGAN with dropout
            if self.use_acgan:
                self.generator = ACGANGenerator(
                    noise_dim=self.noise_dim,
                    output_channels=int(self.config.get("output_channels", input_shape[0])),
                    base_channels=self.base_channels,
                    num_classes=self.num_classes,
                    output_size=int(input_shape[1]),
                    dropout_prob=0.25,  # Paper-mandated dropout
                ).to(device)
            else:
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
            
            # [P0 ARCHITECTURE ENFORCEMENT] GAME requires ACGAN with dropout
            if self.use_acgan:
                self.discriminator = ACGANDiscriminator(
                    input_channels=int(self.config.get("output_channels", input_shape[0])),
                    base_channels=self.base_channels,
                    num_classes=self.num_classes,
                    input_size=int(input_shape[1]),
                    dropout_prob=0.25,  # Paper-mandated dropout
                ).to(device)
            else:
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
            
            arch = sub_config.get("arch") or self.config.get("student_arch", "resnet18-8x")
            width_mult = int(sub_config.get("width_mult", 1))
            dropout_prob = float(sub_config.get("dropout_prob", 0.0))
            self.student = create_substitute(
                arch=arch,
                num_classes=self.num_classes,
                input_channels=int(input_shape[0]),
                width_mult=width_mult,
                dropout_prob=dropout_prob,
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

    def _update_victim_stats(self, state, victim_probs, labels):
        if labels is None:
            return
        
        cache = state.attack_state.get("victim_class_avg_prob")
        counts = state.attack_state.get("victim_class_counts")
        
        if cache is None or counts is None:
            cache = torch.zeros(self.num_classes, self.num_classes, device=victim_probs.device)
            # Init with uniform to avoid log(0)
            cache.fill_(1.0 / self.num_classes)
            counts = torch.zeros(self.num_classes, device=victim_probs.device)
        else:
            if cache.device != victim_probs.device:
                cache = cache.to(victim_probs.device)
            if counts.device != victim_probs.device:
                counts = counts.to(victim_probs.device)
            
        labels = labels.to(victim_probs.device)
        
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.any():
                n_new = mask.sum()
                mean_new = victim_probs[mask].mean(dim=0)
                
                prev_n = counts[c]
                prev_mean = cache[c]
                
                new_mean = (prev_mean * prev_n + mean_new * n_new) / (prev_n + n_new)
                cache[c] = new_mean
                counts[c] += n_new
                
        state.attack_state["victim_class_avg_prob"] = cache
        state.attack_state["victim_class_counts"] = counts

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
            # Only apply class loss for labels within valid range (handles OOD proxy)
            valid_mask = real_labels < self.num_classes
            if valid_mask.any():
                loss += F.cross_entropy(real_class[valid_mask], real_labels[valid_mask].long())
            
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
        
        # l_dif = KL( student || victim ) as per Eq. 13 and Eq. 9
        # F.kl_div(p, q) computes KL(q || p). We need KL(p || q).
        student_log_probs = F.log_softmax(student_logits, dim=1)
        # KL(P||Q) = sum P * (logP - logQ)
        # Note: victim_probs are already softmax probabilities.
        # student_log_probs is logP.
        # We need logQ = log(victim_probs).
        victim_log_probs = torch.log(victim_probs + 1e-10)
        
        # Manual implementation of KL(pS || pV)
        # kl = (student_probs * (student_log_probs - victim_log_probs)).sum(dim=1).mean()
        # Alternatively, use F.kl_div by swapping and using log_target=True if available, 
        # but manual is clearer for fidelity.
        l_dif_val = (student_probs * (student_log_probs - victim_log_probs)).sum(dim=1).mean()
        l_dif = -l_dif_val

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
        """Compute class distribution from FRESH victim queries for ACS deviation.
        
        [P0 FIX] Paper requires fresh victim queries for ACS deviation, not cached stats.
        """
        if self.student is None or self.generator is None:
            return torch.full((self.num_classes,), 1.0 / self.num_classes, device=device)
        victim_config = self.state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization")
        if normalization is None:
            normalization = {"mean": [0.0], "std": [1.0]}
        norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(device)
        norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(device)
        def _norm(img):
            return (img * 0.5 + 0.5 - norm_mean) / norm_std

        if self.acs_strategy != "deviation":
            z = torch.randn(self.num_classes, self.noise_dim, device=device)
            class_ids = torch.arange(self.num_classes, device=device)
            with torch.no_grad():
                x_gen = self.generator(z, class_ids)
                student_logits = self.student(_norm(x_gen))
                student_probs = F.softmax(student_logits, dim=1)
            score = 1.0 - student_probs.max(dim=1).values
        else:
            if self._query_fn is None:
                self.logger.warning("GAME ACS deviation requires oracle query; returning uniform distribution.")
                return torch.full((self.num_classes,), 1.0 / self.num_classes, device=device)
            budget_remaining = None
            if self._ctx is not None:
                budget_remaining = int(self._ctx.budget_remaining)

            probe_budget = self.num_classes
            if self.acs_probe_size > 0:
                probe_budget = min(self.acs_probe_size, self.num_classes)
            if budget_remaining is not None and self._pending_query_k is not None:
                probe_budget = min(probe_budget, max(0, budget_remaining - int(self._pending_query_k)))

            if probe_budget <= 0:
                self.logger.warning("GAME ACS deviation probe skipped due to budget constraints.")
                return torch.full((self.num_classes,), 1.0 / self.num_classes, device=device)

            if probe_budget < self.num_classes:
                class_ids = torch.randperm(self.num_classes, device=device)[:probe_budget]
            else:
                class_ids = torch.arange(self.num_classes, device=device)

            z = torch.randn(class_ids.size(0), self.noise_dim, device=device)
            with torch.no_grad():
                x_gen = self.generator(z, class_ids)
                student_logits = self.student(_norm(x_gen))
                student_probs = F.softmax(student_logits, dim=1)

            x_query = x_gen * 0.5 + 0.5
            oracle_output = self._query_fn(x_query, meta={"acs_probe": True, "y_g": class_ids.detach().cpu()})
            if oracle_output.kind == "soft_prob":
                victim_probs = oracle_output.y.to(device)
            else:
                victim_probs = F.one_hot(oracle_output.y, num_classes=self.num_classes).float().to(device)

            self._update_victim_stats(state, victim_probs, class_ids)
            s_log = torch.log(student_probs + 1e-10)
            v_log = torch.log(victim_probs + 1e-10)
            kl_div = (student_probs * (s_log - v_log)).sum(dim=1)
            score = torch.full((self.num_classes,), 1e-6, device=device)
            score[class_ids] = kl_div

        score = score - score.min()
        score = score + 1e-6
        return score / score.sum()
