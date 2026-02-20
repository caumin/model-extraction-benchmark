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
from mebench.utils.dataloader import load_pool_to_memory
from mebench.utils.scaling import tanh_to_unit, clamp_unit


class GAME(AttackRunner):
    """GAME with shared generator/discriminator and adaptive losses."""

    def __init__(self, config: dict, state: BenchmarkState):
        super().__init__(config, state)

        # Official GAME attack.py default: --batch_size 1024.
        self.batch_size = int(config.get("batch_size", 1024))
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
        # Official methods.game starts with uniform random sampler_weights.
        self.acs_strategy = str(config.get("acs_strategy", "random")).strip().lower()
        self.acs_probe_size = int(config.get("acs_probe_size", 0))

        # Loss weights (Eq.14). Defaults follow the reference implementation.
        self.beta1 = float(config.get("beta1", 0.002))  # L_res
        self.beta2 = float(config.get("beta2", 0.01))   # L_bou
        self.beta3 = float(config.get("beta3", 10.0))   # L_adv
        self.beta4 = float(config.get("beta4", 100.0))  # L_dif

        # Official attack.py defaults.
        self.querybudget = int(config.get("querybudget", config.get("budget", 2000)))
        self.attack_train_epoch = int(config.get("attack_train_epoch", 40))
        self.round_train_epochs = int(config.get("round_train_epochs", 20))
        self.train_on_full_buffer = bool(config.get("train_on_full_buffer", True))

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
        self.proxy_data: torch.Tensor | None = None
        self.proxy_loader = None
        self._proxy_iter = None
        self.tdl_done = False
        # Cache for normalization tensors.
        # NOTE (Benchmark scaling unification vs GAME.pdf): The benchmark contract enforces
        # oracle/eval inputs in [0,1] and ignores dataset mean/std normalization. For
        # benchmark-wide consistency, GAME uses identity normalization (mean=0,std=1).
        # We keep the cache to avoid per-step tensor allocs.
        self._norm_cache: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._ctx: Optional[BenchmarkContext] = None
        self._query_fn: Optional[Callable[..., OracleOutput]] = None
        self._pending_query_k: Optional[int] = None

        self._initialize_state(state)

    def _get_norm_tensors(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = self.state.metadata.get("input_shape", (3, 32, 32))
        channels = int(input_shape[0])
        key = (str(device), channels)
        cached = self._norm_cache.get(key)
        if cached is not None:
            return cached

        norm_mean = torch.zeros((1, channels, 1, 1), device=device)
        norm_std = torch.ones((1, channels, 1, 1), device=device)
        self._norm_cache[key] = (norm_mean, norm_std)
        return norm_mean, norm_std

    def run(self, ctx: BenchmarkContext) -> None:
        self.victim = ctx.oracle.model
        self._ctx = ctx
        self._query_fn = ctx.query
        device = self.state.metadata.get("device", "cpu")

        # Target Distribution Learning (TDL) phase (Algorithm 1, Lines 2-3).
        # Pre-train G/D on proxy dataset before consuming any victim queries.
        self._init_models(self.state)
        if not self.tdl_done and self.tdl_steps > 0:
            self._tdl_phase(device)

        total_budget = min(int(self.state.budget_remaining), int(self.querybudget))
        # [FEATURE] Clean progress bar for Data-Free (Query Progress Only)
        pbar = self._create_progress_bar(total_budget, "[GAME] Extracting")
        
        budget_left = int(total_budget)
        while budget_left > 0 and ctx.budget_remaining > 0:
            step_size = min(int(self.batch_size), int(budget_left), int(ctx.budget_remaining))
            query_batch = self._select_query_batch(step_size, self.state)
            if query_batch.x.size(0) == 0:
                break
            oracle_output = ctx.query(query_batch.x, meta=query_batch.meta)
            self._handle_oracle_output(query_batch.x, query_batch.meta, oracle_output, self.state)
            pbar.update(query_batch.x.size(0))
            budget_left -= int(query_batch.x.size(0))

        # Official baseline final full-buffer training uses attack_train_epoch.
        if self.train_on_full_buffer:
            self._train_student_from_buffer(int(self.attack_train_epoch))

        self._evaluate_current_substitute(self.student, device)
                
        pbar.close()
        self._ctx = None
        self._query_fn = None

    def _select_query_batch(self, k: int, state: BenchmarkState) -> QueryBatch:
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
                x_empty = torch.empty((0, *input_shape), device=device)
                return QueryBatch(x=x_empty, meta={"acs_probs": class_probs.cpu()})

        z = torch.randn(k, self.noise_dim, device=device)
        y_g = torch.multinomial(class_probs, k, replacement=True)
        with torch.no_grad():
            x = self.generator(z, y_g)

        # Benchmark scaling unification (DFME-style): convert tanh [-1,1] -> [0,1] once.
        x_query = tanh_to_unit(x)

        meta = {
            "generator_step": state.attack_state["step"],
            "synthetic": True,
            "z": z.cpu(),
            "y_g": y_g.cpu(),
            "acs_probs": class_probs.cpu(),
        }
        return QueryBatch(x=x_query, meta=meta)

    def observe(
        self,
        query_batch: QueryBatch,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        self._handle_oracle_output(query_batch.x, query_batch.meta, oracle_output, state)

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

        # Paper Algorithm 1 ordering: distill student (GMD) before generator update (AGU).
        self._gmd_phase(x_query, victim_probs)
        self._agu_phase(x_query, victim_probs, device, meta.get("z"), meta.get("y_g"))

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
            opt_params = dict(sub_config.get("optimizer", {}))
            opt_params.setdefault("name", "sgd")
            opt_params.setdefault("lr", float(self.student_lr))
            opt_params.setdefault("momentum", 0.9)
            opt_params.setdefault("weight_decay", 5e-4)
            
            arch = sub_config.get("arch") or self.config.get("student_arch", "resnet18")
            width_mult = int(sub_config.get("width_mult", 1))
            dropout_prob = float(sub_config.get("dropout_prob", 0.0))
            self.student = create_substitute(
                arch=arch,
                num_classes=self.num_classes,
                input_channels=int(input_shape[0]),
                width_mult=width_mult,
                dropout_prob=dropout_prob,
            ).to(device)
            # [UNIFIED] Config-driven optimizer construction
            self.student_optimizer = self._build_optimizer(self.student.parameters(), opt_params)

        if self.proxy_data is None or self.proxy_loader is None:
            proxy_config = self.config.get("attack", {}).get("proxy_dataset")
            if proxy_config is None:
                proxy_config = self.config.get("proxy_dataset")
            if proxy_config is None:
                # Fallback to surrogate if not specified (Track B)
                proxy_config = state.metadata.get("dataset_config", {}).copy()
                proxy_config["data_mode"] = "surrogate"
                # Ensure we have a valid surrogate config
                if "surrogate_name" not in proxy_config:
                    # Try to infer or fail
                    pass
            
            if proxy_config:
                proxy_config = dict(proxy_config)

                input_shape = state.metadata.get("input_shape", (3, 32, 32))
                proxy_config.setdefault("channels", int(input_shape[0]))
                proxy_config.setdefault("input_size", [int(input_shape[1]), int(input_shape[2])])

                if self.proxy_loader is None:
                    self.proxy_loader = create_dataloader(
                        proxy_config,
                        batch_size=self.batch_size,
                        shuffle=True,
                    )
                    self._proxy_iter = iter(self.proxy_loader)

                if self.proxy_data is None:
                    self.proxy_data = load_pool_to_memory(
                        proxy_config,
                        device=device,
                        desc="[GAME] Caching proxy data",
                        max_samples=100_000,
                    )

    def _next_proxy_batch(self, device: str) -> torch.Tensor:
        if self.proxy_data is None or self.proxy_data.size(0) == 0:
            # Fallback if no proxy data (should not happen if config correct)
            return torch.randn(self.batch_size, 3, 32, 32, device=device)
            
        indices = torch.randint(0, self.proxy_data.size(0), (self.batch_size,), device=self.proxy_data.device)
        return self.proxy_data[indices].to(device)

    def _next_proxy_batch_with_labels(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proxy_loader is None:
            raise ValueError("GAME TDL requires a labeled proxy_dataset (proxy_loader is None)")

        if self._proxy_iter is None:
            self._proxy_iter = iter(self.proxy_loader)

        try:
            x_batch, y_batch = next(self._proxy_iter)
        except StopIteration:
            self._proxy_iter = iter(self.proxy_loader)
            x_batch, y_batch = next(self._proxy_iter)

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device).long()
        return x_batch, y_batch

    def _tdl_phase(self, device: str) -> None:
        """Target Distribution Learning (TDL): pre-train G/D on proxy data.

        Paper: Algorithm 1 (Lines 2-3) with AC-GAN objectives (Eq.4-7).
        """

        if self.tdl_done:
            return

        if self.generator is None or self.discriminator is None:
            raise RuntimeError("GAME TDL requires generator/discriminator initialized")
        if self.generator_optimizer is None or self.discriminator_optimizer is None:
            raise RuntimeError("GAME TDL requires generator/discriminator optimizers initialized")

        if self.proxy_loader is None:
            raise ValueError("GAME requires proxy_dataset for TDL pretraining")

        self.logger.info(f"[GAME] Starting TDL phase for {self.tdl_steps} steps")

        bce = nn.BCEWithLogitsLoss()

        self.generator.train()
        self.discriminator.train()

        for _ in range(int(self.tdl_steps)):
            real_x, real_y = self._next_proxy_batch_with_labels(device)
            batch_size = int(real_x.size(0))
            if batch_size <= 0:
                continue

            if int(real_y.min().item()) < 0 or int(real_y.max().item()) >= int(self.num_classes):
                raise ValueError(
                    "Proxy labels must be within [0, num_classes). "
                    f"Got min={int(real_y.min().item())}, max={int(real_y.max().item())}, "
                    f"num_classes={int(self.num_classes)}."
                )

            # -------------------------
            # Train Discriminator: maximize L_C + L_S (Eq.7)
            # -------------------------
            z = torch.randn(batch_size, self.noise_dim, device=device)
            y_g = torch.randint(0, self.num_classes, (batch_size,), device=device)
            fake_x = tanh_to_unit(self.generator(z, y_g))

            self.discriminator_optimizer.zero_grad()

            d_real_out = self.discriminator(real_x)
            if self.use_acgan:
                real_validity, real_cls_logits = d_real_out
            else:
                real_validity = d_real_out
                real_cls_logits = None

            d_fake_out = self.discriminator(fake_x.detach())
            if self.use_acgan:
                fake_validity, fake_cls_logits = d_fake_out
            else:
                fake_validity = d_fake_out
                fake_cls_logits = None

            d_loss_source = bce(real_validity, torch.ones_like(real_validity)) + bce(
                fake_validity, torch.zeros_like(fake_validity)
            )

            d_loss_class = torch.zeros((), device=device)
            if self.use_acgan and real_cls_logits is not None and fake_cls_logits is not None:
                d_loss_class = F.cross_entropy(real_cls_logits, real_y) + F.cross_entropy(
                    fake_cls_logits, y_g
                )

            d_loss = d_loss_source + d_loss_class
            d_loss.backward()
            self.discriminator_optimizer.step()

            # -------------------------
            # Train Generator: maximize L_C - L_S (Eq.6)
            # -------------------------
            z = torch.randn(batch_size, self.noise_dim, device=device)
            y_g = torch.randint(0, self.num_classes, (batch_size,), device=device)
            fake_x = tanh_to_unit(self.generator(z, y_g))

            self.generator_optimizer.zero_grad()
            g_fake_out = self.discriminator(fake_x)
            if self.use_acgan:
                fake_validity, fake_cls_logits = g_fake_out
            else:
                fake_validity = g_fake_out
                fake_cls_logits = None

            g_loss_source = bce(fake_validity, torch.ones_like(fake_validity))
            g_loss_class = torch.zeros((), device=device)
            if self.use_acgan and fake_cls_logits is not None:
                g_loss_class = F.cross_entropy(fake_cls_logits, y_g)

            g_loss = g_loss_source + g_loss_class
            g_loss.backward()
            self.generator_optimizer.step()

        self.tdl_done = True

    def _compute_class_distribution(self, state: BenchmarkState, device: str) -> torch.Tensor:
        """Compute class distribution from FRESH victim queries for ACS deviation.
        
        [P0 FIX] Paper requires fresh victim queries for ACS deviation, not cached stats.
        """
        if self.student is None or self.generator is None:
            return torch.full((self.num_classes,), 1.0 / self.num_classes, device=device)
        # Benchmark scaling unification: identity normalization (mean=0,std=1).
        norm_mean, norm_std = self._get_norm_tensors(device)
        def _norm(img: torch.Tensor) -> torch.Tensor:
            # NOTE: `img` here is already in [0,1]. Do NOT apply an extra *0.5+0.5.
            # The benchmark uses identity normalization.
            return (img - norm_mean) / norm_std

        if self.acs_strategy == "random":
            return torch.full((self.num_classes,), 1.0 / self.num_classes, device=device)

        if self.acs_strategy != "deviation":
            # Uncertainty-based selection (default): Prefer classes where student is uncertain
            z = torch.randn(self.num_classes, self.noise_dim, device=device)
            class_ids = torch.arange(self.num_classes, device=device)
            with torch.no_grad():
                # We need ACGAN to control class generation. If DCGAN, class_ids ignored.
                # GAME assumes ACGAN for class-conditional generation.
                x_gen_raw = self.generator(z, class_ids)
                x_gen = tanh_to_unit(x_gen_raw)
                student_logits = self.student(_norm(x_gen))
                student_probs = F.softmax(student_logits, dim=1)
            
            # Uncertainty: 1 - max_prob (or entropy)
            # Higher uncertainty -> Higher selection probability
            score = 1.0 - student_probs.max(dim=1).values
        else:
            # Deviation-based selection: Prefer classes where Student differs from Victim
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
                x_gen_raw = self.generator(z, class_ids)
                x_gen = tanh_to_unit(x_gen_raw)
                student_logits = self.student(_norm(x_gen))
                student_probs = F.softmax(student_logits, dim=1)

            x_query = x_gen
            oracle_output = self._query_fn(x_query, meta={"acs_probe": True, "y_g": class_ids.detach().cpu()})
            if oracle_output.kind == "soft_prob":
                victim_probs = oracle_output.y.to(device)
            else:
                victim_probs = F.one_hot(oracle_output.y, num_classes=self.num_classes).float().to(device)

            self._update_victim_stats(state, victim_probs, class_ids)
            
            # KL(S || V) approximation
            s_log = torch.log(student_probs + 1e-10)
            v_log = torch.log(victim_probs + 1e-10)
            # We want to select classes with HIGH disagreement
            kl_div = (student_probs * (s_log - v_log)).sum(dim=1)
            
            # Map probe scores back to full class vector
            score = torch.full((self.num_classes,), 1e-6, device=device)
            score[class_ids] = kl_div

        score = score - score.min()
        score = score + 1e-6
        return score / score.sum()

    def _train_student_from_buffer(self, epochs: int) -> None:
        if self.student is None or self.student_optimizer is None:
            return
        if epochs <= 0:
            return
        qx = self.state.attack_state.get("query_data_x", [])
        qy = self.state.attack_state.get("query_data_y", [])
        if len(qx) == 0 or len(qy) == 0:
            return

        x_all = torch.cat(qx, dim=0)
        y_all = torch.cat(qy, dim=0)
        device = next(self.student.parameters()).device
        norm_mean, norm_std = self._get_norm_tensors(str(device))

        dataset = torch.utils.data.TensorDataset(x_all, y_all)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

        self.student.train()
        for _ in range(int(epochs)):
            for x_b, y_b in loader:
                x_b = x_b.to(device)
                y_b = y_b.to(device)
                logits = self.student((x_b - norm_mean) / norm_std)
                probs = F.softmax(logits, dim=1)
                if y_b.ndim == 1:
                    y_b = F.one_hot(y_b.long(), num_classes=self.num_classes).float()
                y_b = y_b / y_b.sum(dim=1, keepdim=True).clamp_min(1e-10)
                loss = F.kl_div(torch.log(probs + 1e-10), y_b, reduction="batchmean")
                self.student_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.student_optimizer.step()

    def _update_victim_stats(self, state: BenchmarkState, probs: torch.Tensor, y_g: Optional[torch.Tensor]) -> None:
        """Update running statistics of victim class probabilities per generated class."""
        if y_g is None:
            return
            
        # Update counts
        # This is a simplified online update for mean vectors per class
        for i, c in enumerate(y_g):
            c_idx = int(c.item())
            if c_idx >= self.num_classes: continue
            
            current_avg = state.attack_state["victim_class_avg_prob"][c_idx].to(probs.device)
            current_n = state.attack_state["victim_class_counts"][c_idx].item()
            
            new_n = current_n + 1
            new_avg = (current_avg * current_n + probs[i]) / new_n
            
            state.attack_state["victim_class_avg_prob"][c_idx] = new_avg
            state.attack_state["victim_class_counts"][c_idx] = new_n

    def _agu_phase(
        self, 
        x_query: torch.Tensor, 
        victim_probs: torch.Tensor, 
        device: str, 
        z: Optional[torch.Tensor],
        y_g: Optional[torch.Tensor]
    ) -> None:
        """Adversarial Generator Update (AGU)."""
        if z is None or y_g is None:
            return

        norm_mean, norm_std = self._get_norm_tensors(device)

        def _norm(img: torch.Tensor) -> torch.Tensor:
            return (img - norm_mean) / norm_std

        for _ in range(self.agu_steps):
            # 1. Train Discriminator
            real_x = self._next_proxy_batch(device)
            self.discriminator_optimizer.zero_grad()
            
            # Re-generate fake batch to detach from previous graph
            fake_x_raw = self.generator(z.to(device), y_g.to(device))
            fake_x = tanh_to_unit(fake_x_raw)
            
            real_validity, real_label = self.discriminator(real_x)
            fake_validity, fake_label = self.discriminator(fake_x.detach())
            
            # Loss D: Real/Fake + Aux Label (ACGAN)
            d_loss_real = F.binary_cross_entropy_with_logits(real_validity, torch.ones_like(real_validity))
            d_loss_fake = F.binary_cross_entropy_with_logits(fake_validity, torch.zeros_like(fake_validity))
            d_loss_cls = 0.0
            if self.use_acgan:
                d_loss_cls = F.cross_entropy(fake_label, y_g.to(device))
                
            d_loss = d_loss_real + d_loss_fake + d_loss_cls
            d_loss.backward()
            self.discriminator_optimizer.step()
            
            # 2. Train Generator
            self.generator_optimizer.zero_grad()
            fake_validity, fake_label = self.discriminator(fake_x)
            
            # GAN/ACGAN generator terms
            g_loss_adv = F.binary_cross_entropy_with_logits(fake_validity, torch.ones_like(fake_validity))
            g_loss_cls = torch.zeros((), device=device)
            if self.use_acgan:
                g_loss_cls = F.cross_entropy(fake_label, y_g.to(device))

            # Paper-aligned GAME objectives on generated samples.
            student_logits = self.student(_norm(fake_x))
            student_probs = F.softmax(student_logits, dim=1)

            # L_res: increase responsivity (negative ReLU activation sum).
            l_res = -torch.relu(student_logits).sum(dim=1).mean()

            # L_bou: move samples toward decision boundary (top1-top2 margin).
            top2_vals = torch.topk(student_probs, k=2, dim=1).values
            l_bou = (top2_vals[:, 0] - top2_vals[:, 1]).mean()

            # L_adv (paper): -CE(N_S(x), argmax N_S(x)).
            l_adv = -F.cross_entropy(student_logits, student_logits.argmax(dim=1))

            # L_dif (paper): -KL(N_S(x), N_V(x)).
            # PyTorch KLDiv computes KL(target || input). To compute KL(S||V):
            # KL(S||V) = KLDiv(log(V), S).
            l_dif = -F.kl_div(
                torch.log(victim_probs + 1e-10),
                student_probs,
                reduction="batchmean",
            )
            g_loss = (
                self.beta1 * l_res
                + self.beta2 * l_bou
                + self.beta3 * l_adv
                + self.beta4 * l_dif
            )
            g_loss.backward()
            self.generator_optimizer.step()

    def _gmd_phase(self, x_query: torch.Tensor, victim_probs: torch.Tensor) -> None:
        """Gradient Maximization Discrepancy (GMD) / Student Training."""
        device = x_query.device
        
        # Train Student
        self.student.train()
        self.student_optimizer.zero_grad()
        
        norm_mean, norm_std = self._get_norm_tensors(device)
        
        student_in = (x_query - norm_mean) / norm_std
        student_logits = self.student(student_in)
        student_probs = F.softmax(student_logits, dim=1)
        
        # L_res: distillation loss.
        # Use KL(teacher || student): KLDiv(log(student), teacher)
        loss_res = F.kl_div(torch.log(student_probs + 1e-10), victim_probs, reduction="batchmean")
        
        # L_dif: Diff Loss (make S different from V?)
        # Wait, GMD is usually for Generator.
        # In GAME paper, S is trained to minimize loss_res.
        # G is trained to maximize discrepancy.
        # This function is called _gmd_phase but actually trains STUDENT.
        # So it should minimize match loss.
        
        loss = self.beta1 * loss_res
        loss.backward()
        self.student_optimizer.step()

        # Official methods.baseline/game retrains attacker for multiple epochs each round.
        # Keep this behavior configurable while preserving existing mebench API.
        if int(self.round_train_epochs) > 1 and self.train_on_full_buffer:
            self._train_student_from_buffer(int(self.round_train_epochs) - 1)
