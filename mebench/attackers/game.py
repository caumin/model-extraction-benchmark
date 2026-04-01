"""GAME (Generative-Based Adaptive Model Extraction) attack."""

from typing import Any, Optional, Callable
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
from mebench.utils.config_aliases import (
    resolve_nominal_query_budget,
    resolve_query_budget,
)
from mebench.utils.dataloader import load_pool_to_memory
from mebench.utils.binary import (
    binary_bce_loss,
    binary_distribution_from_labels,
    binary_distribution_from_logits,
    binary_hard_labels_from_positive_probs,
    is_single_logit_binary_num_classes,
)
from mebench.utils.scaling import clamp_unit, unit_to_tanh


class GAME(AttackRunner):
    """GAME with shared generator/discriminator and adaptive losses."""

    def __init__(self, config: dict, state: BenchmarkState):
        super().__init__(config, state)

        # Official GAME attack.py default: --batch_size 1024.
        self.batch_size = int(config.get("batch_size", 1024))
        self.student_lr = float(config.get("student_lr", 0.1))
        self.generator_lr = float(config.get("generator_lr", 2e-4))
        self.discriminator_lr = float(config.get("discriminator_lr", 2e-4))
        self._noise_dim_explicit = "noise_dim" in config
        self.noise_dim = int(config.get("noise_dim", 100))
        self.num_classes = int(
            state.metadata.get("num_classes")
            or config.get("num_classes")
            or state.metadata.get("dataset_config", {}).get("num_classes", 10)
        )
        self.is_single_logit_binary = is_single_logit_binary_num_classes(self.num_classes)
        self.semantic_num_classes = 2 if self.is_single_logit_binary else self.num_classes
        self.proxy_num_classes = int(config.get("proxy_num_classes", 0))
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
        self.query_budget = resolve_query_budget(config, default=2000, context="game")
        self.nominal_query_budget = resolve_nominal_query_budget(
            config,
            default=0,
            context="game",
        )
        # Backward-compatible attributes used in existing tests/call sites.
        self.querybudget = self.query_budget
        self.nominal_querybudget = self.nominal_query_budget
        self.attack_train_epoch = int(config.get("attack_train_epoch", 40))
        self.final_retrain_epoch_offset = int(config.get("final_retrain_epoch_offset", 10))
        self.round_train_epochs = int(config.get("round_train_epochs", 20))
        self.train_on_full_buffer = bool(config.get("train_on_full_buffer", True))

        # TDL: Training Discriminator and Generator with proxy data.
        # Paper implies iterative training. default to 20 epochs/steps.
        self.tdl_steps = int(config.get("tdl_steps", 20))
        self.agu_steps = int(config.get("agu_steps", 2))
        self.gmd_steps = int(config.get("gmd_steps", 1))
        self.agu_loss_terms = self._parse_agu_loss_terms(
            config.get("agu_loss_terms", config.get("loss_items", ["res", "bou", "dif"]))
        )
        self.agu_update_discriminator = bool(config.get("agu_update_discriminator", False))
        self.use_acgan = bool(config.get("use_acgan", True))

        self.generator: Optional[nn.Module] = None
        self.discriminator: Optional[nn.Module] = None
        self.student: Optional[nn.Module] = None
        self.generator_optimizer: Optional[optim.Optimizer] = None
        self.discriminator_optimizer: Optional[optim.Optimizer] = None
        self.student_optimizer: Optional[optim.Optimizer] = None
        self._student_opt_config: dict[str, Any] = {}
        self.proxy_data: Optional[torch.Tensor] = None
        self.proxy_loader = None
        self.tdl_done = False
        # Cache for normalization tensors.
        # NOTE (Benchmark scaling unification vs GAME.pdf): The benchmark contract enforces
        # GAME keeps identity normalization tensors (mean=0,std=1) and fixed tanh
        # query/internal paths. Cache avoids per-step tensor allocations.
        self._norm_cache: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._ctx: Optional[BenchmarkContext] = None
        self._query_fn: Optional[Callable[..., OracleOutput]] = None
        self.sampler_weights: Optional[torch.Tensor] = None
        eval_interval_raw = int(config.get("eval_interval_queries", 100_000))
        self.eval_interval_queries = eval_interval_raw if eval_interval_raw > 0 else 0
        self._next_eval_query = self.eval_interval_queries
        self._periodic_eval_done: set[int] = set()

        self._initialize_state(state)
        if (not self._noise_dim_explicit) and self.use_acgan:
            # Official GAME uses latent dim = 10 * n_output.
            self.noise_dim = int(10 * int(self.proxy_num_classes))

    @staticmethod
    def _parse_agu_loss_terms(raw_terms: Any) -> tuple[str, ...]:
        allowed_terms = {"res", "bou", "adv", "dif"}
        if raw_terms is None:
            raw_list: list[str] = ["res", "bou", "dif"]
        elif isinstance(raw_terms, str):
            raw_list = [part.strip().lower() for part in raw_terms.split(",") if part.strip()]
        elif isinstance(raw_terms, (list, tuple, set)):
            raw_list = [str(part).strip().lower() for part in raw_terms if str(part).strip()]
        else:
            raise ValueError(
                "GAME agu_loss_terms must be a list/tuple/set of terms or a comma-separated string, "
                f"got {type(raw_terms).__name__}."
            )

        deduped_terms: list[str] = []
        seen_terms: set[str] = set()
        for term in raw_list:
            if term not in allowed_terms:
                raise ValueError(
                    "GAME agu_loss_terms contains unsupported term "
                    f"{term!r}. Allowed terms: {sorted(allowed_terms)}"
                )
            if term not in seen_terms:
                deduped_terms.append(term)
                seen_terms.add(term)
        return tuple(deduped_terms)

    @staticmethod
    def _kl_mean_legacy(log_probs: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        """Torch KLDiv legacy 'mean' behavior without warning.

        Official GAME code relies on `F.kl_div(..., reduction='mean')` semantics.
        Newer PyTorch versions warn that future `mean` will behave like
        `batchmean`. To keep current/official scaling stable across versions,
        we explicitly emulate legacy mean as `sum / numel`.
        """
        denom = float(max(1, int(log_probs.numel())))
        return F.kl_div(log_probs, target_probs, reduction="sum") / denom

    def _resolve_proxy_num_classes(self, state: BenchmarkState) -> int:
        if int(self.proxy_num_classes) > 0:
            return int(self.proxy_num_classes)

        proxy_config = self.config.get("attack", {}).get("proxy_dataset")
        if proxy_config is None:
            proxy_config = self.config.get("proxy_dataset")
        if proxy_config is None:
            proxy_config = state.metadata.get("dataset_config", {}).copy()

        proxy_config = dict(proxy_config or {})
        num_classes = proxy_config.get("num_classes")
        if num_classes is not None:
            return int(num_classes)

        name = str(proxy_config.get("surrogate_name") or proxy_config.get("name") or "").strip().upper()
        known = {
            "CIFAR10": 10,
            "CIFAR100": 100,
            "SVHN": 10,
            "GTSRB": 43,
            "FASHIONMNIST": 10,
            "EMNIST": 47,
        }
        return int(known.get(name, self.num_classes))

    def _publish_substitute(self) -> None:
        if self.student is None:
            return
        self.state.attack_state["substitute"] = self.student

    def _evaluate_current_substitute(
        self,
        substitute: nn.Module,
        device: str,
        *,
        track: str = "track_b",
        query_count: Optional[int] = None,
    ) -> None:
        super()._evaluate_current_substitute(
            substitute,
            device,
            track=track,
            query_count=query_count,
        )

    def _maybe_periodic_eval(self, device: str) -> None:
        if self.victim is None:
            return
        self._drain_deferred_track_b_checkpoints(device)
        if self.eval_interval_queries <= 0:
            return
        current_queries = int(self.state.query_count)
        next_eval_query = int(self._next_eval_query)
        if current_queries < next_eval_query:
            return
        substitute = self.state.attack_state.get("substitute")
        if substitute is None:
            return

        while next_eval_query <= current_queries:
            if next_eval_query not in self._periodic_eval_done:
                self._evaluate_current_substitute(
                    substitute,
                    device,
                    track="track_b",
                    query_count=next_eval_query,
                )
                self._periodic_eval_done.add(next_eval_query)
            next_eval_query += self.eval_interval_queries

        self._next_eval_query = next_eval_query

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

    def _proxy_to_internal_scale(self, x: torch.Tensor) -> torch.Tensor:
        return unit_to_tanh(clamp_unit(x))

    def _generator_to_internal_scale(self, x_tanh: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x_tanh, -1.0, 1.0)

    def _internal_to_oracle_scale(self, x_internal: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x_internal, -1.0, 1.0)

    def _oracle_to_internal_scale(self, x_oracle: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x_oracle, -1.0, 1.0)

    def run(self, ctx: BenchmarkContext) -> None:
        self.victim = ctx.oracle.model
        self._ctx = ctx
        self._query_fn = ctx.query
        device = self.state.metadata.get("device", "cpu")

        # Target Distribution Learning (TDL) phase (Algorithm 1, Lines 2-3).
        # Pre-train G/D on proxy dataset before consuming any victim queries.
        self._init_models(self.state)
        self._publish_substitute()
        self.sampler_weights = torch.ones(self.proxy_num_classes, device=device)
        if not self.tdl_done and self.tdl_steps > 0:
            self._tdl_phase(device)

        total_budget = min(int(self.state.budget_remaining), int(self.query_budget))
        start_query_count = int(self.state.query_count)
        nominal_budget_left = int(self.nominal_query_budget) if int(self.nominal_query_budget) > 0 else None
        # [FEATURE] Clean progress bar for Data-Free (Query Progress Only)
        pbar = self._create_progress_bar(total_budget, "[GAME] Extracting")
        
        while ctx.budget_remaining > 0:
            if nominal_budget_left is not None and int(nominal_budget_left) <= 0:
                break
            consumed = int(self.state.query_count) - int(start_query_count)
            budget_left = int(total_budget) - int(consumed)
            if budget_left <= 0:
                break

            step_size = min(int(self.batch_size), int(budget_left), int(ctx.budget_remaining))
            if nominal_budget_left is not None:
                step_size = min(int(step_size), int(nominal_budget_left))
                if self._query_fn is not None and int(self.agu_steps) > 0:
                    per_round = 1 + int(self.agu_steps)
                    max_outer_from_meter = int(ctx.budget_remaining) // int(per_round)
                    step_size = min(int(step_size), int(max_outer_from_meter))
                if step_size <= 0:
                    break
            query_batch = self._select_query_batch(step_size, self.state)
            if query_batch.x.size(0) == 0:
                break
            before_queries = int(self.state.query_count)
            oracle_output = ctx.query(query_batch.x, meta=query_batch.meta)
            self._handle_oracle_output(query_batch.x, query_batch.meta, oracle_output, self.state)
            if nominal_budget_left is not None:
                nominal_budget_left -= int(query_batch.x.size(0))
            delta_queries = int(self.state.query_count) - int(before_queries)
            if delta_queries > 0:
                pbar.update(delta_queries)
            self._publish_substitute()
            self._maybe_periodic_eval(device)

        # Official baseline final full-buffer training uses attack_train_epoch.
        if self.train_on_full_buffer:
            final_epochs = max(0, int(self.attack_train_epoch) - int(self.final_retrain_epoch_offset))
            self._train_student_from_buffer(final_epochs)

        self._publish_substitute()
        self._evaluate_current_substitute(self.student, device)
                
        pbar.close()
        self._ctx = None
        self._query_fn = None

    def _select_query_batch(self, k: int, state: BenchmarkState) -> QueryBatch:
        self._init_models(state)
        device = state.metadata.get("device", "cpu")

        weights = self.sampler_weights
        if weights is None or int(weights.numel()) != int(self.proxy_num_classes):
            weights = torch.ones(self.proxy_num_classes, device=device)
        else:
            weights = weights.to(device=device, dtype=torch.float32)
        weights = torch.nan_to_num(weights, nan=1.0)
        weights = torch.clamp(weights, min=1e-9)

        if self._ctx is not None:
            k = min(k, self._ctx.budget_remaining)
            if k <= 0:
                input_shape = state.metadata.get("input_shape", (3, 32, 32))
                x_empty = torch.empty((0, *input_shape), device=device)
                return QueryBatch(x=x_empty, meta={"sampler_weights": weights.cpu()})

        z = torch.randn(k, self.noise_dim, device=device)
        y_g = torch.multinomial(weights, k, replacement=True)
        with torch.no_grad():
            x = self.generator(z, y_g)

        x_internal = self._generator_to_internal_scale(x)
        x_query = self._internal_to_oracle_scale(x_internal)

        meta = {
            "generator_step": state.attack_state["step"],
            "synthetic": True,
            "z": z.cpu(),
            "y_g": y_g.cpu(),
            "sampler_weights": weights.cpu(),
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
        x_internal = self._oracle_to_internal_scale(x_query)
        if oracle_output.kind == "soft_prob":
            victim_probs = oracle_output.y.to(device)
        else:
            victim_probs = (
                binary_distribution_from_labels(oracle_output.y).float().to(device)
                if self.is_single_logit_binary
                else F.one_hot(oracle_output.y, num_classes=self.num_classes).float().to(device)
            )

        qx = state.attack_state.setdefault("query_data_x", [])
        qy = state.attack_state.setdefault("query_data_y", [])
        qx.append(x_internal.detach().cpu())
        qy.append(victim_probs.detach().cpu())

        state.attack_state["last_victim_probs"] = victim_probs.detach().cpu()

        # Official order per iteration: train student on full running buffer, then AGU.
        round_epochs = max(1, int(self.round_train_epochs))
        self._train_student_from_buffer(round_epochs)
        self._agu_phase(
            victim_probs,
            device,
            meta.get("z"),
        )

        state.attack_state["step"] += 1
        self._publish_substitute()

    def _initialize_state(self, state: BenchmarkState) -> None:
        self.proxy_num_classes = self._resolve_proxy_num_classes(state)
        state.attack_state["step"] = 0
        state.attack_state["query_data_x"] = []
        state.attack_state["query_data_y"] = []

    def _init_models(self, state: BenchmarkState) -> None:
        device = state.metadata.get("device", "cpu")

        def _apply_game_init(module: nn.Module) -> None:
            classname = module.__class__.__name__
            if "Conv" in classname and hasattr(module, "weight") and module.weight is not None:
                nn.init.normal_(module.weight.data, 0.0, 0.02)
            elif "BatchNorm" in classname and hasattr(module, "weight") and module.weight is not None:
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.constant_(module.bias.data, 0.0)

        if self.generator is None:
            input_shape = state.metadata.get("input_shape", (3, 32, 32))
            
            # [P0 ARCHITECTURE ENFORCEMENT] GAME requires ACGAN with dropout
            if self.use_acgan:
                self.generator = ACGANGenerator(
                    noise_dim=self.noise_dim,
                    output_channels=int(self.config.get("output_channels", input_shape[0])),
                    base_channels=self.base_channels,
                    num_classes=self.proxy_num_classes,
                    output_size=int(input_shape[1]),
                    dropout_prob=0.0,
                ).to(device)
            else:
                self.generator = DCGANGenerator(
                    noise_dim=self.noise_dim,
                    output_channels=int(self.config.get("output_channels", input_shape[0])),
                    base_channels=self.base_channels,
                    num_classes=self.proxy_num_classes,
                    output_size=int(input_shape[1]),
                ).to(device)
            self.generator.apply(_apply_game_init)
                
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
                    num_classes=self.proxy_num_classes,
                    input_size=int(input_shape[1]),
                    dropout_prob=0.25,  # Paper-mandated dropout
                ).to(device)
            else:
                self.discriminator = DCGANDiscriminator(
                    input_channels=int(self.config.get("output_channels", input_shape[0])),
                    base_channels=self.base_channels,
                    num_classes=self.proxy_num_classes if self.use_acgan else None,
                    input_size=int(input_shape[1]),
                ).to(device)
            self.discriminator.apply(_apply_game_init)
                
            self.discriminator_optimizer = optim.Adam(
                self.discriminator.parameters(), lr=self.discriminator_lr, betas=(0.5, 0.999)
            )

        if self.student is None:
            input_shape = state.metadata.get("input_shape", (3, 32, 32))
            sub_config = state.metadata.get("substitute_config", {})
            opt_params = dict(sub_config.get("optimizer", {}))
            attack_opt_name = str(self.config.get("optimizer", "")).strip().lower()
            if attack_opt_name in {"sgd", "adam", "adamw"}:
                opt_params["name"] = attack_opt_name
            if "student_lr" in self.config:
                opt_params["lr"] = float(self.student_lr)
            opt_params.setdefault("name", "sgd")
            opt_params.setdefault("lr", float(self.student_lr))
            opt_params.setdefault("momentum", 0.9)
            opt_params.setdefault("weight_decay", 5e-4)
            if str(opt_params.get("name", "")).lower() == "adam":
                opt_params.setdefault("betas", [0.5, 0.999])
            
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
            self._student_opt_config = dict(opt_params)
            self.student_optimizer = self._build_optimizer(self.student.parameters(), opt_params)

        if self.proxy_data is None or self.proxy_loader is None:
            proxy_config = self.config.get("attack", {}).get("proxy_dataset")
            if proxy_config is None:
                proxy_config = self.config.get("proxy_dataset")
            if proxy_config is None:
                # Fallback to surrogate if not specified (Track B)
                proxy_config = state.metadata.get("dataset_config", {}).copy()
                proxy_config["data_mode"] = "surrogate"

            if proxy_config:
                proxy_config = dict(proxy_config)
                if "surrogate_name" not in proxy_config:
                    fallback_name = str(
                        proxy_config.get("name")
                        or state.metadata.get("dataset_config", {}).get("name")
                        or ""
                    ).strip()
                    if fallback_name:
                        proxy_config["surrogate_name"] = fallback_name

                if "surrogate_name" not in proxy_config:
                    raise ValueError(
                        "GAME requires proxy_dataset.surrogate_name (or dataset name fallback) "
                        "to initialize proxy data loader."
                    )

                input_shape = state.metadata.get("input_shape", (3, 32, 32))
                proxy_config.setdefault("channels", int(input_shape[0]))
                proxy_config.setdefault("input_size", [int(input_shape[1]), int(input_shape[2])])
                # Keep GAME proxy tensors in raw/unit space before internal
                # tanh conversion. Surrogate-standard normalization is for
                # pool-based attacks, not data-free GAME proxy pretraining.
                proxy_config.setdefault("surrogate_normalization", "none")

                if self.proxy_loader is None:
                    self.proxy_loader = create_dataloader(
                        proxy_config,
                        batch_size=self.batch_size,
                        shuffle=True,
                    )

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
            x_rand = torch.rand(self.batch_size, 3, 32, 32, device=device)
            return self._proxy_to_internal_scale(x_rand)
            
        indices = torch.randint(0, self.proxy_data.size(0), (self.batch_size,), device=self.proxy_data.device)
        x_batch = self.proxy_data[indices].to(device)
        return self._proxy_to_internal_scale(x_batch)

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

        def _soft_bce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            pred = torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)
            return -(target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred)).mean()

        self.generator.train()
        self.discriminator.train()

        real_label_bank = 0.7 + 0.5 * torch.rand(self.proxy_num_classes, device=device)
        fake_label_bank = 0.3 * torch.rand(self.proxy_num_classes, device=device)

        for _epoch in range(int(self.tdl_steps)):
            for batch_idx, (real_x_cpu, real_y_cpu) in enumerate(self.proxy_loader):
                real_x = self._proxy_to_internal_scale(real_x_cpu.to(device))
                real_y = real_y_cpu.to(device).long()

                batch_size = int(real_x.size(0))
                if batch_size <= 0:
                    continue

                if int(real_y.min().item()) < 0 or int(real_y.max().item()) >= int(self.proxy_num_classes):
                    raise ValueError(
                        "Proxy labels must be within [0, proxy_num_classes). "
                        f"Got min={int(real_y.min().item())}, max={int(real_y.max().item())}, "
                        f"proxy_num_classes={int(self.proxy_num_classes)}."
                    )

                bank_idx = int(batch_idx % max(1, int(self.proxy_num_classes)))
                real_label_scalar = real_label_bank[bank_idx]
                fake_label_scalar = fake_label_bank[bank_idx]
                if batch_idx % 25 == 0:
                    real_label_scalar, fake_label_scalar = fake_label_scalar, real_label_scalar

                fake_class_labels = torch.full(
                    (batch_size,),
                    int(self.proxy_num_classes),
                    dtype=torch.long,
                    device=device,
                )

                # -------------------------
                # Train Discriminator
                # -------------------------
                z = torch.randn(batch_size, self.noise_dim, device=device)
                y_g = torch.randint(0, self.proxy_num_classes, (batch_size,), device=device)
                fake_x = self._generator_to_internal_scale(self.generator(z, y_g))

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

                d_real_targets = torch.full_like(real_validity, float(real_label_scalar.item()))
                d_fake_targets = torch.full_like(fake_validity, float(fake_label_scalar.item()))
                d_loss_source = _soft_bce(real_validity, d_real_targets) + _soft_bce(fake_validity, d_fake_targets)

                d_loss_class = torch.zeros((), device=device)
                if self.use_acgan and real_cls_logits is not None and fake_cls_logits is not None:
                    d_loss_class = F.cross_entropy(real_cls_logits, real_y) + F.cross_entropy(fake_cls_logits, fake_class_labels)

                d_loss = d_loss_source + d_loss_class
                d_loss.backward()
                self.discriminator_optimizer.step()

                # -------------------------
                # Train Generator
                # -------------------------
                z = torch.randn(batch_size, self.noise_dim, device=device)
                y_g = torch.randint(0, self.proxy_num_classes, (batch_size,), device=device)
                fake_x = self._generator_to_internal_scale(self.generator(z, y_g))

                self.generator_optimizer.zero_grad()
                g_fake_out = self.discriminator(fake_x)
                if self.use_acgan:
                    fake_validity, fake_cls_logits = g_fake_out
                else:
                    fake_validity = g_fake_out
                    fake_cls_logits = None

                g_loss_source = _soft_bce(fake_validity, torch.ones_like(fake_validity))
                g_loss_class = torch.zeros((), device=device)
                if self.use_acgan and fake_cls_logits is not None:
                    g_loss_class = F.cross_entropy(fake_cls_logits, y_g)

                g_loss = g_loss_source + g_loss_class
                g_loss.backward()
                self.generator_optimizer.step()

        self.tdl_done = True

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

        opt_cfg = dict(self._student_opt_config or {})
        if not opt_cfg:
            opt_cfg = {"name": "sgd", "lr": float(self.student_lr), "momentum": 0.9, "weight_decay": 5e-4}

        current_lr = float(opt_cfg.get("lr", self.student_lr))
        local_opt_cfg = dict(opt_cfg)
        local_opt_cfg["lr"] = current_lr
        local_optimizer = self._build_optimizer(self.student.parameters(), local_opt_cfg)

        lr_dropped = False
        self.student.train()
        for epoch in range(int(epochs)):
            if (not lr_dropped) and int(epochs) > 1 and (2 * int(epoch) >= int(epochs)):
                current_lr *= 0.1
                local_opt_cfg = dict(opt_cfg)
                local_opt_cfg["lr"] = current_lr
                local_optimizer = self._build_optimizer(self.student.parameters(), local_opt_cfg)
                lr_dropped = True
            for x_b, y_b in loader:
                if int(x_b.size(0)) == 1:
                    break
                x_b = x_b.to(device)
                y_b = y_b.to(device)
                logits = self.student((x_b - norm_mean) / norm_std)
                if self.is_single_logit_binary:
                    if y_b.ndim == 1:
                        y_b = y_b.float().unsqueeze(1)
                    loss = binary_bce_loss(logits, y_b)
                else:
                    probs = F.softmax(logits, dim=1)
                    if y_b.ndim == 1:
                        y_b = F.one_hot(y_b.long(), num_classes=self.num_classes).float()
                    y_b = y_b / y_b.sum(dim=1, keepdim=True).clamp_min(1e-10)
                    loss = F.kl_div(torch.log(probs + 1e-10), y_b, reduction="batchmean")
                local_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                local_optimizer.step()

        self.student_optimizer = local_optimizer

    def _agu_phase(
        self,
        victim_probs: torch.Tensor,
        device: str,
        z: Optional[torch.Tensor],
    ) -> None:
        """Adversarial Generator Update (AGU)."""
        if z is None:
            return

        active_loss_terms = set(self.agu_loss_terms)
        if not active_loss_terms:
            return

        norm_mean, norm_std = self._get_norm_tensors(device)

        def _norm(img: torch.Tensor) -> torch.Tensor:
            return (img - norm_mean) / norm_std

        sample_weights = self.sampler_weights
        if sample_weights is None or int(sample_weights.numel()) != int(self.proxy_num_classes):
            sample_weights = torch.ones(self.proxy_num_classes, device=device)
        else:
            sample_weights = sample_weights.to(device=device, dtype=torch.float32)
        sample_weights = torch.nan_to_num(sample_weights, nan=1.0)
        sample_weights = torch.clamp(sample_weights, min=1e-9)

        last_sample_labels: Optional[torch.Tensor] = None
        last_pred_victim_softmax: Optional[torch.Tensor] = None
        last_pred_attacker_softmax: Optional[torch.Tensor] = None
        last_pred_attacker_logsoftmax: Optional[torch.Tensor] = None

        for _ in range(self.agu_steps):
            batch_k = int(z.size(0))
            if self._ctx is not None:
                batch_k = min(batch_k, int(self._ctx.budget_remaining))
                if batch_k <= 0:
                    break

            noise = torch.randn(batch_k, self.noise_dim, device=device)
            labels = torch.multinomial(sample_weights, batch_k, replacement=True)

            # Re-generate fake batch for this AGU step.
            fake_x_raw = self.generator(noise, labels)
            fake_x = self._generator_to_internal_scale(fake_x_raw)

            victim_probs_step = victim_probs
            if self._query_fn is not None:
                oracle_output = self._query_fn(
                    self._internal_to_oracle_scale(fake_x).detach(),
                    meta={"synthetic": True, "internal_agu": True},
                )
                if oracle_output.kind == "soft_prob":
                    victim_probs_step = oracle_output.y.to(device)
                else:
                    victim_probs_step = (
                        binary_distribution_from_labels(oracle_output.y).float().to(device)
                        if self.is_single_logit_binary
                        else F.one_hot(
                            oracle_output.y,
                            num_classes=self.num_classes,
                        ).float().to(device)
                    )

            last_sample_labels = labels.detach()
            last_pred_victim_softmax = victim_probs_step.detach()

            # Optional discriminator update (disabled by default for official parity).
            if self.agu_update_discriminator:
                real_x = self._next_proxy_batch(device)
                self.discriminator_optimizer.zero_grad()

                real_validity, _real_label = self.discriminator(real_x)
                fake_validity, fake_label = self.discriminator(fake_x.detach())

                d_loss_real = F.binary_cross_entropy(real_validity, torch.ones_like(real_validity))
                d_loss_fake = F.binary_cross_entropy(fake_validity, torch.zeros_like(fake_validity))
                d_loss_cls = 0.0
                if self.use_acgan:
                    fake_class_labels = torch.full(
                        (batch_k,),
                        int(self.proxy_num_classes),
                        dtype=torch.long,
                        device=device,
                    )
                    d_loss_cls = F.cross_entropy(fake_label, fake_class_labels)

                d_loss = d_loss_real + d_loss_fake + d_loss_cls
                d_loss.backward()
                self.discriminator_optimizer.step()

            # Train Generator
            self.generator_optimizer.zero_grad()
            fake_validity, fake_label = self.discriminator(fake_x)
            
            # Paper-aligned GAME objectives on generated samples.
            student_logits = self.student(_norm(fake_x))
            if self.is_single_logit_binary:
                student_probs = binary_distribution_from_logits(student_logits)
                student_log_probs = torch.log(student_probs.clamp_min(1e-10))
            else:
                student_probs = F.softmax(student_logits, dim=1)
                student_log_probs = F.log_softmax(student_logits, dim=1)

            last_pred_attacker_softmax = student_probs.detach()
            last_pred_attacker_logsoftmax = student_log_probs.detach()

            g_loss = torch.zeros((), device=device)

            if "res" in active_loss_terms:
                # L_res (official code path): positive sum of positive attacker logits.
                # This is minimized directly in the reference implementation.
                pos_logits = student_logits[student_logits > 0]
                if pos_logits.numel() == 0:
                    l_res = torch.zeros((), device=device)
                else:
                    l_res = pos_logits.sum()
                g_loss = g_loss + (self.beta1 * l_res)

            if "bou" in active_loss_terms:
                # L_bou (official): scaled top1-top2 margin sum over batch.
                if self.is_single_logit_binary:
                    l_bou = student_logits.abs().sum()
                else:
                    sorted_logits = torch.sort(student_logits, descending=True, dim=1).values
                    l_bou = (sorted_logits[:, 0] - sorted_logits[:, 1]).sum()
                g_loss = g_loss + (self.beta2 * l_bou)

            if "adv" in active_loss_terms:
                # L_adv (paper): -CE(N_S(x), argmax N_S(x)).
                if self.is_single_logit_binary:
                    pseudo = binary_hard_labels_from_positive_probs(victim_probs_step[:, 1:2] if victim_probs_step.ndim == 2 and victim_probs_step.size(1) == 2 else victim_probs_step)
                    l_adv = -binary_bce_loss(student_logits, pseudo.float().unsqueeze(1))
                else:
                    l_adv = -F.cross_entropy(student_logits, student_logits.argmax(dim=1))
                g_loss = g_loss + (self.beta3 * l_adv)

            if "dif" in active_loss_terms:
                # L_dif (official implementation parity):
                # -100 * KL(N_V(x) || N_S(x)) with torch default "mean" reduction.
                l_dif = -self._kl_mean_legacy(student_log_probs, victim_probs_step)
                g_loss = g_loss + (self.beta4 * l_dif)

            g_loss.backward()
            self.generator_optimizer.step()

        if (
            last_sample_labels is None
            or last_pred_victim_softmax is None
            or last_pred_attacker_softmax is None
            or last_pred_attacker_logsoftmax is None
        ):
            return

        if self.acs_strategy == "random":
            self.sampler_weights = torch.ones(self.proxy_num_classes, device=device)
            return

        strategy = self.acs_strategy
        if strategy == "uncertainty":
            strategy = "unconfident"

        if strategy in {"unconfident", "confident"}:
            confident_log = torch.ones(self.proxy_num_classes, 2, device=device)
            confident_log[:, 0] = 1.0 / float(max(1, int(self.proxy_num_classes)))
            sample_confident = last_pred_victim_softmax.max(1)[0]
            for index, confident in enumerate(sample_confident):
                cls = int(last_sample_labels[index].item())
                if 0 <= cls < int(self.proxy_num_classes):
                    confident_log[cls, 0] += confident
                    confident_log[cls, 1] += 1.0
            class_confidence = confident_log[:, 0] / confident_log[:, 1].clamp_min(1e-9)
            if strategy == "unconfident":
                self.sampler_weights = 1.0 - class_confidence
            else:
                self.sampler_weights = class_confidence
        elif strategy == "deviation":
            deviation_log = torch.ones(self.proxy_num_classes, 2, device=device)
            deviation_loss = F.kl_div(
                last_pred_attacker_logsoftmax,
                last_pred_victim_softmax,
                reduction="none",
            )
            sample_deviation = deviation_loss.mean(dim=1)
            for index, deviation in enumerate(sample_deviation):
                cls = int(last_sample_labels[index].item())
                if 0 <= cls < int(self.proxy_num_classes):
                    deviation_log[cls, 0] += deviation
                    deviation_log[cls, 1] += 1.0
            self.sampler_weights = deviation_log[:, 0] / deviation_log[:, 1].clamp_min(1e-9)
        else:
            raise ValueError(f"Unsupported GAME acs_strategy: {self.acs_strategy}")

        self.sampler_weights = torch.nan_to_num(self.sampler_weights, nan=1.0)
        self.sampler_weights = torch.clamp(self.sampler_weights, min=1e-9)
