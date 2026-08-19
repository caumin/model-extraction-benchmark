"""DFMS-HL (Data-Free Model Stealing with Hard Labels) attack."""

from typing import Dict, Any, List, Tuple, Optional, Iterator
import math
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
from mebench.models.gan import (
    DCGANGenerator,
    DCGANDiscriminator,
    OfficialDFMSDCGANGenerator,
    OfficialDFMSDCGANDiscriminator,
)
from mebench.models.substitute_factory import create_substitute
from mebench.attackers.dfms_budget import DFMSBudgetPlan, planned_stage5_epochs
from mebench.data.loaders import create_dataloader
from mebench.utils.dataloader import load_pool_to_memory
from mebench.utils.binary import (
    binary_bce_loss,
    binary_distribution_from_logits,
    binary_hard_labels_from_positive_probs,
    is_single_logit_binary_num_classes,
)
from mebench.utils.scaling import tanh_to_unit, clamp_unit, unit_to_tanh


class DFMSHL(AttackRunner):
    """DFMS-HL with proxy data, GAN training, and hard-label cloning."""

    def __init__(self, config: dict, state: BenchmarkState):
        super().__init__(config, state)

        # Official repo defaults to batchSize=64 for GAN training scripts.
        # See: temp_dfms_hl/code/train_generator/dcgan.py (arg --batchSize)
        self.batch_size = int(config.get("batch_size", 64))

        # Official repo uses student_lr=0.01 in alternate training (train_generator_clone.py),
        # while paper discussion often cites max lr=0.1 for clone training.
        # We default to the official repo value when aligning to the reference implementation.
        self.clone_lr = float(config.get("clone_lr", 0.01))
        self.generator_lr = float(config.get("generator_lr", 2e-4))
        self.discriminator_lr = float(config.get("discriminator_lr", 2e-4))
        self.noise_dim = int(config.get("noise_dim", 100))
        self.num_classes = int(
            state.metadata.get("num_classes")
            or config.get("num_classes")
            or state.metadata.get("dataset_config", {}).get("num_classes", 10)
        )
        self.is_single_logit_binary = is_single_logit_binary_num_classes(self.num_classes)
        self.base_channels = int(config.get("base_channels", 64))

        dataset_name = state.metadata.get("dataset_config", {}).get("name", "cifar10")
        dataset_name_lower = str(dataset_name).lower()

        # Paper specifies div=500 for CIFAR-10 and div=100 for CIFAR-100.
        default_div = 500.0
        if dataset_name_lower == "cifar100":
            default_div = 100.0
        self.diversity_weight = float(config.get("diversity_weight", default_div))

        # [P0 FIX] Paper mandates 50,000 initial queries for CIFAR-10, not 1,000
        if dataset_name_lower in ["cifar10", "svhn", "cifar100"]:
            self.init_nc = int(config.get("init_nc", 50000))
        else:
            self.init_nc = int(config.get("init_nc", 1000))
        self.init_ng = int(config.get("init_ng", 100))
        
        self.pretrain_steps = int(config.get("pretrain_steps", 200))
        self.use_clone_cosine = bool(config.get("use_clone_cosine", True))

        # Official repo stage structure (scripts):
        # 1) DCGAN pretrain (dcgan.py): niter=200, batchSize=64
        # 2) Student init w/ proxy+DCGAN (train_student.py): max_epochs=200, lr=0.1
        # 3) DeGAN/DivGAN train (train_gen.py): niter=100, d_l=10
        # 4) Student init w/ proxy+DeGAN (train_student.py): max_epochs=200, lr=0.1
        # 5) Alternate training (train_generator_clone.py): niter=800, student_lr=0.01, d_l=500
        # We implement this stage pipeline inside `run()` when enabled.
        self.use_official_stages = bool(config.get("use_official_stages", True))
        planner_cfg_raw = config.get("budget_planner", {})
        planner_cfg = planner_cfg_raw if isinstance(planner_cfg_raw, dict) else {}
        self.budget_planner_mode = str(
            planner_cfg.get("mode", config.get("budget_planner_mode", "paper_fair"))
        ).strip().lower()
        if self.budget_planner_mode not in {"paper_fair", "legacy_fixed_epochs"}:
            raise ValueError(
                "DFMS-HL budget planner mode must be 'paper_fair' or 'legacy_fixed_epochs', "
                f"got {self.budget_planner_mode!r}"
            )
        self.n_c_target = int(planner_cfg.get("nC_target", config.get("nC_target", 50_000)))
        self.enforce_exact_budget = bool(planner_cfg.get("enforce_exact_budget", True))
        self.dcgan_epochs = int(config.get("dcgan_epochs", 200))
        self.student_init_epochs = int(config.get("student_init_epochs", 200))
        self.degan_epochs = int(config.get("degan_epochs", 100))
        self.student_degan_epochs = int(config.get("student_degan_epochs", self.student_init_epochs))
        self.alternate_epochs = int(config.get("alternate_epochs", 800))

        # Dataset synthesis sizes/ratios from official scripts.
        # Official train_student.py uses `max_samples=50000` and then adds:
        #   proxy_samples = min(len(proxy_subset), proxy_data_ratio*max_samples)
        #   gan_samples = gan_data_ratio*max_samples
        # producing ~45k total for 10-class CIFAR100 proxy (5k + 40k).
        self.max_synth_samples = int(config.get("max_synth_samples", 50_000))
        self.proxy_data_ratio = float(config.get("proxy_data_ratio", 1.0))
        self.dcgan_data_ratio = float(config.get("dcgan_data_ratio", 0.8))
        self.div_gan_data_ratio = float(config.get("div_gan_data_ratio", 0.8))

        # Student training defaults from official scripts.
        self.student_batch_size = int(config.get("student_batch_size", 128))
        self.student_init_lr = float(config.get("student_init_lr", 0.1))
        self.student_alt_lr = float(config.get("student_alt_lr", 0.01))
        self.student_momentum = float(config.get("student_momentum", 0.9))
        self.student_weight_decay = float(config.get("student_weight_decay", 5e-4))

        # Official train_generator_clone.py warmup: inital_lr=0.001, epoch<10.
        self.use_student_warmup = bool(config.get("use_student_warmup", True))
        self.student_warmup_init_lr = float(config.get("student_warmup_init_lr", 0.001))
        self.student_warmup_epochs = int(config.get("student_warmup_epochs", 10))

        # Official DeGAN stage uses d_l=10; alternate uses d_l=500.
        self.degan_diversity_weight = float(config.get("degan_diversity_weight", 10.0))
        self.alternate_diversity_weight = float(
            config.get("alternate_diversity_weight", self.diversity_weight)
        )

        # Query chunk size for oracle labeling; budget is counted per-image.
        self.oracle_batch_size = int(config.get("oracle_batch_size", self.batch_size))

        self.use_official_dcgan_arch = bool(config.get("use_official_dcgan_arch", True))
        self.proxy_pad_crop = bool(config.get("proxy_pad_crop", True))
        self.alternate_auto_augment = bool(config.get("alternate_auto_augment", True))
        self.auto_augment_policy = str(config.get("auto_augment_policy", "cifar10")).strip().lower()

        self.generator: Optional[nn.Module] = None
        self.discriminator: Optional[nn.Module] = None
        self.clone: Optional[nn.Module] = None
        self.generator_optimizer: Optional[optim.Optimizer] = None
        self.discriminator_optimizer: Optional[optim.Optimizer] = None
        self.clone_optimizer: Optional[optim.Optimizer] = None
        self.clone_scheduler: Optional[optim.lr_scheduler.CosineAnnealingLR] = None
        self.proxy_data: Optional[torch.Tensor] = None
        self.pretrained = False
        self._auto_augment = self._build_auto_augment()
        eval_interval_raw = int(config.get("eval_interval_queries", 100_000))
        self.eval_interval_queries = eval_interval_raw if eval_interval_raw > 0 else 0
        self._next_eval_query = self.eval_interval_queries
        self._periodic_eval_done: set[int] = set()
        self._eval_substitute: Optional[nn.Module] = None
        self._budget_plan: Optional[DFMSBudgetPlan] = None
        self._stage_query_ledger: Dict[str, int] = {
            "student_init_dcgan": 0,
            "student_init_degan": 0,
            "alternate": 0,
        }

        self._initialize_state(state)

    def _unit_to_internal(self, x: torch.Tensor) -> torch.Tensor:
        """Convert canonical [0,1] tensors to fixed tanh internal scale."""

        x_unit = clamp_unit(x)
        return unit_to_tanh(x_unit)

    def _query_scale_generated_from_unit(self, x_unit: torch.Tensor) -> torch.Tensor:
        """Map generated unit-scale tensors to fixed tanh oracle-query scale."""

        x_01 = clamp_unit(x_unit)
        return torch.clamp(unit_to_tanh(x_01), -1.0, 1.0)

    def _query_scale_generated_from_tanh(self, x_tanh: torch.Tensor) -> torch.Tensor:
        """Map generated tanh tensors to fixed tanh oracle-query scale."""

        return torch.clamp(x_tanh, -1.0, 1.0)

    def _get_substitute_for_eval(self) -> Optional[nn.Module]:
        if self.clone is None:
            return None
        return self.clone

    def _publish_substitute(self) -> None:
        substitute = self._get_substitute_for_eval()
        if substitute is None:
            return
        self.state.attack_state["substitute"] = substitute

    def _maybe_periodic_eval(self, device: str) -> None:
        if self.victim is None:
            return
        self._drain_deferred_track_b_checkpoints(device)
        if self.eval_interval_queries <= 0:
            return
        current_queries = int(self.state.query_count)
        if current_queries < int(self._next_eval_query):
            return
        if current_queries in self._periodic_eval_done:
            return
        substitute = self.state.attack_state.get("substitute")
        if substitute is None:
            return

        self._evaluate_current_substitute(
            substitute,
            device,
            track="track_b",
            query_count=current_queries,
        )
        self._periodic_eval_done.add(current_queries)
        while self._next_eval_query <= current_queries:
            self._next_eval_query += self.eval_interval_queries

    def run(self, ctx: BenchmarkContext) -> None:
        self.victim = ctx.oracle.model
        device = self.state.metadata.get("device", "cpu")
        total_budget = self.state.budget_remaining
        # [FEATURE] Clean progress bar for Data-Free (Query Progress Only)
        # Replaced manual tqdm with self._create_progress_bar for consistency
        pbar = self._create_progress_bar(total_budget, "[DFMSHL] Extracting")

        self.logger.info(
            "[DFMSHL] Starting (budget=%d, device=%s, official_stages=%s)",
            int(total_budget),
            str(device),
            str(bool(self.use_official_stages)),
        )
        if self.use_official_stages:
            self.logger.info(
                "[DFMSHL] Query progress bar tracks oracle queries only; stage bars show no-query training."
            )

        # Paper (DFMS.pdf) defines the optimization objectives (Eq.(2)-(4)) but does not
        # specify any budget-percentage LR milestone schedule. The official repo scripts
        # keep constant learning rates for GAN components and use cosine annealing for
        # the student (clone) in some stages. We therefore avoid applying the benchmark's
        # unified budget-milestone LR decay here to stay aligned with the official repo.

        if not self.use_official_stages:
            self._init_models(self.state)
            self._publish_substitute()
            # Legacy/paper-style loop (kept for backwards-compat and unit tests).
            while ctx.budget_remaining > 0:
                step_size = self._default_step_size(ctx)
                query_batch = self._select_query_batch(step_size, self.state)
                oracle_output = ctx.query(query_batch.x, meta=query_batch.meta)
                self._handle_oracle_output(query_batch.x, oracle_output, self.state)
                pbar.update(query_batch.x.size(0))
                self._publish_substitute()
                self._maybe_periodic_eval(device)
            self._publish_substitute()
            pbar.close()
            return

        # Official-repo aligned stage pipeline (see `temp_dfms_hl/run_*.sh`).
        self.logger.info("[DFMSHL] Initializing models + caching proxy data (if needed)...")
        self._init_models(self.state)
        self._publish_substitute()

        if isinstance(self.proxy_data, torch.Tensor) and self.proxy_data.numel() > 0:
            self.logger.info(
                "[DFMSHL] Proxy cache ready: shape=%s device=%s",
                tuple(int(x) for x in self.proxy_data.shape),
                str(self.proxy_data.device),
            )
        self._budget_plan = self._build_budget_plan(int(total_budget))
        self.state.attack_state["dfmshl_budget_plan"] = {
            "mode": self._budget_plan.mode,
            "total_budget": int(self._budget_plan.total_budget),
            "stage2_target_queries": int(self._budget_plan.stage2_target_queries),
            "stage4_target_queries": int(self._budget_plan.stage4_target_queries),
            "stage5_target_queries": int(self._budget_plan.stage5_target_queries),
            "proxy_subset_size": int(self._budget_plan.proxy_subset_size),
            "stage5_planned_epochs": int(self._budget_plan.stage5_planned_epochs),
        }
        self.logger.info(
            "[DFMSHL] Budget plan mode=%s total=%d s2=%d s4=%d s5=%d np=%d e5=%d",
            str(self._budget_plan.mode),
            int(self._budget_plan.total_budget),
            int(self._budget_plan.stage2_target_queries),
            int(self._budget_plan.stage4_target_queries),
            int(self._budget_plan.stage5_target_queries),
            int(self._budget_plan.proxy_subset_size),
            int(self._budget_plan.stage5_planned_epochs),
        )
        self.logger.info(
            "[DFMSHL] Stages: dcgan_epochs=%d student_init_epochs=%d degan_epochs=%d student_degan_epochs=%d alternate_epochs=%d",
            int(self.dcgan_epochs),
            int(self.student_init_epochs),
            int(self.degan_epochs),
            int(self.student_degan_epochs),
            int(self.alternate_epochs),
        )

        # Stage 1: DCGAN pretrain (no oracle queries).
        ctx.log_event("dfmshl_stage", {"stage": "dcgan_pretrain", "queries": 0})
        self.logger.info("[DFMSHL] Stage 1/5: DCGAN pretrain (no queries)")
        self._official_stage_dcgan_pretrain(device)

        # Stage 2: student init with proxy + DCGAN images (oracle labeling consumes budget).
        if ctx.budget_remaining > 0:
            ctx.log_event("dfmshl_stage", {"stage": "student_init_dcgan", "queries": "labels"})
            self.logger.info("[DFMSHL] Stage 2/5: student init (proxy + DCGAN) (queries for labels)")
            before_q = int(ctx.query_count)
            self._official_stage_student_init(
                ctx,
                device,
                pbar=pbar,
                synth_ratio=self.dcgan_data_ratio,
                stage_name="student_init_dcgan",
                train_epochs=self.student_init_epochs,
                student_lr=self.student_init_lr,
                target_queries=int(self._budget_plan.stage2_target_queries),
            )
            self._stage_query_ledger["student_init_dcgan"] = int(ctx.query_count) - before_q
            self._publish_substitute()
            self._maybe_periodic_eval(device)

        # Stage 3: train DeGAN/DivGAN (no oracle queries).
        ctx.log_event("dfmshl_stage", {"stage": "train_degan", "queries": 0})
        self.logger.info("[DFMSHL] Stage 3/5: train DeGAN/DivGAN (no queries)")
        self._official_stage_train_degan(device)

        # Stage 4: student init with proxy + DeGAN images (oracle labeling consumes budget).
        if ctx.budget_remaining > 0:
            ctx.log_event("dfmshl_stage", {"stage": "student_init_degan", "queries": "labels"})
            self.logger.info("[DFMSHL] Stage 4/5: student init (proxy + DeGAN) (queries for labels)")
            before_q = int(ctx.query_count)
            self._official_stage_student_init(
                ctx,
                device,
                pbar=pbar,
                synth_ratio=self.div_gan_data_ratio,
                stage_name="student_init_degan",
                train_epochs=self.student_degan_epochs,
                student_lr=self.student_init_lr,
                target_queries=int(self._budget_plan.stage4_target_queries),
            )
            self._stage_query_ledger["student_init_degan"] = int(ctx.query_count) - before_q
            self._publish_substitute()
            self._maybe_periodic_eval(device)

        # Stage 5: alternate training until budget exhausted (budget-capped).
        if ctx.budget_remaining > 0:
            ctx.log_event("dfmshl_stage", {"stage": "alternate", "queries": "labels"})
            self.logger.info("[DFMSHL] Stage 5/5: alternate training (queries for labels)")
            before_q = int(ctx.query_count)
            self._official_stage_alternate(
                ctx,
                device,
                pbar=pbar,
                target_queries=int(self._budget_plan.stage5_target_queries),
                planned_epochs=int(self._budget_plan.stage5_planned_epochs),
            )
            self._stage_query_ledger["alternate"] = int(ctx.query_count) - before_q
            self._publish_substitute()
            self._maybe_periodic_eval(device)

        self._publish_substitute()
        self.state.attack_state["dfmshl_stage_query_ledger"] = {
            key: int(value) for key, value in self._stage_query_ledger.items()
        }
        ctx.log_event(
            "dfmshl_budget_summary",
            {
                "mode": str(self.budget_planner_mode),
                "planned_budget": int(total_budget),
                "realized_budget": int(ctx.query_count),
                "budget_remaining": int(ctx.budget_remaining),
                "q_stage2": int(self._stage_query_ledger["student_init_dcgan"]),
                "q_stage4": int(self._stage_query_ledger["student_init_degan"]),
                "q_stage5": int(self._stage_query_ledger["alternate"]),
            },
        )

        if self.budget_planner_mode == "paper_fair" and self.enforce_exact_budget:
            if int(ctx.budget_remaining) != 0:
                raise RuntimeError(
                    "DFMS-HL paper_fair mode requires exact budget exhaustion, "
                    f"but {int(ctx.budget_remaining)} queries remain."
                )
        pbar.close()

    def _official_proxy_subset_size(self) -> int:
        if self.proxy_data is None:
            return 0
        if self.proxy_data.numel() == 0:
            return 0
        proxy_cap = int(max(0, int(self.proxy_data_ratio * self.max_synth_samples)))
        if proxy_cap <= 0:
            return 0
        return min(int(self.proxy_data.size(0)), int(proxy_cap))

    def _legacy_stage_target_queries(self, synth_ratio: float) -> int:
        proxy_n = self._official_proxy_subset_size()
        synth_n = int(max(0, int(float(synth_ratio) * int(self.max_synth_samples))))
        return max(0, int(proxy_n + synth_n))

    def _build_budget_plan(self, total_budget: int) -> DFMSBudgetPlan:
        total = max(0, int(total_budget))
        proxy_subset = self._official_proxy_subset_size()

        if self.budget_planner_mode == "legacy_fixed_epochs":
            s2_req = self._legacy_stage_target_queries(self.dcgan_data_ratio)
            s4_req = self._legacy_stage_target_queries(self.div_gan_data_ratio)
        else:
            s2_req = max(0, int(self.n_c_target))
            s4_req = max(0, int(self.n_c_target))

        s2_target = min(total, s2_req)
        s4_target = min(max(0, total - s2_target), s4_req)
        s5_target = max(0, total - s2_target - s4_target)

        if proxy_subset > 0:
            required_epochs = planned_stage5_epochs(s5_target, proxy_subset)
        else:
            required_epochs = 0

        if self.budget_planner_mode == "legacy_fixed_epochs":
            stage5_epochs = max(int(self.alternate_epochs), int(required_epochs))
        else:
            stage5_epochs = int(required_epochs)

        return DFMSBudgetPlan(
            mode=str(self.budget_planner_mode),
            total_budget=total,
            stage2_target_queries=int(s2_target),
            stage4_target_queries=int(s4_target),
            stage5_target_queries=int(s5_target),
            proxy_subset_size=int(proxy_subset),
            stage5_planned_epochs=int(stage5_epochs),
        )

    @staticmethod
    def _allocate_stage_mix(
        *,
        target_queries: int,
        proxy_cap: int,
        synth_cap: int,
        proxy_weight: float,
        synth_weight: float,
    ) -> Tuple[int, int]:
        target = max(0, int(target_queries))
        proxy_cap_i = max(0, int(proxy_cap))
        synth_cap_i = max(0, int(synth_cap))
        if target <= 0 or (proxy_cap_i <= 0 and synth_cap_i <= 0):
            return 0, 0

        if proxy_cap_i <= 0:
            return 0, min(target, synth_cap_i)
        if synth_cap_i <= 0:
            return min(target, proxy_cap_i), 0

        p_w = max(0.0, float(proxy_weight))
        s_w = max(0.0, float(synth_weight))
        if (p_w + s_w) <= 0.0:
            p_w = 1.0
            s_w = 1.0

        proxy_target = int(round(float(target) * p_w / (p_w + s_w)))
        proxy_take = min(proxy_cap_i, proxy_target)
        synth_take = min(synth_cap_i, max(0, target - proxy_take))

        remaining = max(0, target - proxy_take - synth_take)
        if remaining > 0:
            synth_spare = max(0, synth_cap_i - synth_take)
            add_synth = min(remaining, synth_spare)
            synth_take += add_synth
            remaining -= add_synth
        if remaining > 0:
            proxy_spare = max(0, proxy_cap_i - proxy_take)
            add_proxy = min(remaining, proxy_spare)
            proxy_take += add_proxy

        return int(proxy_take), int(synth_take)

    def _iter_proxy_epoch_batches(
        self, *, device: str, batch_size: int, subset_size: Optional[int] = None
    ) -> Iterator[torch.Tensor]:
        if self.proxy_data is None or self.proxy_data.size(0) == 0:
            return

        n_total = int(self.proxy_data.size(0))
        n = n_total if subset_size is None else min(int(subset_size), n_total)
        if n <= 0:
            return

        proxy = self.proxy_data[:n]
        perm = torch.randperm(n, device=proxy.device)
        bs = max(1, int(batch_size))
        for start in range(0, n, bs):
            idx = perm[start : start + bs]
            batch = proxy.index_select(0, idx)
            yield batch.to(device)

    def _oracle_hard_labels(self, ctx: BenchmarkContext, x_query: torch.Tensor) -> torch.Tensor:
        """Query oracle and return hard top-1 labels on x_query's device."""

        oracle_output = ctx.query(x_query)
        self._publish_substitute()
        self._maybe_periodic_eval(str(x_query.device))
        if oracle_output.kind == "hard_top1":
            y = oracle_output.y
        else:
            y = torch.argmax(oracle_output.y, dim=1)

        y = y.detach()
        if y.device != x_query.device:
            y = y.to(x_query.device)
        return y.long()

    def _augment_pad_crop_hflip(self, x: torch.Tensor, *, padding: int = 4) -> torch.Tensor:
        """GPU-friendly pad+random-crop + horizontal flip (tensor-only).

        Matches the intent of official repo's `--pad_crop` option in train_student.py.
        """

        if x.ndim != 4:
            return x
        n, _c, h, w = x.shape
        if n <= 0:
            return x

        x_aug = x
        if int(padding) > 0:
            x_pad = F.pad(x_aug, (padding, padding, padding, padding), mode="constant", value=0.0)
            # windows: (N, C, H_out, W_out, H, W)
            windows = x_pad.unfold(2, h, 1).unfold(3, w, 1)
            h_out = int(windows.size(2))
            w_out = int(windows.size(3))
            top = torch.randint(0, h_out, (n,), device=x.device)
            left = torch.randint(0, w_out, (n,), device=x.device)
            batch = torch.arange(n, device=x.device)
            x_aug = windows[batch, :, top, left, :, :]

        # Random horizontal flip
        flip_mask = torch.rand(n, device=x.device) < 0.5
        if bool(flip_mask.any()):
            x_aug = x_aug.clone()
            x_aug[flip_mask] = x_aug[flip_mask].flip(-1)

        return x_aug

    def _build_auto_augment(self):
        if not self.alternate_auto_augment:
            return None
        try:
            from torchvision.transforms import AutoAugment, AutoAugmentPolicy

            policy = {
                "cifar10": AutoAugmentPolicy.CIFAR10,
                "imagenet": AutoAugmentPolicy.IMAGENET,
                "svhn": AutoAugmentPolicy.SVHN,
            }.get(self.auto_augment_policy, AutoAugmentPolicy.CIFAR10)
            return AutoAugment(policy)
        except Exception as exc:  # pragma: no cover
            self.logger.warning("[DFMSHL] AutoAugment unavailable, disabled: %s", str(exc))
            return None

    def _augment_auto_augment(self, x: torch.Tensor) -> torch.Tensor:
        if self._auto_augment is None:
            return x
        if x.ndim != 4 or x.size(0) <= 0:
            return x

        x_unit = clamp_unit(x)
        augmented: List[torch.Tensor] = []
        for i in range(int(x_unit.size(0))):
            img = x_unit[i].detach().cpu()
            # torchvision AutoAugment may apply Equalize, which only supports uint8 tensors.
            # Convert [0,1] float tensor -> uint8 before augmentation, then back to float.
            img_u8 = (img * 255.0).round().clamp(0.0, 255.0).to(torch.uint8)
            out = self._auto_augment(img_u8)
            if not isinstance(out, torch.Tensor):
                return x
            if out.dtype == torch.uint8:
                out = out.to(dtype=torch.float32).div(255.0)
            else:
                out = clamp_unit(out.to(dtype=torch.float32))
            augmented.append(out)

        x_aug = torch.stack(augmented, dim=0)
        return x_aug.to(x.device)

    def _reset_clone_for_stage(
        self,
        *,
        device: str,
        lr: float,
        cosine_t_max: int,
        keep_weights: bool,
    ) -> None:
        """(Re)initialize clone optimizer/scheduler for a stage.

        Official scripts create new optimizers for each stage. For stage transitions that
        load a pretrained student, they keep weights and reset optimizer/scheduler.
        """

        if not keep_weights or self.clone is None:
            input_shape = self.state.metadata.get("input_shape", (3, 32, 32))
            sub_config = self.state.metadata.get("substitute_config", {})
            arch = sub_config.get("arch") or self.config.get("clone_arch", "resnet18-8x")
            width_mult = int(sub_config.get("width_mult", 1))
            dropout_prob = float(sub_config.get("dropout_prob", 0.0))

            base_clone = create_substitute(
                arch=arch,
                num_classes=self.num_classes,
                input_channels=int(input_shape[0]),
                width_mult=width_mult,
                dropout_prob=dropout_prob,
            ).to(device)

            # Clone internal path uses fixed tanh scale.
            self.clone = base_clone
            self._eval_substitute = None

        # Config-driven optimizer construction (consistent with other data-free
        # attacks). Defaults preserve paper-canonical SGD baseline: when nothing
        # in `sub_config.optimizer` overrides, we use SGD with the stage's
        # `lr` argument and DFMS-specific momentum/weight_decay.
        sub_config = self.state.metadata.get("substitute_config", {}) or {}
        opt_spec = dict(sub_config.get("optimizer") or {})
        opt_spec.setdefault("name", "sgd")
        opt_spec.setdefault("lr", float(lr))
        opt_spec.setdefault("momentum", float(self.student_momentum))
        opt_spec.setdefault("weight_decay", float(self.student_weight_decay))
        self.clone_optimizer = self._build_optimizer(
            self.clone.parameters(), opt_spec
        )
        self.clone_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.clone_optimizer,
            T_max=max(1, int(cosine_t_max)),
        )

    def _generator_adv_only_step(self, *, fake_x_01: torch.Tensor) -> None:
        """DCGAN-style generator update (non-saturating BCE(fake->real))."""

        if self.generator is None or self.discriminator is None:
            return
        self.generator_optimizer.zero_grad(set_to_none=True)
        fake_logits = self.discriminator(self._unit_to_internal(fake_x_01))
        loss_g = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
        loss_g.backward()
        self.generator_optimizer.step()

    def _generator_adv_div_step(
        self, *, fake_x_01: torch.Tensor, diversity_weight: float
    ) -> None:
        """Generator update with adversarial + diversity loss (official DeGAN/alternate)."""

        if self.generator is None or self.discriminator is None or self.clone is None:
            return

        self.generator_optimizer.zero_grad(set_to_none=True)
        fake_logits = self.discriminator(self._unit_to_internal(fake_x_01))
        adv_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))

        # Diversity term uses clone as proxy gradient.
        clone_was_training = self.clone.training
        self.clone.eval()
        prev_requires_grad = [p.requires_grad for p in self.clone.parameters()]
        for p in self.clone.parameters():
            p.requires_grad_(False)
        try:
            clone_logits = self.clone(self._unit_to_internal(fake_x_01))
        finally:
            for p, rg in zip(self.clone.parameters(), prev_requires_grad, strict=False):
                p.requires_grad_(rg)
            if clone_was_training:
                self.clone.train()

        probs = F.softmax(clone_logits, dim=1)
        alpha = probs.mean(dim=0)
        div_loss = torch.sum(alpha * torch.log(alpha + 1e-6))

        loss = adv_loss + float(diversity_weight) * div_loss
        loss.backward()
        self.generator_optimizer.step()

    def _official_stage_dcgan_pretrain(self, device: str) -> None:
        """Stage 1: pretrain DCGAN on proxy data (official dcgan.py).

        Official script reference:
        - temp_dfms_hl/run_cifar10_rand_class_resnet.sh: calls dcgan.py with niter=200, batchSize=64
        - temp_dfms_hl/code/train_generator/dcgan.py: DCGAN training on proxy data
        """

        if self.generator is None or self.discriminator is None or self.proxy_data is None:
            self.logger.info("[DFMSHL] Stage 1/5 skipped (models/proxy not initialized)")
            return
        if int(self.dcgan_epochs) <= 0:
            self.logger.info("[DFMSHL] Stage 1/5 skipped (dcgan_epochs<=0)")
            return

        subset_n = self._official_proxy_subset_size()
        if subset_n <= 0:
            self.logger.info("[DFMSHL] Stage 1/5 skipped (proxy_subset_size=0)")
            return

        self.generator.train()
        self.discriminator.train()

        batch_size = int(self.batch_size)
        for _ep in tqdm(
            range(int(self.dcgan_epochs)),
            desc="[DFMSHL] Stage 1/5 DCGAN pretrain (epochs)",
            leave=False,
            position=1,
            mininterval=1.0,
        ):
            for real_x in self._iter_proxy_epoch_batches(
                device=device, batch_size=batch_size, subset_size=subset_n
            ):
                if self.proxy_pad_crop:
                    real_x = self._augment_pad_crop_hflip(real_x)
                z = torch.randn(real_x.size(0), self.noise_dim, device=device)
                fake_x = tanh_to_unit(self.generator(z))
                self._train_discriminator(real_x, fake_x)

                z2 = torch.randn(real_x.size(0), self.noise_dim, device=device)
                fake_x_2 = tanh_to_unit(self.generator(z2))
                self._generator_adv_only_step(fake_x_01=fake_x_2)

    def _official_stage_train_degan(self, device: str) -> None:
        """Stage 3: train DeGAN/DivGAN generator with diversity loss (official train_gen.py)."""

        if self.generator is None or self.discriminator is None or self.clone is None or self.proxy_data is None:
            self.logger.info("[DFMSHL] Stage 3/5 skipped (models/proxy not initialized)")
            return
        if int(self.degan_epochs) <= 0:
            self.logger.info("[DFMSHL] Stage 3/5 skipped (degan_epochs<=0)")
            return

        subset_n = self._official_proxy_subset_size()
        if subset_n <= 0:
            self.logger.info("[DFMSHL] Stage 3/5 skipped (proxy_subset_size=0)")
            return

        self.generator.train()
        self.discriminator.train()
        # Clone acts as a fixed proxy (netC.eval() in official code).
        self.clone.eval()

        batch_size = int(self.batch_size)
        for _ep in tqdm(
            range(int(self.degan_epochs)),
            desc="[DFMSHL] Stage 3/5 DeGAN train (epochs)",
            leave=False,
            position=1,
            mininterval=1.0,
        ):
            for real_x in self._iter_proxy_epoch_batches(
                device=device, batch_size=batch_size, subset_size=subset_n
            ):
                if self.proxy_pad_crop:
                    real_x = self._augment_pad_crop_hflip(real_x)
                # Official train_gen.py updates D first, then G (with d_l=10 by default).
                z = torch.randn(real_x.size(0), self.noise_dim, device=device)
                fake_x = tanh_to_unit(self.generator(z))
                self._train_discriminator(real_x, fake_x)
                self._generator_adv_div_step(
                    fake_x_01=fake_x,
                    diversity_weight=float(self.degan_diversity_weight),
                )

    def _official_stage_student_init(
        self,
        ctx: BenchmarkContext,
        device: str,
        *,
        pbar: tqdm,
        synth_ratio: float,
        stage_name: str,
        train_epochs: int,
        student_lr: float,
        target_queries: Optional[int] = None,
    ) -> None:
        """Stage 2/4: student init on proxy + generator images (official train_student.py).

        Notes:
        - Official repo labels a fixed buffer of proxy+GAN images using the teacher.
        - In this benchmark, labeling MUST go through the oracle (budget accounting).
        """

        if self.generator is None or self.proxy_data is None:
            return
        if ctx.budget_remaining <= 0:
            return
        if int(train_epochs) <= 0:
            return

        proxy_cap = self._official_proxy_subset_size()
        synth_cap = int(max(0, int(float(synth_ratio) * int(self.max_synth_samples))))
        if self.budget_planner_mode == "paper_fair":
            synth_cap = int(self.max_synth_samples)

        effective_target = int(target_queries) if target_queries is not None else int(proxy_cap + synth_cap)
        effective_target = min(effective_target, int(ctx.budget_remaining))

        proxy_n, synth_n = self._allocate_stage_mix(
            target_queries=effective_target,
            proxy_cap=proxy_cap,
            synth_cap=synth_cap,
            proxy_weight=float(self.proxy_data_ratio),
            synth_weight=float(synth_ratio),
        )

        if proxy_n <= 0 and synth_n <= 0:
            return

        target_total = int(proxy_n + synth_n)
        if target_total <= 0:
            return

        # Prepare fixed proxy indices (first proxy_n samples, matching official deterministic fill).
        proxy_indices = torch.arange(proxy_n, dtype=torch.long)
        # Sample synthetic noise on CPU (seeded) and move once to device.
        # This avoids per-step CPU->GPU transfers while keeping the same RNG source.
        z_synth = torch.randn(synth_n, self.noise_dim, dtype=torch.float32).to(device)

        # Query labels for proxy samples.
        y_proxy_list: List[torch.Tensor] = []
        oracle_bs = max(1, int(self.oracle_batch_size))
        proxy_cursor = 0
        while proxy_cursor < proxy_n and ctx.budget_remaining > 0:
            take = min(
                oracle_bs,
                proxy_n - proxy_cursor,
                int(ctx.budget_remaining),
            )
            if take <= 0:
                break
            x = self.proxy_data[proxy_cursor : proxy_cursor + take].to(device)
            y = self._oracle_hard_labels(
                ctx,
                self._query_scale_generated_from_unit(x),
            )
            y_proxy_list.append(y)
            pbar.update(int(take))
            self._publish_substitute()
            self._maybe_periodic_eval(device)
            proxy_cursor += int(take)

        # Query labels for synthetic samples.
        y_synth_list: List[torch.Tensor] = []
        synth_cursor = 0
        self.generator.eval()
        while synth_cursor < synth_n and ctx.budget_remaining > 0:
            take = min(
                oracle_bs,
                synth_n - synth_cursor,
                int(ctx.budget_remaining),
            )
            if take <= 0:
                break
            z = z_synth[synth_cursor : synth_cursor + take]
            with torch.no_grad():
                x_tanh = self.generator(z)
                x_query = self._query_scale_generated_from_tanh(x_tanh)
            y = self._oracle_hard_labels(ctx, x_query)
            y_synth_list.append(y)
            pbar.update(int(take))
            self._publish_substitute()
            self._maybe_periodic_eval(device)
            synth_cursor += int(take)

        if len(y_proxy_list) == 0 and len(y_synth_list) == 0:
            return

        # Truncate to what we actually labeled.
        proxy_labeled = int(sum(int(t.size(0)) for t in y_proxy_list))
        synth_labeled = int(sum(int(t.size(0)) for t in y_synth_list))
        proxy_indices = proxy_indices[:proxy_labeled]
        z_synth = z_synth[:synth_labeled]
        y_proxy = (
            torch.cat(y_proxy_list, dim=0)
            if y_proxy_list
            else torch.empty(0, dtype=torch.long, device=device)
        )
        y_synth = (
            torch.cat(y_synth_list, dim=0)
            if y_synth_list
            else torch.empty(0, dtype=torch.long, device=device)
        )

        self.logger.info(
            "[DFMSHL] %s labeled: proxy=%d synth=%d target=%d (query_used=%d remaining=%d)",
            str(stage_name),
            int(proxy_labeled),
            int(synth_labeled),
            int(effective_target),
            int(ctx.query_count),
            int(ctx.budget_remaining),
        )

        # Reset clone from scratch and train as in official train_student.py (SGD + cosine).
        self._reset_clone_for_stage(
            device=device,
            lr=float(student_lr),
            cosine_t_max=int(train_epochs),
            keep_weights=False,
        )

        # Official train_student.py uses --pad_crop in scripts.
        use_pad_crop = bool(self.config.get("student_pad_crop", True))

        proxy_batch = int(max(0, int(self.student_batch_size * float(self.proxy_data_ratio))))
        synth_batch = int(max(0, int(self.student_batch_size * float(synth_ratio))))
        if proxy_batch == 0 and proxy_indices.numel() > 0:
            proxy_batch = min(int(self.student_batch_size), int(proxy_indices.numel()))
        if synth_batch == 0 and z_synth.numel() > 0:
            synth_batch = int(self.student_batch_size)

        # Steps per epoch match official: min(iters_proxy, iters_gan).
        steps = []
        if proxy_batch > 0 and int(proxy_indices.numel()) > 0:
            steps.append(int(math.ceil(int(proxy_indices.numel()) / float(proxy_batch))))
        if synth_batch > 0 and int(z_synth.size(0)) > 0:
            steps.append(int(math.ceil(int(z_synth.size(0)) / float(synth_batch))))
        steps_per_epoch = int(min(steps)) if steps else 0
        if steps_per_epoch <= 0:
            return

        self.clone.train()
        for ep in tqdm(
            range(int(train_epochs)),
            desc=f"[DFMSHL] {stage_name} train (epochs)",
            leave=False,
            position=1,
            mininterval=1.0,
        ):
            proxy_perm = torch.randperm(int(proxy_indices.numel())) if proxy_indices.numel() > 0 else None
            synth_perm = torch.randperm(int(z_synth.size(0))) if z_synth.size(0) > 0 else None
            p_ptr = 0
            s_ptr = 0

            for _ in range(steps_per_epoch):
                x_parts: List[torch.Tensor] = []
                y_parts: List[torch.Tensor] = []

                if proxy_perm is not None and proxy_batch > 0 and p_ptr < proxy_perm.numel():
                    sel = proxy_perm[p_ptr : p_ptr + proxy_batch]
                    p_ptr += int(sel.numel())
                    idx = proxy_indices.index_select(0, sel).to(self.proxy_data.device)
                    x_p = self.proxy_data.index_select(0, idx).to(device)
                    sel_dev = sel.to(y_proxy.device) if y_proxy.device != sel.device else sel
                    y_p = y_proxy.index_select(0, sel_dev)
                    if y_p.device != x_p.device:
                        y_p = y_p.to(x_p.device)
                    x_parts.append(x_p)
                    y_parts.append(y_p)

                if synth_perm is not None and synth_batch > 0 and s_ptr < synth_perm.numel():
                    sel = synth_perm[s_ptr : s_ptr + synth_batch]
                    s_ptr += int(sel.numel())
                    sel_dev = sel.to(z_synth.device) if z_synth.device != sel.device else sel
                    z_b = z_synth.index_select(0, sel_dev)
                    with torch.no_grad():
                        x_s = tanh_to_unit(self.generator(z_b))
                    sel_y = sel.to(y_synth.device) if y_synth.device != sel.device else sel
                    y_s = y_synth.index_select(0, sel_y)
                    if y_s.device != x_s.device:
                        y_s = y_s.to(x_s.device)
                    x_parts.append(x_s)
                    y_parts.append(y_s)

                if not x_parts:
                    continue

                x_b = torch.cat(x_parts, dim=0)
                y_b = torch.cat(y_parts, dim=0).long().view(-1)
                if use_pad_crop:
                    x_b = self._augment_pad_crop_hflip(x_b)
                x_b = self._unit_to_internal(x_b)

                self.clone_optimizer.zero_grad(set_to_none=True)
                logits = self.clone(x_b)
                if self.is_single_logit_binary:
                    loss = binary_bce_loss(logits, y_b.float().unsqueeze(1))
                else:
                    loss = F.cross_entropy(logits, y_b)
                loss.backward()
                self.clone_optimizer.step()

            if self.clone_scheduler is not None:
                self.clone_scheduler.step()

        # Track substitute for final evaluation.
        self.state.attack_state["substitute"] = self.clone

    def _required_alternate_epochs(self, budget_remaining: int, subset_size: int) -> int:
        """Return minimum alternate epochs needed to consume the remaining budget."""

        budget = max(0, int(budget_remaining))
        subset_n = max(1, int(subset_size))
        if budget <= 0:
            return 0
        return int((budget + subset_n - 1) // subset_n)

    def _official_stage_alternate(
        self,
        ctx: BenchmarkContext,
        device: str,
        *,
        pbar: tqdm,
        target_queries: Optional[int] = None,
        planned_epochs: Optional[int] = None,
    ) -> None:
        """Stage 5: alternate training (official train_generator_clone.py).

        Paper vs official note:
        - Paper writes minimax adversarial term with log(1-D(G(z))) (Eq.(3)), but official
          uses non-saturating BCE(fake->real) for generator updates.
        - Official code feeds images normalized to [-1,1] into the teacher.
        - This implementation uses fixed tanh query routing for synthetic queries.
        - Budget-aware epoch scheduling is applied to avoid early termination on
          small surrogate subsets while preserving the official fixed-stage structure.
        """

        if self.generator is None or self.discriminator is None or self.clone is None or self.proxy_data is None:
            return

        subset_n = self._official_proxy_subset_size()
        if subset_n <= 0:
            return

        stage_target = (
            int(ctx.budget_remaining)
            if target_queries is None
            else min(int(target_queries), int(ctx.budget_remaining))
        )
        if stage_target <= 0:
            return

        total_epochs = (
            int(planned_epochs)
            if planned_epochs is not None
            else self._required_alternate_epochs(stage_target, int(subset_n))
        )
        if total_epochs <= 0:
            return

        # Reset optimizer/scheduler for alternate stage (keep weights).
        self._reset_clone_for_stage(
            device=device,
            lr=float(self.student_alt_lr),
            cosine_t_max=int(total_epochs),
            keep_weights=True,
        )

        use_pad_crop = bool(self.config.get("alternate_pad_crop", False))

        self.generator.train()
        self.discriminator.train()

        batch_size = int(self.batch_size)
        stage_remaining = int(stage_target)
        ep = 0
        with tqdm(
            total=int(total_epochs),
            desc="[DFMSHL] Stage 5/5 alternate (epochs)",
            leave=False,
            position=1,
            mininterval=1.0,
        ) as epbar:
            while stage_remaining > 0 and ctx.budget_remaining > 0:
                if ep >= int(total_epochs) and self.budget_planner_mode == "legacy_fixed_epochs":
                    break

                epoch_queries = 0

                for real_x in self._iter_proxy_epoch_batches(
                    device=device, batch_size=batch_size, subset_size=subset_n
                ):
                    if ctx.budget_remaining <= 0 or stage_remaining <= 0:
                        break

                    # Budget-capped microbatch
                    b = min(int(real_x.size(0)), int(ctx.budget_remaining), int(stage_remaining))
                    if b <= 0:
                        break
                    real_x = real_x[:b]
                    if self.proxy_pad_crop:
                        real_x = self._augment_pad_crop_hflip(real_x)

                    # 1) Generate synthetic batch (keep graph for G update)
                    z = torch.randn(b, self.noise_dim, device=device)
                    fake_raw = self.generator(z)
                    fake_x = tanh_to_unit(fake_raw)

                    # 2) Student/clone update: label with oracle and fit on (optionally) augmented inputs
                    x_for_student = fake_x.detach()
                    if use_pad_crop:
                        x_for_student = self._augment_pad_crop_hflip(x_for_student)
                    if self.alternate_auto_augment:
                        x_for_student = self._augment_auto_augment(x_for_student)

                    y = self._oracle_hard_labels(
                        ctx,
                        self._query_scale_generated_from_unit(x_for_student),
                    )
                    pbar.update(int(b))
                    self._publish_substitute()
                    self._maybe_periodic_eval(device)
                    stage_remaining -= int(b)
                    epoch_queries += int(b)

                    self.clone.train()
                    self.clone_optimizer.zero_grad(set_to_none=True)
                    logits = self.clone(self._unit_to_internal(x_for_student))
                    if self.is_single_logit_binary:
                        loss_s = binary_bce_loss(logits, y.float().unsqueeze(1))
                    else:
                        loss_s = F.cross_entropy(logits, y)
                    loss_s.backward()
                    self.clone_optimizer.step()

                    # 3) Generator update (official order: after student, before discriminator)
                    self._generator_adv_div_step(
                        fake_x_01=fake_x,
                        diversity_weight=float(self.alternate_diversity_weight),
                    )

                    # 4) Discriminator update
                    self._train_discriminator(real_x, fake_x)

                # End of epoch: cosine step + warmup (official)
                if self.clone_scheduler is not None:
                    self.clone_scheduler.step()
                if self.use_student_warmup and ep < int(self.student_warmup_epochs):
                    for param_group in self.clone_optimizer.param_groups:
                        param_group["lr"] = float(self.student_warmup_init_lr) * float(ep)

                ep += 1
                epbar.update(1)

                if epoch_queries <= 0:
                    break

        self.state.attack_state["substitute"] = self.clone

    def _select_query_batch(self, k: int, state: BenchmarkState) -> QueryBatch:
        self._init_models(state)
        device = state.metadata.get("device", "cpu")
        phase = state.attack_state["phase"]

        if phase == "init_collect":
            k_proxy = k // 2
            k_synth = k - k_proxy

            x_proxy = self._query_scale_generated_from_unit(
                self._next_proxy_batch(device, batch_size=k_proxy)
            )

            z = torch.randn(k_synth, self.noise_dim, device=device)
            with torch.no_grad():
                x_synth_tanh = self.generator(z)
                x_synth = self._query_scale_generated_from_tanh(x_synth_tanh)

            x = torch.cat([x_proxy, x_synth], dim=0)
            meta = {"phase": phase, "k_proxy": k_proxy, "k_synth": k_synth}
            return QueryBatch(x=x, meta=meta)

        if phase == "init_retrain_collect":
            z = torch.randn(k, self.noise_dim, device=device)
            with torch.no_grad():
                x = self._query_scale_generated_from_tanh(self.generator(z))
            meta = {"phase": phase, "synthetic": True}
            return QueryBatch(x=x, meta=meta)

        z = torch.randn(k, self.noise_dim, device=device)
        with torch.no_grad():
            x_raw = self.generator(z)
        x = self._query_scale_generated_from_tanh(x_raw)
        meta = {"generator_step": state.attack_state["step"], "synthetic": True, "phase": phase}
        return QueryBatch(x=x, meta=meta)

    def _handle_oracle_output(
        self,
        x_query: torch.Tensor,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        if self.generator is None or self.discriminator is None or self.clone is None:
            return

        device = x_query.device

        if oracle_output.kind == "hard_top1":
            hard_labels = oracle_output.y
        else:
            if self.is_single_logit_binary:
                hard_labels = binary_hard_labels_from_positive_probs(oracle_output.y)
            else:
                hard_labels = torch.argmax(oracle_output.y, dim=1)

        hard_labels = hard_labels.to(device)
        phase = state.attack_state["phase"]

        if phase == "init_collect":
            state.attack_state["init_x"].append(x_query.cpu())
            state.attack_state["init_y"].append(hard_labels.cpu())
            state.attack_state["collected_count"] += x_query.size(0)

            if state.attack_state["collected_count"] >= self.init_nc:
                x_all = torch.cat(state.attack_state["init_x"], dim=0)[:self.init_nc].to(device)
                y_all = torch.cat(state.attack_state["init_y"], dim=0)[:self.init_nc].to(device)

                self._fine_tune_generator(x_all, y_all, self.init_ng)

                self._reset_clone()
                state.attack_state["phase"] = "init_retrain_collect"
                state.attack_state["collected_count"] = 0
                state.attack_state["init_x"] = []
                state.attack_state["init_y"] = []

        elif phase == "init_retrain_collect":
            self._train_clone(x_query, hard_labels)
            state.attack_state["collected_count"] += x_query.size(0)

            if state.attack_state["collected_count"] >= self.init_nc:
                state.attack_state["phase"] = "alternate"

        elif phase == "alternate":
            # Official repo ordering (per minibatch): update clone (student) first,
            # then update generator, then update discriminator.
            # See: temp_dfms_hl/code/train_generator/train_generator_clone.py
            # - student update: around the "loss_student" backward/step
            # - generator update: comment "maximize log(D(G(z)))"
            # - discriminator update: comment "maximize log(D(x)) + log(1 - D(G(z)))"
            self._train_clone(x_query, hard_labels)
            self._train_generator(x_query)
            real_x = self._next_proxy_batch(device)
            self._train_discriminator(real_x, x_query)

            if self.clone_scheduler is not None:
                self.clone_scheduler.step()

        state.attack_state["step"] += 1
        state.attack_state["substitute"] = self.clone

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
            if self.use_official_dcgan_arch:
                self.generator = OfficialDFMSDCGANGenerator(
                    noise_dim=self.noise_dim,
                    output_channels=int(self.config.get("output_channels", input_shape[0])),
                    base_channels=self.base_channels,
                    output_size=int(input_shape[1]),
                ).to(device)
            else:
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
            if self.use_official_dcgan_arch:
                self.discriminator = OfficialDFMSDCGANDiscriminator(
                    input_channels=int(self.config.get("output_channels", input_shape[0])),
                    base_channels=self.base_channels,
                    input_size=int(input_shape[1]),
                ).to(device)
            else:
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
            sub_config = state.metadata.get("substitute_config", {})
            arch = sub_config.get("arch") or self.config.get("clone_arch", "resnet18-8x")
            width_mult = int(sub_config.get("width_mult", 1))
            dropout_prob = float(sub_config.get("dropout_prob", 0.0))
            opt_params = sub_config.get("optimizer", {})
            
            base_clone = create_substitute(
                arch=arch,
                num_classes=self.num_classes,
                input_channels=int(input_shape[0]),
                width_mult=width_mult,
                dropout_prob=dropout_prob,
            ).to(device)

            # Clone internal path uses fixed tanh scale.
            self.clone = base_clone
            self._eval_substitute = None

            # [UNIFIED] Config-driven optimizer construction
            self.clone_optimizer = self._build_optimizer(self.clone.parameters(), opt_params)

            # [UNIFIED] Use Manual MultiStepLR logic in run() loop instead of Cosine
            self.clone_scheduler = None

        if self.proxy_data is None:
            proxy_config = self.config.get("attack", {}).get("proxy_dataset")
            if proxy_config is None:
                proxy_config = self.config.get("proxy_dataset")
            if proxy_config is None:
                # Unit/regression tests instantiate attacks with minimal configs and do not
                # provide proxy dataset specs. For those cases (budget_remaining==0), fall back
                # to a small synthetic proxy buffer so `_select_query_batch` can succeed.
                if int(getattr(state, "budget_remaining", 0)) <= 0:
                    input_shape = state.metadata.get("input_shape", (3, 32, 32))
                    c, h, w = int(input_shape[0]), int(input_shape[1]), int(input_shape[2])
                    n_proxy = max(64, int(self.batch_size) * 4)
                    self.proxy_data = torch.rand(n_proxy, c, h, w, device=device)
                    self.pretrained = True
                else:
                    raise ValueError("DFMS-HL requires proxy_dataset configuration")
            else:
                proxy_config = dict(proxy_config)
                # Keep DFMS proxy tensors in raw/unit space before internal
                # unit->tanh conversion. Surrogate-standard normalization is a
                # pool-based policy and should not alter data-free DFMS proxy flow.
                proxy_config.setdefault("surrogate_normalization", "none")
                # Cache entire proxy dataset to RAM/GPU
                self.proxy_data = load_pool_to_memory(
                    proxy_config,
                    device=device,
                    desc="[DFMSHL] Caching proxy data",
                    max_samples=100_000,
                )

        # Legacy pretrain hook. Official-stage pipeline already performs DCGAN pretraining.
        if (
            not self.use_official_stages
            and not self.pretrained
            and self.pretrain_steps > 0
            and int(getattr(state, "budget_remaining", 0)) > 0
        ):
            self._pretrain_gan(device)
            self.pretrained = True

    def _next_proxy_batch(self, device: str, batch_size: Optional[int] = None) -> torch.Tensor:
        bs = batch_size or self.batch_size
        if self.proxy_data is None or self.proxy_data.size(0) == 0:
            raise RuntimeError("Proxy data not loaded")
            
        # Random sampling from cached tensor
        indices = torch.randint(0, self.proxy_data.size(0), (bs,), device=self.proxy_data.device)
        batch = self.proxy_data[indices]
        
        # If cache is on CPU, move to target device. If on GPU, it's a no-op (or cheap copy).
        return batch.to(device)

    def _train_discriminator(self, real_x: torch.Tensor, fake_x: torch.Tensor) -> None:
        self.discriminator_optimizer.zero_grad()

        # Discriminator path uses fixed tanh scale.
        real_logits = self.discriminator(self._unit_to_internal(real_x))
        fake_logits = self.discriminator(self._unit_to_internal(fake_x.detach()))
        real_labels = torch.ones_like(real_logits)
        fake_labels = torch.zeros_like(fake_logits)
        loss_real = F.binary_cross_entropy_with_logits(real_logits, real_labels)
        loss_fake = F.binary_cross_entropy_with_logits(fake_logits, fake_labels)
        loss = loss_real + loss_fake
        loss.backward()
        self.discriminator_optimizer.step()

    def _train_generator(self, fake_x: torch.Tensor) -> None:
        # Regenerate to maintain gradient graph from z to G(z)
        z = torch.randn(int(fake_x.size(0)), self.noise_dim, device=fake_x.device)
        fake_x_gen_raw = self.generator(z)
        fake_x_gen = tanh_to_unit(fake_x_gen_raw)  # [-1, 1] -> [0, 1]

        self.generator_optimizer.zero_grad()

        # Paper adversarial generator term is written as:
        #   L_adv,fake = E_z[ log(1 - D(G(z))) ]  (DFMS.pdf Eq.(3))
        # and the minimax game: min_G max_D (Eq.(4)).
        # However, the *official reference implementation* optimizes the non-saturating
        # generator objective by setting fake labels as real and minimizing BCE:
        #   minimize_G BCE(D(G(z)), 1)  <=>  maximize log(D(G(z)))
        # See: temp_dfms_hl/code/train_generator/train_generator_clone.py
        #   - comment: "Update G network: maximize log(D(G(z)))" (around line 624)
        #   - errG_adv = criterion(output, label) with label=real_label (around line 663-674)
        fake_logits = self.discriminator(self._unit_to_internal(fake_x_gen))
        adv_targets = torch.ones_like(fake_logits)
        adv_loss = F.binary_cross_entropy_with_logits(fake_logits, adv_targets)
        
        # Paper protocol: during generator update, clone C is fixed.
        # Freeze clone parameters and BN/Dropout state while keeping gradients w.r.t. input.
        clone_was_training = self.clone.training
        self.clone.eval()
        prev_requires_grad = [p.requires_grad for p in self.clone.parameters()]
        for p in self.clone.parameters():
            p.requires_grad_(False)
        try:
            clone_logits = self.clone(self._unit_to_internal(fake_x_gen))
        finally:
            for p, rg in zip(self.clone.parameters(), prev_requires_grad, strict=False):
                p.requires_grad_(rg)
            if clone_was_training:
                self.clone.train()
        if self.is_single_logit_binary:
            probs = binary_distribution_from_logits(clone_logits)
        else:
            probs = F.softmax(clone_logits, dim=1)
        
        # Diversity (paper Eq.(5)-(7)): use batch-mean class distribution alpha.
        # Paper sets div=500 for CIFAR-10 and div=100 for CIFAR-100 (DFMS.pdf ablation).
        alpha = probs.mean(dim=0)
        class_div = torch.sum(alpha * torch.log(alpha + 1e-6))

        loss = adv_loss + self.diversity_weight * class_div
        loss.backward()
        self.generator_optimizer.step()

    def _pretrain_gan(self, device: str) -> None:
        pre_pbar = tqdm(
            range(self.pretrain_steps),
            desc="[DFMSHL] Pre-training GAN",
            leave=False,
            disable=bool(self.config.get("disable_stage_pbar", False)),
        )
        for _ in pre_pbar:
            real_x = self._next_proxy_batch(device)
            z = torch.randn(real_x.size(0), self.noise_dim, device=device)
            fake_x = tanh_to_unit(self.generator(z))
            
            self._train_discriminator(real_x, fake_x)
            
            z2 = torch.randn(real_x.size(0), self.noise_dim, device=device)
            fake_x_2 = tanh_to_unit(self.generator(z2))
            
            self.generator_optimizer.zero_grad()
            # Align pretraining with official repo behavior (non-saturating generator loss).
            # Paper writes minimax L_adv,fake as log(1 - D(G(z))) (Eq.(3)), but the official
            # repo uses BCE(fake->real) for generator updates.
            fake_logits = self.discriminator(self._unit_to_internal(clamp_unit(fake_x_2)))
            loss_g = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
            loss_g.backward()
            self.generator_optimizer.step()
            pre_pbar.set_postfix({"Loss G": f"{loss_g.item():.4f}"})

    def _train_clone(self, x_fake: torch.Tensor, hard_labels: torch.Tensor) -> None:
        device = x_fake.device
        x_cpu = x_fake.detach().cpu()
        y_cpu = hard_labels.detach().to(dtype=torch.long).cpu()

        # Batching to avoid OOM for large buffers during init.
        # Benchmark scaling unification: keep clone inputs in [0,1].
        dataset = torch.utils.data.TensorDataset(x_cpu, y_cpu)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        for batch_x_cpu, batch_y_cpu in loader:
            batch_x = batch_x_cpu.to(device)
            batch_y = batch_y_cpu.to(device)
            self.clone_optimizer.zero_grad()
            logits = self.clone(self._unit_to_internal(batch_x))
            if self.is_single_logit_binary:
                loss = binary_bce_loss(logits, batch_y.float().unsqueeze(1))
            else:
                loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            self.clone_optimizer.step()

    def _reset_clone(self) -> None:
        device = self.state.metadata.get("device", "cpu")
        input_shape = self.state.metadata.get("input_shape", (3, 32, 32))
        sub_config = self.state.metadata.get("substitute_config", {})
        arch = sub_config.get("arch") or self.config.get("clone_arch", "resnet18-8x")
        width_mult = int(sub_config.get("width_mult", 1))
        dropout_prob = float(sub_config.get("dropout_prob", 0.0))
        opt_params = sub_config.get("optimizer", {})

        base_clone = create_substitute(
            arch=arch,
            num_classes=self.num_classes,
            input_channels=int(input_shape[0]),
            width_mult=width_mult,
            dropout_prob=dropout_prob,
        ).to(device)

        # Benchmark scaling unification (DFME-style): clone consumes [0,1] inputs.
        self.clone = base_clone
        self._eval_substitute = None
        # Config-driven optimizer (consistent with `_reset_clone_for_stage`).
        opt_spec = dict(opt_params)
        opt_spec.setdefault("name", "sgd")
        opt_spec.setdefault("lr", float(self.clone_lr))
        opt_spec.setdefault("momentum", 0.9)
        opt_spec.setdefault("weight_decay", 5e-4)
        self.clone_optimizer = self._build_optimizer(self.clone.parameters(), opt_spec)

        if self.use_clone_cosine:
            max_budget = self.state.metadata.get("max_budget", 1000)
            t_max = max(1, int(max_budget / self.batch_size))
            self.clone_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.clone_optimizer, t_max
            )

    def _fine_tune_generator(self, x_collected: torch.Tensor, y_collected: torch.Tensor, epochs: int) -> None:
        device = x_collected.device
        # Train clone first on collected data
        self.logger.debug(
            "Fine-tuning: Training clone for 50 epochs on %s samples...",
            x_collected.size(0),
        )
        for _ in tqdm(
            range(50),
            desc="[DFMSHL] Fine-tuning Clone",
            leave=False,
            disable=bool(self.config.get("disable_stage_pbar", False)),
        ):
            self._train_clone(x_collected, y_collected)
        
        # Then tune G
        self.logger.debug("Fine-tuning: Training generator for %s epochs...", epochs)
        self.generator.train()
        for i in tqdm(
            range(epochs),
            desc="[DFMSHL] Fine-tuning Generator",
            leave=False,
            disable=bool(self.config.get("disable_stage_pbar", False)),
        ):
            real_x = self._next_proxy_batch(device)
            z = torch.randn(self.batch_size, self.noise_dim, device=device)
            fake_x = tanh_to_unit(self.generator(z))
            
            self._train_discriminator(real_x, fake_x)
            self._train_generator(fake_x)
            # if (i+1) % 10 == 0 or i == epochs - 1:
            #     print(f"  Epoch {i+1}/{epochs} done")
