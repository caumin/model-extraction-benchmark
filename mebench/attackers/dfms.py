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
from mebench.models.gan import DCGANGenerator, DCGANDiscriminator
from mebench.models.substitute_factory import create_substitute
from mebench.data.loaders import create_dataloader
from mebench.utils.dataloader import load_pool_to_memory
from mebench.utils.scaling import tanh_to_unit, clamp_unit


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

        self.generator: nn.Module | None = None
        self.discriminator: nn.Module | None = None
        self.clone: nn.Module | None = None
        self.generator_optimizer: optim.Optimizer | None = None
        self.discriminator_optimizer: optim.Optimizer | None = None
        self.clone_optimizer: optim.Optimizer | None = None
        self.clone_scheduler: optim.lr_scheduler.CosineAnnealingLR | None = None
        self.proxy_data: torch.Tensor | None = None
        self.pretrained = False

        self._initialize_state(state)

    # NOTE (Benchmark scaling unification vs DFMS.pdf):
    # Many DFMS/DFMS-HL implementations normalize images to [-1,1] using (x-0.5)/0.5.
    # This benchmark enforces a global contract that oracle/eval inputs are in [0,1]
    # with no additional mean/std normalization. To keep all attacks comparable, this
    # implementation uses [0,1] as the canonical image scale throughout the attack.

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
            # Legacy/paper-style loop (kept for backwards-compat and unit tests).
            while ctx.budget_remaining > 0:
                step_size = self._default_step_size(ctx)
                query_batch = self._select_query_batch(step_size, self.state)
                oracle_output = ctx.query(query_batch.x, meta=query_batch.meta)
                self._handle_oracle_output(query_batch.x, oracle_output, self.state)
                pbar.update(query_batch.x.size(0))
            pbar.close()
            return

        # Official-repo aligned stage pipeline (see `temp_dfms_hl/run_*.sh`).
        self.logger.info("[DFMSHL] Initializing models + caching proxy data (if needed)...")
        self._init_models(self.state)

        if isinstance(self.proxy_data, torch.Tensor) and self.proxy_data.numel() > 0:
            self.logger.info(
                "[DFMSHL] Proxy cache ready: shape=%s device=%s",
                tuple(int(x) for x in self.proxy_data.shape),
                str(self.proxy_data.device),
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
            self._official_stage_student_init(
                ctx,
                device,
                pbar=pbar,
                synth_ratio=self.dcgan_data_ratio,
                stage_name="student_init_dcgan",
                train_epochs=self.student_init_epochs,
                student_lr=self.student_init_lr,
            )

        # Stage 3: train DeGAN/DivGAN (no oracle queries).
        ctx.log_event("dfmshl_stage", {"stage": "train_degan", "queries": 0})
        self.logger.info("[DFMSHL] Stage 3/5: train DeGAN/DivGAN (no queries)")
        self._official_stage_train_degan(device)

        # Stage 4: student init with proxy + DeGAN images (oracle labeling consumes budget).
        if ctx.budget_remaining > 0:
            ctx.log_event("dfmshl_stage", {"stage": "student_init_degan", "queries": "labels"})
            self.logger.info("[DFMSHL] Stage 4/5: student init (proxy + DeGAN) (queries for labels)")
            self._official_stage_student_init(
                ctx,
                device,
                pbar=pbar,
                synth_ratio=self.div_gan_data_ratio,
                stage_name="student_init_degan",
                train_epochs=self.student_degan_epochs,
                student_lr=self.student_init_lr,
            )

        # Stage 5: alternate training until budget exhausted (budget-capped).
        if ctx.budget_remaining > 0:
            ctx.log_event("dfmshl_stage", {"stage": "alternate", "queries": "labels"})
            self.logger.info("[DFMSHL] Stage 5/5: alternate training (queries for labels)")
            self._official_stage_alternate(ctx, device, pbar=pbar)

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

            # Benchmark scaling unification (DFME-style): the clone consumes [0,1] inputs.
            # Paper/official DFMS-HL code commonly normalizes to [-1,1]; we intentionally
            # do not, to keep scaling consistent across all attacks in this benchmark.
            self.clone = base_clone

        self.clone_optimizer = optim.SGD(
            self.clone.parameters(),
            lr=float(lr),
            momentum=float(self.student_momentum),
            weight_decay=float(self.student_weight_decay),
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
        fake_logits = self.discriminator(clamp_unit(fake_x_01))
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
        fake_logits = self.discriminator(clamp_unit(fake_x_01))
        adv_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))

        # Diversity term uses clone as proxy gradient.
        clone_was_training = self.clone.training
        self.clone.eval()
        prev_requires_grad = [p.requires_grad for p in self.clone.parameters()]
        for p in self.clone.parameters():
            p.requires_grad_(False)
        try:
            clone_logits = self.clone(fake_x_01)
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

        proxy_n = self._official_proxy_subset_size()
        synth_n = int(max(0, int(float(synth_ratio) * int(self.max_synth_samples))))
        if proxy_n <= 0 and synth_n <= 0:
            return

        # Cap to remaining budget (best effort). Official scripts assume enough access.
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
            take = min(oracle_bs, proxy_n - proxy_cursor, int(ctx.budget_remaining))
            if take <= 0:
                break
            x = self.proxy_data[proxy_cursor : proxy_cursor + take].to(device)
            y = self._oracle_hard_labels(ctx, x)
            y_proxy_list.append(y)
            pbar.update(int(take))
            proxy_cursor += int(take)

        # Query labels for synthetic samples.
        y_synth_list: List[torch.Tensor] = []
        synth_cursor = 0
        self.generator.eval()
        while synth_cursor < synth_n and ctx.budget_remaining > 0:
            take = min(oracle_bs, synth_n - synth_cursor, int(ctx.budget_remaining))
            if take <= 0:
                break
            z = z_synth[synth_cursor : synth_cursor + take]
            with torch.no_grad():
                x = tanh_to_unit(self.generator(z))
            y = self._oracle_hard_labels(ctx, x)
            y_synth_list.append(y)
            pbar.update(int(take))
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
            "[DFMSHL] %s labeled: proxy=%d synth=%d (query_used=%d remaining=%d)",
            str(stage_name),
            int(proxy_labeled),
            int(synth_labeled),
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

                self.clone_optimizer.zero_grad(set_to_none=True)
                logits = self.clone(x_b)
                loss = F.cross_entropy(logits, y_b)
                loss.backward()
                self.clone_optimizer.step()

            if self.clone_scheduler is not None:
                self.clone_scheduler.step()

        # Track substitute for final evaluation.
        self.state.attack_state["substitute"] = self.clone

    def _official_stage_alternate(self, ctx: BenchmarkContext, device: str, *, pbar: tqdm) -> None:
        """Stage 5: alternate training (official train_generator_clone.py).

        Paper vs official note:
        - Paper writes minimax adversarial term with log(1-D(G(z))) (Eq.(3)), but official
          uses non-saturating BCE(fake->real) for generator updates.
        - Official code feeds images normalized to [-1,1] into the teacher; this benchmark's
          oracle contract assumes queries are in [0,1] (no extra normalization). We therefore
          keep oracle inputs in [0,1] and apply the official-style normalization only inside
          clone/discriminator training paths.
        """

        if self.generator is None or self.discriminator is None or self.clone is None or self.proxy_data is None:
            return
        if int(self.alternate_epochs) <= 0:
            return

        subset_n = self._official_proxy_subset_size()
        if subset_n <= 0:
            return

        # Reset optimizer/scheduler for alternate stage (keep weights).
        self._reset_clone_for_stage(
            device=device,
            lr=float(self.student_alt_lr),
            cosine_t_max=int(self.alternate_epochs),
            keep_weights=True,
        )

        use_pad_crop = bool(self.config.get("alternate_pad_crop", False))

        self.generator.train()
        self.discriminator.train()

        batch_size = int(self.batch_size)
        for ep in tqdm(
            range(int(self.alternate_epochs)),
            desc="[DFMSHL] Stage 5/5 alternate (epochs)",
            leave=False,
            position=1,
            mininterval=1.0,
        ):
            if ctx.budget_remaining <= 0:
                break

            for real_x in self._iter_proxy_epoch_batches(
                device=device, batch_size=batch_size, subset_size=subset_n
            ):
                if ctx.budget_remaining <= 0:
                    break

                # Budget-capped microbatch
                b = min(int(real_x.size(0)), int(ctx.budget_remaining))
                if b <= 0:
                    break
                real_x = real_x[:b]

                # 1) Generate synthetic batch (keep graph for G update)
                z = torch.randn(b, self.noise_dim, device=device)
                fake_raw = self.generator(z)
                fake_x = tanh_to_unit(fake_raw)

                # 2) Student/clone update: label with oracle and fit on (optionally) augmented inputs
                x_for_student = fake_x.detach()
                if use_pad_crop:
                    x_for_student = self._augment_pad_crop_hflip(x_for_student)

                y = self._oracle_hard_labels(ctx, x_for_student)
                pbar.update(int(b))

                self.clone.train()
                self.clone_optimizer.zero_grad(set_to_none=True)
                logits = self.clone(x_for_student)
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

        self.state.attack_state["substitute"] = self.clone

    def _select_query_batch(self, k: int, state: BenchmarkState) -> QueryBatch:
        self._init_models(state)
        device = state.metadata.get("device", "cpu")
        phase = state.attack_state["phase"]

        if phase == "init_collect":
            k_proxy = k // 2
            k_synth = k - k_proxy

            x_proxy = self._next_proxy_batch(device, batch_size=k_proxy)

            z = torch.randn(k_synth, self.noise_dim, device=device)
            with torch.no_grad():
                # Generator output is tanh in [-1,1]; oracle expects [0,1] under benchmark contract.
                x_synth = tanh_to_unit(self.generator(z))

            x = torch.cat([x_proxy, x_synth], dim=0)
            meta = {"phase": phase, "k_proxy": k_proxy, "k_synth": k_synth}
            return QueryBatch(x=x, meta=meta)

        if phase == "init_retrain_collect":
            z = torch.randn(k, self.noise_dim, device=device)
            with torch.no_grad():
                x = tanh_to_unit(self.generator(z))
            meta = {"phase": phase, "synthetic": True}
            return QueryBatch(x=x, meta=meta)

        z = torch.randn(k, self.noise_dim, device=device)
        with torch.no_grad():
            x_raw = self.generator(z)
        x = tanh_to_unit(x_raw)
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

            # Benchmark scaling unification (DFME-style): the clone consumes [0,1] inputs.
            # Paper/official DFMS-HL code may apply (x-0.5)/0.5 to feed [-1,1] tensors.
            # We intentionally do not, to keep all attacks consistent under the global
            # oracle/eval [0,1] contract.
            self.clone = base_clone

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

    def _next_proxy_batch(self, device: str, batch_size: int | None = None) -> torch.Tensor:
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

        # Benchmark scaling unification (DFME-style): train D directly on [0,1].
        # Paper/official implementations may use inputs normalized to [-1,1]. We deviate
        # intentionally for benchmark-wide consistency across data-free attacks.
        real_logits = self.discriminator(clamp_unit(real_x))
        fake_logits = self.discriminator(clamp_unit(fake_x.detach()))
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
        fake_logits = self.discriminator(clamp_unit(fake_x_gen))
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
            clone_logits = self.clone(fake_x_gen)
        finally:
            for p, rg in zip(self.clone.parameters(), prev_requires_grad, strict=False):
                p.requires_grad_(rg)
            if clone_was_training:
                self.clone.train()
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
            fake_logits = self.discriminator(clamp_unit(fake_x_2))
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
            logits = self.clone(batch_x)
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
        self.clone_optimizer = optim.SGD(
            self.clone.parameters(),
            lr=float(opt_params.get("lr", self.clone_lr)),
            momentum=float(opt_params.get("momentum", 0.9)),
            weight_decay=float(opt_params.get("weight_decay", 5e-4)),
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
