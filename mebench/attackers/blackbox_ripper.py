from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.state import BenchmarkState
from mebench.models.blackbox_ripper import (
    create_blackbox_ripper_generator,
    load_blackbox_ripper_generator_weights,
)
from mebench.models.substitute_factory import create_substitute
from mebench.data.loaders import create_dataloader
from mebench.utils.dataloader import load_pool_to_memory


class BlackboxRipper(AttackRunner):
    """Upstream-faithful Black-Box Ripper (NeurIPS 2020).

    Reference implementations/spec:
    - Official repo: https://github.com/antoniobarbalau/black-box-ripper
    - Paper: `papers/blackbox-ripper.pdf`

    Key behavior we replicate:
    - Fixed pretrained generator (no GAN training during extraction)
    - Per-sample evolutionary search in latent space:
      - Population K=30, elites k=10
      - Init: U(-3.3, 3.3)
      - Offspring: elites + N(0, 0.5) twice (two mutated copies)
    - Student training uses SGD + BCELoss on softmax outputs.

    Benchmark adaptation:
    - All oracle queries go through `ctx.query` so budget is respected (1 image = 1 query).
    - Generator outputs are clamped/rescaled to [0,1] to satisfy the global contract.
    """

    def __init__(self, config: Dict[str, Any], state: BenchmarkState) -> None:
        super().__init__(config, state)

        self.num_classes = int(
            state.metadata.get("num_classes")
            or config.get("num_classes")
            or state.metadata.get("dataset_config", {}).get("num_classes", 10)
        )

        # Generator selection (upstream uses SNGAN or ProGAN checkpoints).
        self.generator_name = str(config.get("generator_name") or config.get("gan_backbone") or "cifar_sngan")
        self.generator_checkpoint = config.get("generator_checkpoint") or config.get("generator_ckpt")
        self.generator_strict_load = bool(config.get("generator_strict_load", True))

        # Evolutionary search hyperparameters (upstream defaults).
        self.population_size = int(config.get("population_size", 30))
        self.elite_size = int(config.get("elite_size", 10))
        self.latent_bound = float(config.get("latent_bound", 3.3))
        self.mutation_scale = float(config.get("mutation_scale", 0.5))
        self.confidence_threshold = float(config.get("confidence_threshold", 0.9))
        # Keep as optional stop criterion.
        # Upstream `torch_optimizer.optimize()` stops based on confidence; default to 0.0 (disabled).
        self.fitness_threshold = float(config.get("fitness_threshold", 0.0))
        # Official repo has variants; `optimize()` uses up to 300 iterations.
        self.max_evolve_iters = int(config.get("max_evolve_iters", 300))

        # Student training hyperparameters (upstream train_or_restore_predictor).
        self.train_batch_size = int(config.get("train_batch_size", config.get("batch_size", 64)))
        self.substitute_epochs = int(config.get("substitute_epochs", 200))
        self.batches_per_epoch = int(config.get("batches_per_epoch", 1000))
        self.substitute_lr = float(config.get("substitute_lr", 0.01))
        self.momentum = float(config.get("momentum", 0.9))
        self.weight_decay = float(config.get("weight_decay", 5e-4))
        self.grad_clip = float(config.get("grad_clip", 0.1))
        self.lr_decay_start = int(config.get("lr_decay_start", 17))
        self.lr_decay_every = int(config.get("lr_decay_every", 1))
        self.lr_decay_rate = float(config.get("lr_decay_rate", 0.9))
        self.log_interval = int(config.get("log_interval", 25))

        # Strict upstream artifact: in the official code, teacher is queried again to label
        # the final optimized samples. Keep this optional since it doubles query cost.
        self.strict_label_query = bool(config.get("strict_label_query", False))

        self.generator: Optional[nn.Module] = None
        self.substitute: Optional[nn.Module] = None
        self.substitute_optimizer: Optional[torch.optim.Optimizer] = None
        self.substitute_loss: Optional[nn.Module] = None

        self._evaluated_checkpoints: set[int] = set(
            state.attack_state.get("bbr_evaluated_checkpoints", [])
        )

    def run(self, ctx: BenchmarkContext) -> None:
        self.victim = ctx.oracle.model
        device = str(self.state.metadata.get("device", "cpu"))

        if self.config.get("output_mode", "soft_prob") != "soft_prob":
            raise ValueError("BlackboxRipper requires soft_prob output mode")

        self._init_generator(device)
        self._init_substitute(device)

        total_budget = int(self.state.budget_remaining)
        pbar = tqdm(total=total_budget, desc="[BlackboxRipper] Queries", leave=True)

        stop_all = False
        for epoch in range(self.substitute_epochs):
            if stop_all:
                break

            self._set_epoch_lr(epoch)
            if self.substitute is None:
                break
            self.substitute.train(True)

            for iter_n in range(self.batches_per_epoch):
                if int(ctx.budget_remaining) < self.population_size:
                    stop_all = True
                    break

                batch = self._generate_optimized_batch(ctx, device)
                if batch is None:
                    stop_all = True
                    break
                x_batch, y_batch = batch
                self._train_on_batch(x_batch, y_batch, device)

                # Expose current substitute for engine FINAL EVAL.
                self.state.attack_state["substitute"] = self.substitute

                # Progress bar: budget spent is tracked in ctx/state.
                pbar.n = int(self.state.query_count)
                pbar.refresh()

                if self.log_interval > 0 and (iter_n % self.log_interval == 0):
                    with torch.no_grad():
                        logits = self.substitute(x_batch)
                        probs = torch.softmax(logits, dim=-1)
                        acc = probs.argmax(dim=1).eq(y_batch.argmax(dim=1)).float().mean().item()
                    self.logger.info(
                        "Epoch %d/%d, iter %d/%d, acc=%.4f",
                        epoch,
                        self.substitute_epochs,
                        iter_n,
                        self.batches_per_epoch,
                        acc,
                    )

                self._maybe_evaluate_on_checkpoints(ctx, device)

        pbar.close()

        # Persist evaluation state.
        self.state.attack_state["bbr_evaluated_checkpoints"] = sorted(self._evaluated_checkpoints)

    def _init_generator(self, device: str) -> None:
        if self.generator is not None:
            return

        gen_name = str(self.generator_name).lower()
        # Backwards-compat: older config used gan_backbone="sngan"/"progan".
        if gen_name in {"sngan", "sndcgan", "sn-dcgan", "sn_dcgan"}:
            gen_name = "cifar_sngan"
        if gen_name in {"progan", "pro-gan"}:
            gen_name = "cifar_progan"

        # Backwards-compat: official repo generator names.
        if gen_name in {"cifar_10_gan", "cifar_100_90_classes_gan", "cifar_100_40_classes_gan"}:
            gen_name = "cifar_sngan"
        if gen_name in {"cifar_100_6_classes_gan", "cifar_100_10_classes_gan"}:
            gen_name = "cifar_progan"

        self.generator = create_blackbox_ripper_generator(gen_name, device)
        if self.generator_checkpoint is None:
            raise ValueError(
                "BlackboxRipper requires a pretrained generator checkpoint. "
                "Set `attack.generator_checkpoint` to the official weights."
            )
        load_blackbox_ripper_generator_weights(
            self.generator,
            str(self.generator_checkpoint),
            device,
            strict=self.generator_strict_load,
        )
        self.generator.eval()

    def _init_substitute(self, device: str) -> None:
        if self.substitute is not None:
            return

        sub_config = self.state.metadata.get("substitute_config", {})
        width_mult = int(sub_config.get("width_mult", 1))
        dropout_prob = float(sub_config.get("dropout_prob", 0.0))

        self.substitute = create_substitute(
            arch=str(sub_config.get("arch", "resnet18")),
            num_classes=self.num_classes,
            input_channels=int(self.state.metadata.get("input_shape", (3, 32, 32))[0]),
            width_mult=width_mult,
            dropout_prob=dropout_prob,
        ).to(device)

        # [UNIFIED] Use runner's build_optimizer to respect config (LR, optimizer type)
        opt_config = sub_config.get("optimizer", {})
        # Override with specific BlackboxRipper defaults if not present in config
        if "lr" not in opt_config:
            opt_config["lr"] = self.substitute_lr
        if "momentum" not in opt_config:
            opt_config["momentum"] = self.momentum
        if "weight_decay" not in opt_config:
            opt_config["weight_decay"] = self.weight_decay
            
        self.substitute_optimizer = self._build_optimizer(self.substitute.parameters(), opt_config)
        self.substitute_loss = nn.BCELoss()

        # Make visible for FINAL EVAL even if budget ends early.
        self.state.attack_state["substitute"] = self.substitute

    def _set_epoch_lr(self, epoch: int) -> None:
        if self.substitute_optimizer is None:
            return

        lr = float(self.substitute_lr)
        if epoch > self.lr_decay_start and self.lr_decay_start >= 0:
            frac = (epoch - self.lr_decay_start) // max(1, int(self.lr_decay_every))
            decay_factor = float(self.lr_decay_rate) ** int(frac)
            lr = float(self.substitute_lr) * decay_factor

        for group in self.substitute_optimizer.param_groups:
            group["lr"] = lr

    def _generate_optimized_batch(
        self, ctx: BenchmarkContext, device: str
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self.generator is None:
            raise RuntimeError("Generator not initialized")

        input_shape = tuple(self.state.metadata.get("input_shape", (3, 32, 32)))
        if len(input_shape) != 3:
            raise ValueError(f"Invalid input_shape in state.metadata: {input_shape}")

        x_list = []
        y_list = []
        for sample_id in range(self.train_batch_size):
            sample = self._optimize_single_sample(ctx, device, input_shape, sample_id)
            if sample is None:
                break
            x_i, y_i = sample
            x_list.append(x_i)
            y_list.append(y_i)

        if not x_list:
            return None

        x_batch = torch.cat(x_list, dim=0).to(device)
        y_batch = torch.cat(y_list, dim=0).to(device)
        return x_batch, y_batch

    def _optimize_single_sample(
        self,
        ctx: BenchmarkContext,
        device: str,
        input_shape: Tuple[int, int, int],
        sample_id: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self.generator is None:
            raise RuntimeError("Generator not initialized")

        # Need at least one population query.
        if int(ctx.budget_remaining) < self.population_size:
            return None

        latent_dim = int(getattr(self.generator, "latent_dim", 128))
        population = torch.empty(self.population_size, latent_dim, device=device)
        population.uniform_(-self.latent_bound, self.latent_bound)
        target_cls = int(torch.randint(0, self.num_classes, (1,), device=device).item())
        target_onehot = F.one_hot(torch.tensor(target_cls, device=device), num_classes=self.num_classes).float()

        best_x: Optional[torch.Tensor] = None
        best_probs: Optional[torch.Tensor] = None
        best_obj: float = float("inf")
        best_conf: float = 0.0

        for it in range(self.max_evolve_iters):
            if int(ctx.budget_remaining) < self.population_size:
                break

            with torch.no_grad():
                x_raw = self.generator(population)
                x = _clamp_to_unit_range(x_raw)
                x = _match_input_shape(x, input_shape)

            oracle_out = ctx.query(
                x,
                meta={
                    "attack": "blackbox_ripper",
                    "phase": "evolution",
                    "sample_id": int(sample_id),
                    "iter": int(it),
                    "target_cls": int(target_cls),
                },
            )
            if oracle_out.kind != "soft_prob":
                raise ValueError("BlackboxRipper requires soft_prob oracle output")
            probs = oracle_out.y.to(device)

            obj = self._objective_mse_sum(probs, target_onehot)
            # Select top-10 elites by objective (lower is better).
            elite_k = min(self.elite_size, int(obj.numel()))
            elite_indices = torch.argsort(obj)[:elite_k]

            best_idx = int(elite_indices[0].item())
            best_obj = float(obj[best_idx].item())
            best_conf = float(probs[best_idx, target_cls].item())
            best_x = x[best_idx : best_idx + 1].detach()
            best_probs = probs[best_idx : best_idx + 1].detach()

            if best_conf >= self.confidence_threshold or best_obj <= self.fitness_threshold:
                break

            elites = population[elite_indices]
            population = self._make_next_population_from_elites(elites)

        if best_x is None or best_probs is None:
            return None

        # Strict upstream artifact: re-query teacher for final labels.
        if self.strict_label_query and int(ctx.budget_remaining) >= 1:
            out2 = ctx.query(
                best_x,
                meta={
                    "attack": "blackbox_ripper",
                    "phase": "label",
                    "sample_id": int(sample_id),
                    "target_cls": int(target_cls),
                    "best_conf": float(best_conf),
                    "best_obj": float(best_obj),
                },
            )
            if out2.kind != "soft_prob":
                raise ValueError("BlackboxRipper requires soft_prob oracle output")
            best_probs = out2.y.to(device)

        return best_x, best_probs

    @staticmethod
    def _objective_mse_sum(probs: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
        """Objective from paper Eq.(2): sum_j (p_j - y_j)^2.

        Upstream uses an equivalent MSE-on-softmax objective during evolution.
        """

        diff = probs - target_onehot.view(1, -1)
        return (diff * diff).sum(dim=1)

    def _make_next_population_from_elites(self, elites: torch.Tensor) -> torch.Tensor:
        """Upstream mutation rule.

        Official repo pattern (see `temp_ripper/torch_optimizer.py`):
        - keep elites
        - add two mutated copies of elites with Gaussian noise (scale=0.5)
        """

        noise1 = torch.randn_like(elites) * self.mutation_scale
        noise2 = torch.randn_like(elites) * self.mutation_scale
        return torch.cat([elites, elites + noise1, elites + noise2], dim=0)

    def _train_on_batch(self, x_batch: torch.Tensor, y_batch: torch.Tensor, device: str) -> None:
        if self.substitute is None or self.substitute_optimizer is None or self.substitute_loss is None:
            return

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        self.substitute_optimizer.zero_grad(set_to_none=True)
        logits = self.substitute(x_batch)
        probs = torch.softmax(logits, dim=-1)
        loss = self.substitute_loss(probs, y_batch)
        loss.backward()

        # Upstream gradient clipping: clamp individual gradients.
        if self.grad_clip > 0:
            for param in self.substitute.parameters():
                if param.grad is None:
                    continue
                param.grad.data.clamp_(-self.grad_clip, self.grad_clip)

        self.substitute_optimizer.step()

    def _maybe_evaluate_on_checkpoints(self, ctx: BenchmarkContext, device: str) -> None:
        if self.substitute is None:
            return

        reached = self.state.attack_state.get("checkpoint_reached", [])
        if not reached:
            return

        for checkpoint in reached:
            cp = int(checkpoint)
            if cp in self._evaluated_checkpoints:
                continue
            self._evaluated_checkpoints.add(cp)
            self._evaluate_current_substitute(self.substitute, device)
