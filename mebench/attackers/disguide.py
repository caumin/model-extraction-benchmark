"""DisGUIDE: Disagreement-Guided Data-Free Model Extraction (AAAI 2023).

Official reference:
- `official_repo_clones/disguide/disguide/train.py`

This implementation keeps the core loop structure from the official code:
- Generator update with disagreement + class-diversity losses (no victim query)
- Clone update with victim-labeled synthetic queries
- Optional replay updates without additional victim queries

Benchmark adaptations:
- All victim queries go through `ctx.query` for strict image-based budget accounting.
- Data-free query tensors remain tanh-scale at oracle boundary.
- Internal clone input scale is configurable (`tanh` or `unit`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.state import BenchmarkState
from mebench.models.gan import DFMEGenerator
from mebench.models.substitute_factory import create_substitute
from mebench.utils.scaling import normalize_input_scale


class _ScaledSubstituteWrapper(nn.Module):
    """Apply input scaling before substitute forward for evaluation."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scaled = normalize_input_scale(x, "tanh")
        return self.model(x_scaled)


class _DisguideEnsemble(nn.Module):
    """Simple clone ensemble with soft-vote output.

    - `forward(x, idx>=0)` returns logits of a specific clone.
    - `forward(x)` returns `log(mean_softmax(clone_logits))`.
    """

    def __init__(self, clones: list[nn.Module]) -> None:
        super().__init__()
        self.clones = nn.ModuleList(clones)

    def size(self) -> int:
        return len(self.clones)

    def get_model_by_idx(self, idx: int) -> nn.Module:
        return self.clones[int(idx)]

    def forward(self, x: torch.Tensor, idx: int = -1) -> torch.Tensor:
        if idx >= 0:
            return self.clones[int(idx)](x)

        logits = [model(x) for model in self.clones]
        probs = torch.stack([F.softmax(logit, dim=1) for logit in logits], dim=1)
        mean_probs = probs.mean(dim=1).clamp_min(1e-10)
        return torch.log(mean_probs)


@dataclass
class _ReplayBatch:
    x: torch.Tensor
    y: torch.Tensor


class _ClassicalReplayMemory:
    """Circular FIFO replay memory (official 'Classic' replay)."""

    def __init__(self, max_length: int) -> None:
        self.max_length = int(max(1, max_length))
        self.x_store: Optional[torch.Tensor] = None
        self.y_store: Optional[torch.Tensor] = None
        self.size = 0
        self.head = 0

    def __len__(self) -> int:
        return int(self.size)

    def update(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> None:
        x_batch = x_batch.detach().cpu()
        y_batch = y_batch.detach().cpu()
        b = int(x_batch.size(0))
        if b <= 0:
            return

        if self.x_store is None:
            self.x_store = torch.zeros(
                (self.max_length, *x_batch.shape[1:]),
                dtype=x_batch.dtype,
            )
            self.y_store = torch.zeros(
                (self.max_length, *y_batch.shape[1:]),
                dtype=y_batch.dtype,
            )

        assert self.x_store is not None and self.y_store is not None

        tail = self.head + b
        if tail <= self.max_length:
            self.x_store[self.head:tail] = x_batch
            self.y_store[self.head:tail] = y_batch
            self.size = min(self.max_length, max(self.size, tail))
        else:
            first = self.max_length - self.head
            self.x_store[self.head:] = x_batch[:first]
            self.y_store[self.head:] = y_batch[:first]
            rem = b - first
            self.x_store[:rem] = x_batch[first:]
            self.y_store[:rem] = y_batch[first:]
            self.size = self.max_length

        self.head = tail % self.max_length

    def sample(self, batch_size: int, device: str) -> _ReplayBatch:
        if self.x_store is None or self.y_store is None or self.size <= 0:
            raise ValueError("Replay memory is empty")

        bs = int(max(1, min(batch_size, self.size)))
        idx = torch.randperm(self.size)[:bs]
        return _ReplayBatch(
            x=self.x_store[idx].to(device),
            y=self.y_store[idx].to(device),
        )


class DisGUIDE(AttackRunner):
    """DisGUIDE implementation for benchmark Track B."""

    def __init__(self, config: dict, state: BenchmarkState) -> None:
        super().__init__(config, state)

        # Fixed-required family for default parity runs: batch=256, g_iter/d_iter
        # schedule and lr_S/lr_G anchors from official DisGUIDE setup.
        self.batch_size = int(config.get("batch_size", 256))
        self.g_iter = int(config.get("g_iter", config.get("n_g_steps", 1)))
        self.d_iter = int(config.get("d_iter", config.get("n_s_steps", 1)))
        self.rep_iter = int(config.get("rep_iter", 3))
        self.replay_mode = str(config.get("replay", "Classic"))
        self.replay_size = int(config.get("replay_size", 1_000_000))

        self.noise_dim = int(config.get("noise_dim", config.get("nz", 256)))
        self.generator_lr = float(config.get("generator_lr", config.get("lr_G", 1e-4)))
        self.ensemble_size = int(config.get("ensemble_size", 2))
        if self.ensemble_size < 2:
            raise ValueError("DisGUIDE requires ensemble_size >= 2")

        self.grayscale_freq = int(config.get("grayscale_freq", config.get("grayscale", 8)))
        if self.grayscale_freq < 0:
            raise ValueError(f"grayscale_freq must be >= 0, got {self.grayscale_freq}")

        self.lambda_div = float(config.get("lambda_div", config.get("lambda_diversity", -1.0)))

        self.loss_mode = str(config.get("loss", "")).strip().lower()
        self.output_mode = str(config.get("output_mode", "soft_prob")).strip().lower()
        if self.loss_mode == "":
            self.loss_mode = "hl" if self.output_mode == "hard_top1" else "l1"
        if self.loss_mode not in {"l1", "kl", "hl"}:
            raise ValueError(f"DisGUIDE loss must be one of l1/kl/hl, got {self.loss_mode!r}")
        if self.output_mode == "hard_top1" and self.loss_mode != "hl":
            raise ValueError("DisGUIDE hard_top1 mode requires loss='hl'")

        self.lr_decay_gamma = float(config.get("lr_decay_gamma", config.get("scale", 0.3)))
        self.lr_decay_ratios = [
            float(v) for v in config.get("lr_decay_milestones_ratio", config.get("step", [0.4, 0.8]))
        ]

        eval_interval_raw = int(config.get("eval_interval_queries", 0))
        self.eval_interval_queries = eval_interval_raw if eval_interval_raw > 0 else 0
        self._next_eval_query = self.eval_interval_queries
        self._periodic_eval_done: set[int] = set()

        self.log_interval = int(config.get("log_interval", 30))
        self.strict_iteration_budget = bool(config.get("strict_iteration_budget", False))

        self.generator: Optional[nn.Module] = None
        self.student_ensemble: Optional[_DisguideEnsemble] = None
        self._eval_substitute: Optional[nn.Module] = None
        self.g_opt: Optional[torch.optim.Optimizer] = None
        self.s_opt: Optional[torch.optim.Optimizer] = None
        self.replay_memory: Optional[_ClassicalReplayMemory] = None

        self._milestones: list[int] = []
        self._milestone_idx = 0

        self._initialize_models(state)

    @property
    def num_classes(self) -> int:
        return int(
            self.state.metadata.get("num_classes")
            or self.state.metadata.get("victim_config", {}).get("num_classes")
            or self.config.get("num_classes")
            or 10
        )

    def _initialize_models(self, state: BenchmarkState) -> None:
        device = str(state.metadata.get("device", "cpu"))
        input_shape = state.metadata.get("input_shape", (3, 32, 32))
        channels = int(input_shape[0])
        image_size = int(input_shape[1])

        if self.lambda_div < 0:
            # Paper defaults: CIFAR10=0.2, CIFAR100=0.04
            self.lambda_div = 0.2 if int(self.num_classes) <= 10 else 0.04

        self.generator = DFMEGenerator(
            noise_dim=self.noise_dim,
            output_channels=channels,
            output_size=image_size,
        ).to(device)
        self.g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.generator_lr)

        sub_config = state.metadata.get("substitute_config", {})
        arch = str(sub_config.get("arch") or self.config.get("student_arch", "resnet18"))
        width_mult = int(sub_config.get("width_mult", self.config.get("student_width_mult", 1)))
        dropout_prob = float(sub_config.get("dropout_prob", 0.0))

        clones: list[nn.Module] = []
        for _ in range(self.ensemble_size):
            clone = create_substitute(
                arch=arch,
                num_classes=self.num_classes,
                input_channels=channels,
                width_mult=width_mult,
                dropout_prob=dropout_prob,
            ).to(device)
            clones.append(clone)
        self.student_ensemble = _DisguideEnsemble(clones)

        opt_config = dict(sub_config.get("optimizer", {}))
        opt_config.setdefault("name", "sgd")
        opt_config.setdefault("lr", float(self.config.get("student_lr", 0.01)))
        opt_config.setdefault("momentum", 0.9)
        opt_config.setdefault("weight_decay", 5e-4)
        self.s_opt = self._build_optimizer(self.student_ensemble.parameters(), opt_config)

        if self.replay_mode == "Classic":
            self.replay_memory = _ClassicalReplayMemory(self.replay_size)
        elif self.replay_mode == "Off":
            self.replay_memory = None
        else:
            raise ValueError(f"Unknown replay mode: {self.replay_mode}")

    def _student_scale(self, x_tanh: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x_tanh, -1.0, 1.0)

    @staticmethod
    def _recover_logits_from_probs(probs: torch.Tensor) -> torch.Tensor:
        log_p = torch.log(probs + 1e-10)
        return log_p - log_p.mean(dim=1, keepdim=True)

    @staticmethod
    def _apply_grayscale(x: torch.Tensor, freq: int) -> torch.Tensor:
        if int(freq) <= 0:
            return x
        if x.ndim != 4 or int(x.size(1)) < 3:
            return x

        out = x.clone()
        with torch.no_grad():
            rng_channel = int(torch.randint(0, 3, (1,), device=x.device).item())
            idx = slice(None, None, int(freq))
            out[idx, (rng_channel + 1) % 3] = out[idx, rng_channel]
            out[idx, (rng_channel + 2) % 3] = out[idx, rng_channel]
        return out

    def _get_substitute_for_eval(self) -> nn.Module:
        if self.student_ensemble is None:
            raise RuntimeError("DisGUIDE student ensemble is not initialized")
        if self._eval_substitute is None:
            self._eval_substitute = _ScaledSubstituteWrapper(self.student_ensemble)
        return self._eval_substitute

    def _publish_substitute(self) -> None:
        if self.student_ensemble is None:
            return
        self.state.attack_state["substitute"] = self._get_substitute_for_eval()

    def _maybe_periodic_eval(self, device: str) -> None:
        if self.eval_interval_queries <= 0:
            return
        if self.victim is None:
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

    def _update_lr_schedule(self, max_budget: int) -> None:
        if not self._milestones:
            points = [int(max_budget * r) for r in self.lr_decay_ratios]
            self._milestones = sorted({p for p in points if p > 0})

        while self._milestone_idx < len(self._milestones):
            if int(self.state.query_count) < int(self._milestones[self._milestone_idx]):
                break
            gamma = float(self.lr_decay_gamma)
            if self.g_opt is not None:
                for group in self.g_opt.param_groups:
                    group["lr"] *= gamma
            if self.s_opt is not None:
                for group in self.s_opt.param_groups:
                    group["lr"] *= gamma
            self._milestone_idx += 1

    def _generator_step(self, device: str) -> float:
        if self.generator is None or self.student_ensemble is None or self.g_opt is None:
            return 0.0

        self.generator.train()
        self.student_ensemble.eval()

        total = 0.0
        for _ in range(self.g_iter):
            z = torch.randn((self.batch_size, self.noise_dim), device=device)
            fake_tanh = self.generator(z)
            fake_tanh = self._apply_grayscale(fake_tanh, self.grayscale_freq)
            x_student = self._student_scale(fake_tanh)

            preds = []
            for idx in range(self.student_ensemble.size()):
                logits = self.student_ensemble(x_student, idx=idx)
                preds.append(logits)
            pred_probs = F.softmax(torch.stack(preds, dim=1), dim=2)

            g_loss = -torch.mean(torch.std(pred_probs, dim=1))
            if self.lambda_div != 0.0:
                soft_vote_mean = torch.mean(torch.mean(pred_probs + 1e-6, dim=1), dim=0)
                g_loss = g_loss + self.lambda_div * torch.sum(soft_vote_mean * torch.log(soft_vote_mean))

            self.g_opt.zero_grad(set_to_none=True)
            g_loss.backward()
            self.g_opt.step()
            total += float(g_loss.item())

        return total / max(1, self.g_iter)

    def _format_teacher_target(self, oracle_output: torch.Tensor) -> torch.Tensor:
        if self.loss_mode == "hl":
            if oracle_output.ndim == 1:
                return oracle_output.long()
            return oracle_output.argmax(dim=1).long()
        if self.loss_mode == "kl":
            if oracle_output.ndim == 1:
                return F.one_hot(oracle_output.long(), num_classes=self.num_classes).float()
            return oracle_output.float()
        # l1 path
        if oracle_output.ndim == 1:
            probs = F.one_hot(oracle_output.long(), num_classes=self.num_classes).float()
            return self._recover_logits_from_probs(probs)
        return self._recover_logits_from_probs(oracle_output.float())

    def _student_loss(self, student_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_mode == "hl":
            return F.cross_entropy(student_logits, target.long())
        if self.loss_mode == "kl":
            target_probs = target.float()
            target_probs = target_probs.clamp_min(1e-10)
            target_probs = target_probs / target_probs.sum(dim=1, keepdim=True).clamp_min(1e-10)
            return F.kl_div(F.log_softmax(student_logits, dim=1), target_probs, reduction="batchmean")
        return F.l1_loss(student_logits, target)

    def _train_students_on_batch(self, x_student: torch.Tensor, target: torch.Tensor) -> float:
        if self.student_ensemble is None or self.s_opt is None:
            return 0.0

        self.student_ensemble.train()
        self.s_opt.zero_grad(set_to_none=True)

        # BatchNorm safety for tiny tail batches.
        batch_size = int(x_student.size(0))
        was_training = []
        if batch_size < 2:
            for idx in range(self.student_ensemble.size()):
                m = self.student_ensemble.get_model_by_idx(idx)
                was_training.append(m.training)
                m.eval()

        total_loss = 0.0
        for idx in range(self.student_ensemble.size()):
            logits = self.student_ensemble(x_student, idx=idx)
            loss = self._student_loss(logits, target)
            loss.backward()
            total_loss += float(loss.item())

        self.s_opt.step()

        if batch_size < 2:
            for idx in range(self.student_ensemble.size()):
                if was_training[idx]:
                    self.student_ensemble.get_model_by_idx(idx).train()

        return total_loss / max(1, self.student_ensemble.size())

    def _student_step(self, ctx: BenchmarkContext, device: str, iter_idx: int, pbar: Any) -> float:
        if self.generator is None or self.student_ensemble is None:
            return 0.0

        self.generator.eval()
        self.student_ensemble.train()

        total_loss = 0.0
        updates = 0

        for d_idx in range(self.d_iter):
            batch = min(int(self.batch_size), int(ctx.budget_remaining))
            if batch <= 0:
                break

            z = torch.randn((batch, self.noise_dim), device=device)
            with torch.no_grad():
                fake_tanh = self.generator(z)
                fake_tanh = self._apply_grayscale(fake_tanh, self.grayscale_freq)
            x_student = self._student_scale(fake_tanh).detach()
            x_query = torch.clamp(fake_tanh, -1.0, 1.0).detach()

            oracle_output = ctx.query(
                x_query,
                meta={
                    "attack": "disguide",
                    "phase": "clone_train",
                    "iter": int(iter_idx),
                    "d_iter": int(d_idx),
                },
            )
            target = self._format_teacher_target(oracle_output.y.to(device))

            self._publish_substitute()
            self._maybe_periodic_eval(device)

            if self.replay_memory is not None:
                self.replay_memory.update(x_student, target)

            total_loss += self._train_students_on_batch(x_student, target)
            updates += 1
            pbar.update(int(batch))

        if self.replay_memory is not None and self.rep_iter > 0 and len(self.replay_memory) > 0:
            for _ in range(self.rep_iter):
                replay_batch = self.replay_memory.sample(self.batch_size, device)
                total_loss += self._train_students_on_batch(replay_batch.x, replay_batch.y)
                updates += 1

        return total_loss / max(1, updates)

    def run(self, ctx: BenchmarkContext) -> None:
        device = str(self.state.metadata.get("device", "cpu"))
        self.victim = ctx.oracle.model

        max_budget = int(
            self.state.metadata.get("max_budget")
            or self.config.get("max_budget")
            or ctx.budget_remaining
            or 0
        )
        if max_budget <= 0:
            max_budget = int(ctx.budget_remaining)

        pbar = self._create_progress_bar(ctx.budget_remaining, "[DisGUIDE] Extracting")
        self._publish_substitute()

        iter_idx = 0
        while ctx.budget_remaining > 0:
            if self.strict_iteration_budget:
                required = int(self.batch_size) * max(1, int(self.d_iter))
                if int(ctx.budget_remaining) < required:
                    break

            self._update_lr_schedule(max_budget)

            g_loss = self._generator_step(device)
            s_loss = self._student_step(ctx, device, iter_idx, pbar)

            self._publish_substitute()
            self._maybe_periodic_eval(device)

            if self.log_interval > 0 and (iter_idx % self.log_interval == 0):
                self.logger.info(
                    "[DisGUIDE] iter=%d queries=%d g_loss=%.6f s_loss=%.6f",
                    int(iter_idx),
                    int(self.state.query_count),
                    float(g_loss),
                    float(s_loss),
                )

            iter_idx += 1
            if self.d_iter <= 0:
                break

        self._publish_substitute()
        pbar.close()

    def observe(self, query_batch, oracle_output, state) -> None:
        # DisGUIDE is implemented as a self-contained AttackRunner (Track B)
        # with all logic in run(). Kept for interface compatibility.
        return None
