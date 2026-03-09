"""Dual Student Networks for Data-Free Model Stealing (ICLR 2023).

Official reference:
- official_repo_clones/dual_students/dual_students/train.py
"""

from __future__ import annotations

import copy
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.state import BenchmarkState
from mebench.models.gan import DFMEGenerator
from mebench.models.substitute_factory import create_substitute


class _MovingAverageModel(nn.Module):
    """Train/test model pair with official-style EMA updates."""

    def __init__(self, model: nn.Module, momentum: float) -> None:
        super().__init__()
        self.train_model = model
        self.momentum = float(momentum)
        self.test_model: Optional[nn.Module]
        if self.momentum > 0.0:
            self.test_model = copy.deepcopy(model)
            self.test_model.eval()
        else:
            self.test_model = None
        self._first_step = True

    def forward(self, x: torch.Tensor, *, test: bool = False) -> torch.Tensor:
        if test and self.test_model is not None and not self._first_step:
            return self.test_model(x)
        return self.train_model(x)

    def step(self) -> None:
        if self.test_model is None:
            return
        with torch.no_grad():
            momentum = 0.0 if self._first_step else self.momentum
            self._first_step = False
            for train_param, test_param in zip(
                self.train_model.parameters(),
                self.test_model.parameters(),
            ):
                test_param.copy_(momentum * test_param + (1.0 - momentum) * train_param)
            for train_buffer, test_buffer in zip(
                self.train_model.buffers(),
                self.test_model.buffers(),
            ):
                test_buffer.copy_(momentum * test_buffer + (1.0 - momentum) * train_buffer)


class _DualStudentEvalWrapper(nn.Module):
    """Expose official DS test-time student combine semantics."""

    def __init__(self, students: list[_MovingAverageModel], combine_mode: str) -> None:
        super().__init__()
        self.students = nn.ModuleList(students)
        self.combine_mode = str(combine_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = [model(x, test=True) for model in self.students]
        if self.combine_mode == "first":
            return logits[0]
        if self.combine_mode == "mean":
            return torch.stack(logits, dim=0).mean(dim=0)
        raise ValueError(f"Unsupported combine_mode: {self.combine_mode}")


class DualStudents(AttackRunner):
    """Dual Student data-free extraction.

    Key behavior aligned with official DS implementation:
    - Generator update uses student-student disagreement (no victim query).
    - Student update queries victim on generated samples.
    - Query cost per training loop is governed by student updates only.
    """

    def __init__(self, config: dict, state: BenchmarkState) -> None:
        super().__init__(config, state)

        # Fixed-required defaults for DS parity profile (official script family).
        self.batch_size = int(config.get("batch_size", 256))
        self.noise_dim = int(config.get("noise_dim", config.get("nz", 256)))
        self.g_iter = int(config.get("g_iter", 1))
        self.d_iter = int(config.get("d_iter", 5))
        self.num_students = int(config.get("num_students", 2))
        if self.num_students < 2:
            raise ValueError("ds requires num_students >= 2")

        self.generator_lr = float(config.get("generator_lr", config.get("lr_G", 1e-4)))
        self.student_lr = float(config.get("student_lr", config.get("lr_S", 0.3)))
        self.combine_student_outputs = str(config.get("combine_student_outputs", "first")).strip().lower()
        if self.combine_student_outputs not in {"first", "mean"}:
            raise ValueError(
                "ds combine_student_outputs must be one of {'first', 'mean'}"
            )
        self.student_momentum = float(config.get("student_momentum", 0.9))
        self.generator_momentum = float(config.get("generator_momentum", 0.9))

        self.output_mode = str(config.get("output_mode", "soft_prob")).strip().lower()
        self.loss_mode = str(config.get("loss", "")).strip().lower()
        if self.loss_mode == "":
            self.loss_mode = "ce" if self.output_mode == "hard_top1" else "l1"
        if self.output_mode == "hard_top1" and self.loss_mode != "ce":
            raise ValueError("ds hard_top1 mode requires loss='ce'")
        if self.loss_mode not in {"l1", "kl", "ce"}:
            raise ValueError(f"ds loss must be one of l1/kl/ce, got {self.loss_mode!r}")

        self.generator_loss_mode = str(config.get("generator_loss", "l1")).strip().lower()
        if self.generator_loss_mode != "l1":
            raise ValueError("ds currently supports generator_loss='l1' only")

        self.lr_decay_gamma = float(config.get("lr_decay_gamma", config.get("scale", 0.3)))
        self.lr_decay_ratios = [
            float(v) for v in config.get("lr_decay_milestones_ratio", config.get("steps", [0.1, 0.3, 0.5]))
        ]
        self._milestones: list[int] = []
        self._milestone_idx = 0

        eval_interval_raw = int(config.get("eval_interval_queries", 0))
        self.eval_interval_queries = eval_interval_raw if eval_interval_raw > 0 else 0
        self._next_eval_query = self.eval_interval_queries
        self._periodic_eval_done: set[int] = set()

        self.log_interval = int(config.get("log_interval", 30))
        self.strict_iteration_budget = bool(config.get("strict_iteration_budget", True))

        self.generator: Optional[_MovingAverageModel] = None
        self.students: list[nn.Module] = []
        self.student_models: list[_MovingAverageModel] = []
        self.g_opt: Optional[torch.optim.Optimizer] = None
        self.s_opts: list[torch.optim.Optimizer] = []
        self._eval_substitute: Optional[nn.Module] = None

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

        generator = DFMEGenerator(
            noise_dim=self.noise_dim,
            output_channels=channels,
            output_size=image_size,
        ).to(device)
        self.generator = _MovingAverageModel(generator, self.generator_momentum).to(device)
        self.g_opt = torch.optim.Adam(self.generator.train_model.parameters(), lr=self.generator_lr)

        sub_config = state.metadata.get("substitute_config", {})
        arch = str(sub_config.get("arch") or self.config.get("student_arch", "resnet18"))
        width_mult = int(sub_config.get("width_mult", self.config.get("student_width_mult", 1)))
        dropout_prob = float(sub_config.get("dropout_prob", 0.0))

        self.students = []
        self.student_models = []
        self.s_opts = []
        for _ in range(self.num_students):
            model = create_substitute(
                arch=arch,
                num_classes=self.num_classes,
                input_channels=channels,
                width_mult=width_mult,
                dropout_prob=dropout_prob,
            ).to(device)
            wrapped = _MovingAverageModel(model, self.student_momentum).to(device)
            self.student_models.append(wrapped)
            self.students.append(wrapped.train_model)
            self.s_opts.append(
                torch.optim.SGD(
                    wrapped.train_model.parameters(),
                    lr=self.student_lr,
                    momentum=0.9,
                    weight_decay=5e-4,
                )
            )

    @staticmethod
    def _recover_logits_from_probs(probs: torch.Tensor) -> torch.Tensor:
        log_p = torch.log(probs + 1e-10)
        return log_p - log_p.mean(dim=1, keepdim=True)

    @staticmethod
    def _student_scale(x_tanh: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x_tanh, -1.0, 1.0)

    @staticmethod
    def _query_scale(x_tanh: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x_tanh, -1.0, 1.0)

    def _get_substitute_for_eval(self) -> nn.Module:
        if self._eval_substitute is None:
            self._eval_substitute = _DualStudentEvalWrapper(
                self.student_models,
                combine_mode=self.combine_student_outputs,
            )
        return self._eval_substitute

    def _publish_substitute(self) -> None:
        if not self.students:
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
            for opt in self.s_opts:
                for group in opt.param_groups:
                    group["lr"] *= gamma
            self._milestone_idx += 1

    def _format_teacher_target(self, oracle_output: torch.Tensor) -> torch.Tensor:
        if self.loss_mode == "ce":
            if oracle_output.ndim == 1:
                return oracle_output.long()
            return oracle_output.argmax(dim=1).long()

        if self.loss_mode == "kl":
            if oracle_output.ndim == 1:
                return F.one_hot(oracle_output.long(), num_classes=self.num_classes).float()
            target = oracle_output.float().clamp_min(1e-10)
            return target / target.sum(dim=1, keepdim=True).clamp_min(1e-10)

        # l1 path
        if oracle_output.ndim == 1:
            probs = F.one_hot(oracle_output.long(), num_classes=self.num_classes).float()
            return self._recover_logits_from_probs(probs)
        return self._recover_logits_from_probs(oracle_output.float())

    def _student_loss(self, student_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_mode == "ce":
            return F.cross_entropy(student_logits, target.long())
        if self.loss_mode == "kl":
            return F.kl_div(F.log_softmax(student_logits, dim=1), target, reduction="batchmean")
        return F.l1_loss(student_logits, target)

    def _generator_step(self, device: str) -> float:
        if self.generator is None or self.g_opt is None or not self.students:
            return 0.0

        self.generator.train()
        for model in self.students:
            model.eval()

        total_loss = 0.0
        for _ in range(self.g_iter):
            z = torch.randn((self.batch_size, self.noise_dim), device=device)
            fake_tanh = self.generator(z)
            fake = self._student_scale(fake_tanh)

            logits = [model(fake) for model in self.students]
            pair_count = 0
            disagreement = torch.tensor(0.0, device=device)
            for i in range(len(logits)):
                for j in range(i + 1, len(logits)):
                    disagreement = disagreement + F.l1_loss(logits[i], logits[j])
                    pair_count += 1
            if pair_count <= 0:
                continue
            g_loss = -(disagreement / float(pair_count))

            self.g_opt.zero_grad(set_to_none=True)
            g_loss.backward()
            self.g_opt.step()
            total_loss += float(g_loss.item())

        return total_loss / max(1, self.g_iter)

    def _student_step(self, ctx: BenchmarkContext, device: str, iter_idx: int, pbar: Any) -> float:
        if self.generator is None or not self.students:
            return 0.0

        self.generator.eval()
        for model in self.students:
            model.train()

        total_loss = 0.0
        updates = 0

        for d_idx in range(self.d_iter):
            batch = min(int(self.batch_size), int(ctx.budget_remaining))
            if batch <= 0:
                break

            z = torch.randn((batch, self.noise_dim), device=device)
            with torch.no_grad():
                fake_tanh = self.generator(z)

            x_student = self._student_scale(fake_tanh).detach()
            x_query = self._query_scale(fake_tanh).detach()

            oracle_output = ctx.query(
                x_query,
                meta={
                    "attack": "ds",
                    "phase": "student_train",
                    "iter": int(iter_idx),
                    "d_iter": int(d_idx),
                },
            )
            target = self._format_teacher_target(oracle_output.y.to(device))

            was_training = []
            if batch < 2:
                for model in self.students:
                    was_training.append(model.training)
                    model.eval()

            student_loss_sum = 0.0
            for model, opt in zip(self.students, self.s_opts):
                opt.zero_grad(set_to_none=True)
                logits = model(x_student)
                loss = self._student_loss(logits, target)
                loss.backward()
                opt.step()
                student_loss_sum += float(loss.item())

            if batch < 2:
                for idx, model in enumerate(self.students):
                    if was_training[idx]:
                        model.train()

            self._publish_substitute()
            self._maybe_periodic_eval(device)

            total_loss += student_loss_sum / float(max(1, len(self.students)))
            updates += 1
            pbar.update(int(batch))

        return total_loss / max(1, updates)

    def _maybe_step_moving_averages(self, iter_idx: int) -> None:
        if iter_idx % 10 != 9:
            return
        for model in self.student_models:
            model.step()
        if self.generator is not None:
            self.generator.step()

    def run(self, ctx: BenchmarkContext) -> None:
        device = str(self.state.metadata.get("device", "cpu"))
        self.victim = ctx.oracle.model

        max_budget = int(
            self.state.metadata.get("max_budget")
            or ctx.budget_remaining
            or 0
        )
        if max_budget <= 0:
            max_budget = int(ctx.budget_remaining)

        pbar = self._create_progress_bar(ctx.budget_remaining, "[DS] Extracting")
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
            self._maybe_step_moving_averages(iter_idx)

            self._publish_substitute()
            self._maybe_periodic_eval(device)

            if self.log_interval > 0 and (iter_idx % self.log_interval == 0):
                self.logger.info(
                    "[DS] iter=%d queries=%d g_loss=%.6f s_loss=%.6f",
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
        # DS is implemented as a self-contained AttackRunner (Track B).
        return None
