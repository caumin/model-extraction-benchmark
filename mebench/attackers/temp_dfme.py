"""Internal DFME ablation: baseline vs unit-query in one run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.state import BenchmarkState
from mebench.models.gan import DFMEGenerator
from mebench.models.substitute_factory import create_substitute
from mebench.utils.scaling import tanh_to_unit


@dataclass
class _BranchState:
    name: str
    query_scale_mode: str
    query_scale_label: str
    student: nn.Module
    generator: nn.Module
    s_opt: optim.Optimizer
    g_opt: optim.Optimizer
    target_budget: int
    consumed_budget: int = 0
    milestone_idx: int = 0
    next_eval_query: int = 0


class TempDFME(AttackRunner):
    """Run two DFME variants side-by-side for internal ablation.

    Variants:
    - baseline: same DFME query scale as current implementation (tanh-clamped)
    - unit_query: convert tanh generator outputs to [0,1] before query
    """

    def __init__(self, config: dict, state: BenchmarkState):
        super().__init__(config, state)
        self.batch_size = int(config.get("batch_size", 256))
        self.n_g = int(config.get("n_g_steps", 1))
        self.n_s = int(config.get("n_s_steps", 5))
        self.epsilon = float(config.get("grad_approx_epsilon", 1e-3))
        self.m = int(config.get("grad_approx_m", 1))
        self.noise_dim = int(config.get("noise_dim", 256))
        self.generator_lr = float(config.get("generator_lr", config.get("lr_G", 1e-4)))
        self.student_lr = float(config.get("student_lr", 0.1))
        self.lr_gamma = float(config.get("lr_gamma", 0.3))
        eval_interval_raw = int(config.get("eval_interval_queries", 0))
        self.eval_interval_queries = eval_interval_raw if eval_interval_raw > 0 else 0
        self._next_paired_eval_query = int(self.eval_interval_queries)
        self._checkpoint_eval_done: set[int] = set()

        self.baseline_branch: Optional[_BranchState] = None
        self.unit_branch: Optional[_BranchState] = None

    def _build_branch(
        self,
        state: BenchmarkState,
        name: str,
        target_budget: int,
        query_scale_mode: str,
        query_scale_label: str,
    ) -> _BranchState:
        device = state.metadata.get("device", "cpu")
        input_shape = state.metadata.get("input_shape", (3, 32, 32))

        generator = DFMEGenerator(
            noise_dim=self.noise_dim,
            output_channels=input_shape[0],
            output_size=input_shape[1],
        ).to(device)
        g_opt = optim.Adam(generator.parameters(), lr=self.generator_lr)

        sub_config = state.metadata.get("substitute_config", {})
        arch = sub_config.get("arch") or self.config.get("student_arch", "resnet18")
        width_mult = int(sub_config.get("width_mult", 1))
        dropout_prob = float(sub_config.get("dropout_prob", 0.0))
        student = create_substitute(
            arch=arch,
            num_classes=state.metadata.get("num_classes", 10),
            input_channels=int(input_shape[0]),
            width_mult=width_mult,
            dropout_prob=dropout_prob,
        ).to(device)

        opt_config = dict(sub_config.get("optimizer", {}))
        opt_config.setdefault("name", "sgd")
        opt_config.setdefault("lr", float(self.student_lr))
        opt_config.setdefault("momentum", 0.9)
        opt_config.setdefault("weight_decay", 5e-4)
        s_opt = self._build_optimizer(student.parameters(), opt_config)

        return _BranchState(
            name=name,
            query_scale_mode=query_scale_mode,
            query_scale_label=query_scale_label,
            student=student,
            generator=generator,
            s_opt=s_opt,
            g_opt=g_opt,
            target_budget=max(0, int(target_budget)),
            next_eval_query=int(self.eval_interval_queries),
        )

    @staticmethod
    def _recover_logits(probs: torch.Tensor) -> torch.Tensor:
        log_p = torch.log(probs + 1e-10)
        return log_p - log_p.mean(dim=1, keepdim=True)

    def _query_scale(self, branch: _BranchState, x_tanh: torch.Tensor) -> torch.Tensor:
        if branch.query_scale_mode == "unit_01":
            return tanh_to_unit(x_tanh)
        return torch.clamp(x_tanh, -1.0, 1.0)

    def _decay_lr_if_needed(self, branch: _BranchState) -> None:
        milestones = [
            int(branch.target_budget * 0.1),
            int(branch.target_budget * 0.3),
            int(branch.target_budget * 0.5),
        ]
        milestones = [m for m in milestones if m > 0]
        while branch.milestone_idx < len(milestones):
            if int(branch.consumed_budget) < int(milestones[branch.milestone_idx]):
                break
            for group in branch.g_opt.param_groups:
                group["lr"] *= self.lr_gamma
            for group in branch.s_opt.param_groups:
                group["lr"] *= self.lr_gamma
            branch.milestone_idx += 1

    def _query_with_budget(self, ctx: BenchmarkContext, x_query: torch.Tensor) -> torch.Tensor:
        return self._recover_logits(ctx.query(x_query).y.to(x_query.device))

    def _run_generator_step(self, branch: _BranchState, ctx: BenchmarkContext, device: str, pbar) -> None:
        for _ in range(self.n_g):
            total_queries = 1 + self.m
            remaining_branch = int(branch.target_budget - branch.consumed_budget)
            batch = min(
                int(self.batch_size),
                int(ctx.budget_remaining) // int(total_queries),
                int(remaining_branch) // int(total_queries),
            )
            if batch <= 0:
                return

            branch.generator.train()
            branch.student.eval()
            z = torch.randn(batch, self.noise_dim, device=device)
            pre_tanh, x_tanh = branch.generator(z, return_pre_tanh=True)
            x_query = self._query_scale(branch, x_tanh)
            x_student = x_query

            with torch.no_grad():
                s_out = branch.student(x_student)

            d = pre_tanh.view(pre_tanh.size(0), -1).size(1)
            grad_est = torch.zeros_like(pre_tanh)
            x_pert_query_list = []
            x_pert_student_list = []
            u_list = []
            for _ in range(self.m):
                u = torch.randn_like(pre_tanh)
                u /= (torch.norm(u.view(batch, -1), dim=1).view(-1, 1, 1, 1) + 1e-8)
                x_pert_tanh = torch.tanh(pre_tanh + self.epsilon * u)
                x_pert_query = self._query_scale(branch, x_pert_tanh)
                x_pert_query_list.append(x_pert_query)
                x_pert_student_list.append(x_pert_query)
                u_list.append(u)

            x_query_all = torch.cat([x_query] + x_pert_query_list, dim=0)
            v_all = self._query_with_budget(ctx, x_query_all)
            branch.consumed_budget += int(x_query_all.size(0))
            pbar.update(int(x_query_all.size(0)))

            v_out = v_all[:batch]
            v_pert_all = v_all[batch:].view(self.m, batch, -1)
            num_classes = int(v_out.size(1))
            loss_base = -F.l1_loss(s_out, v_out, reduction="none").mean(dim=1)

            for j in range(self.m):
                x_pert_j = x_pert_student_list[j]
                with torch.no_grad():
                    s_pert = branch.student(x_pert_j)
                v_pert = v_pert_all[j]
                loss_pert = -F.l1_loss(s_pert, v_pert, reduction="none").mean(dim=1)
                grad_est += (loss_pert - loss_base).view(-1, 1, 1, 1) * u_list[j]

            branch.g_opt.zero_grad()
            grad_est_scaled = grad_est * (d / (max(1, self.m) * self.epsilon))
            grad_est_scaled = grad_est_scaled / float(max(1, num_classes * batch))
            pre_tanh.backward(grad_est_scaled)
            branch.g_opt.step()

    def _run_student_step(self, branch: _BranchState, ctx: BenchmarkContext, device: str, pbar) -> None:
        for _ in range(self.n_s):
            remaining_branch = int(branch.target_budget - branch.consumed_budget)
            batch = min(int(self.batch_size), int(ctx.budget_remaining), int(remaining_branch))
            if batch <= 0:
                return

            branch.generator.eval()
            branch.student.train()
            z = torch.randn(batch, self.noise_dim, device=device)
            x_tanh = branch.generator(z).detach()
            x_query = self._query_scale(branch, x_tanh)
            x_student = x_query
            v_out = self._query_with_budget(ctx, x_query)
            branch.consumed_budget += int(x_query.size(0))
            pbar.update(int(x_query.size(0)))

            branch.s_opt.zero_grad()
            student_was_training = branch.student.training
            if batch < 2:
                branch.student.eval()
            loss = F.l1_loss(branch.student(x_student), v_out)
            loss.backward()
            branch.s_opt.step()
            if student_was_training:
                branch.student.train()

    def _evaluate_branch(self, branch: _BranchState, device: str, query_count: Optional[int] = None) -> None:
        q = int(branch.consumed_budget if query_count is None else query_count)
        eval_key = (f"track_b_{branch.name}", q)
        if eval_key in self._tracked_eval_points:
            return
        self.logger.info(
            "[TempDFME][%s][query_scale=%s] Eval at branch_queries=%d",
            branch.name,
            branch.query_scale_label,
            q,
        )
        self._evaluate_current_substitute(
            branch.student,
            device,
            track=f"track_b_{branch.name}",
            query_count=q,
        )

    def _maybe_periodic_eval_branch(self, branch: _BranchState, device: str) -> None:
        if int(self.eval_interval_queries) <= 0:
            return
        while int(branch.next_eval_query) > 0 and int(branch.consumed_budget) >= int(branch.next_eval_query):
            self._evaluate_branch(branch, device, query_count=int(branch.next_eval_query))
            branch.next_eval_query += int(self.eval_interval_queries)

    def _maybe_paired_eval(self, device: str) -> None:
        if int(self.eval_interval_queries) <= 0:
            return
        if self.baseline_branch is None or self.unit_branch is None:
            return

        paired_progress = min(
            int(self.baseline_branch.consumed_budget),
            int(self.unit_branch.consumed_budget),
        )
        while int(self._next_paired_eval_query) > 0 and paired_progress >= int(self._next_paired_eval_query):
            q = int(self._next_paired_eval_query)
            self.logger.info(
                "[TempDFME] Paired eval at branch_queries=%d for both variants", q
            )
            self._evaluate_branch(self.baseline_branch, device, query_count=q)
            self._evaluate_branch(self.unit_branch, device, query_count=q)
            self._next_paired_eval_query += int(self.eval_interval_queries)

    def _maybe_checkpoint_eval(self, device: str) -> None:
        reached = self.state.attack_state.get("checkpoint_reached", [])
        if not reached:
            return
        for checkpoint in reached:
            q_global = int(checkpoint)
            if q_global in self._checkpoint_eval_done:
                continue
            if self.baseline_branch is not None:
                q_baseline = min(q_global, int(self.baseline_branch.consumed_budget))
                self._evaluate_branch(self.baseline_branch, device, query_count=q_baseline)
            if self.unit_branch is not None:
                q_unit = min(q_global, int(self.unit_branch.consumed_budget))
                self._evaluate_branch(self.unit_branch, device, query_count=q_unit)
            self._checkpoint_eval_done.add(q_global)

    def run(self, ctx: BenchmarkContext) -> None:
        device = str(self.state.metadata.get("device", "cpu"))
        self.victim = ctx.oracle.model

        total_budget = int(ctx.budget_remaining)
        baseline_budget = int(total_budget // 2)
        unit_budget = int(total_budget - baseline_budget)

        self.baseline_branch = self._build_branch(
            self.state,
            "baseline_m11",
            baseline_budget,
            "tanh_m11",
            "[-1,1]",
        )
        self.unit_branch = self._build_branch(
            self.state,
            "unit_query_01",
            unit_budget,
            "unit_01",
            "[0,1]",
        )

        self.logger.info(
            "[TempDFME] Variants: baseline_m11(query_scale=[-1,1]), unit_query_01(query_scale=[0,1])"
        )

        pbar = self._create_progress_bar(total_budget, "[TempDFME] Dual Extracting")

        while ctx.budget_remaining > 0:
            progressed = False
            for branch in (self.baseline_branch, self.unit_branch):
                if branch is None:
                    continue
                if int(branch.consumed_budget) >= int(branch.target_budget):
                    continue

                self._decay_lr_if_needed(branch)
                before = int(branch.consumed_budget)
                self._run_generator_step(branch, ctx, device, pbar)
                self._run_student_step(branch, ctx, device, pbar)
                self._maybe_checkpoint_eval(device)
                if int(branch.consumed_budget) > before:
                    progressed = True
                if ctx.budget_remaining <= 0:
                    break

            self._maybe_paired_eval(device)

            if not progressed:
                break

        if self.baseline_branch is not None:
            self._evaluate_branch(self.baseline_branch, device)
        if self.unit_branch is not None:
            self._evaluate_branch(self.unit_branch, device)
        self._maybe_checkpoint_eval(device)

        baseline_student = self.baseline_branch.student if self.baseline_branch is not None else None
        unit_student = self.unit_branch.student if self.unit_branch is not None else None
        if baseline_student is not None:
            self.state.attack_state["substitute_baseline"] = baseline_student
        if unit_student is not None:
            self.state.attack_state["substitute_unit_query"] = unit_student
        self.state.attack_state["substitute"] = None

        self.state.attack_state["ablation_query_counts"] = {
            "baseline": int(self.baseline_branch.consumed_budget) if self.baseline_branch is not None else 0,
            "unit_query": int(self.unit_branch.consumed_budget) if self.unit_branch is not None else 0,
        }

        pbar.close()

    def observe(self, query_batch, oracle_output, state) -> None:
        return None
