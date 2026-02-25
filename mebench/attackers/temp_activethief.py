"""Internal ActiveThief ablation runner with dual pool/query paths."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset

from mebench.attackers.activethief import ActiveThief
from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.state import BenchmarkState
from mebench.core.types import OracleOutput
from mebench.data.loaders import get_surrogate_standard_normalization, get_test_dataloader
from mebench.eval.metrics import compute_accuracy, evaluate_substitute


@dataclass
class _BranchSpec:
    name: str
    label: str
    budget: int
    query_transform: Callable[[torch.Tensor], torch.Tensor]
    pool_sample_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None


@dataclass
class _BranchRuntime:
    spec: _BranchSpec
    state: BenchmarkState
    attack: ActiveThief
    ctx: _BranchContext
    step_size: int


class _MappedDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        sample_transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self._base_dataset = base_dataset
        self._sample_transform = sample_transform

    def __len__(self) -> int:
        return len(self._base_dataset)

    def __getitem__(self, idx: int):
        x, y = self._base_dataset[idx]
        return self._sample_transform(x), y


@dataclass(frozen=True)
class _IdentityTensorTransform:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


@dataclass(frozen=True)
class _ChannelNormalizeTransform:
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        channels = int(x.size(0))
        if len(self.mean) != channels or len(self.std) != channels:
            return x
        mean_t = torch.tensor(self.mean, dtype=x.dtype, device=x.device).view(channels, 1, 1)
        std_t = torch.tensor(self.std, dtype=x.dtype, device=x.device).view(channels, 1, 1)
        return (x - mean_t) / std_t.clamp_min(1e-8)


class _BranchContext:
    def __init__(
        self,
        parent_ctx: BenchmarkContext,
        branch_state: BenchmarkState,
        branch_budget: int,
        query_transform: Callable[[torch.Tensor], torch.Tensor],
        on_query: Optional[Callable[[int], None]] = None,
    ) -> None:
        self._parent_ctx = parent_ctx
        self.state = branch_state
        self.oracle = parent_ctx.oracle
        self.logger = parent_ctx.logger
        self._branch_budget_remaining = int(max(0, branch_budget))
        self._query_transform = query_transform
        self._on_query = on_query

    @property
    def budget_remaining(self) -> int:
        return int(min(self._branch_budget_remaining, self._parent_ctx.budget_remaining))

    @property
    def branch_budget_remaining(self) -> int:
        return int(self._branch_budget_remaining)

    @property
    def branch_consumed(self) -> int:
        return int(max(0, self.state.metadata.get("max_budget", 0) - self._branch_budget_remaining))

    def query(self, x_batch: torch.Tensor, meta: Optional[dict] = None) -> OracleOutput:
        if int(x_batch.size(0)) > int(self.budget_remaining):
            raise ValueError(
                f"Branch budget exceeded: requested={int(x_batch.size(0))}, "
                f"remaining={int(self.budget_remaining)}"
            )
        x_query = self._query_transform(x_batch)
        oracle_out = self._parent_ctx.query(x_query, meta=meta)
        self._branch_budget_remaining -= int(x_batch.size(0))
        self.state.query_count += int(x_batch.size(0))
        self.state.budget_remaining = int(self._branch_budget_remaining)
        if self._on_query is not None:
            self._on_query(int(x_batch.size(0)))
        return oracle_out


class TempActiveThief(AttackRunner):
    """Compare two ActiveThief variants in one run.

    - raw_query_raw_train: query victim with raw pool images, train on raw queried images
    - norm_pool_query_train: normalize pool tensors with surrogate-standard transform,
      then query and train in that same normalized space
    """

    def __init__(self, config: dict, state: BenchmarkState):
        super().__init__(config, state)
        self._consistency_victim_acc_min = float(config.get("consistency_victim_acc_min", 0.8))
        self._consistency_agreement_min = float(config.get("consistency_agreement_min", 0.9))
        self._consistency_acc_max = float(config.get("consistency_acc_max", 0.2))

    def _identity(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _build_norm_pool_transform(self) -> Callable[[torch.Tensor], torch.Tensor]:
        dataset_cfg = self.state.metadata.get("dataset_config", {}) or {}
        mode = str(dataset_cfg.get("surrogate_normalization", "standard")).strip().lower()

        if mode in {"none", "off", "identity", "raw"}:
            return _IdentityTensorTransform()

        if mode == "custom":
            mean = dataset_cfg.get("surrogate_norm_mean")
            std = dataset_cfg.get("surrogate_norm_std")
            if isinstance(mean, (list, tuple)) and isinstance(std, (list, tuple)) and len(mean) == len(std):
                return _ChannelNormalizeTransform(
                    mean=tuple(float(v) for v in mean),
                    std=tuple(float(v) for v in std),
                )
            self.logger.warning(
                "[TempActiveThief] Invalid custom surrogate normalization; "
                "falling back to identity for norm-pool branch."
            )
            return _IdentityTensorTransform()

        surrogate_name = str(dataset_cfg.get("surrogate_name") or dataset_cfg.get("name") or "SVHN")
        channels_raw = dataset_cfg.get("channels")
        channels = int(channels_raw) if channels_raw is not None else None
        try:
            mean, std = get_surrogate_standard_normalization(
                surrogate_name=surrogate_name,
                channels=channels,
            )
        except ValueError as exc:
            self.logger.warning(
                "[TempActiveThief] Unable to resolve surrogate standard normalization "
                "(surrogate=%s): %s. Falling back to identity.",
                surrogate_name,
                str(exc),
            )
            return _IdentityTensorTransform()
        return _ChannelNormalizeTransform(mean=tuple(float(v) for v in mean), std=tuple(float(v) for v in std))

    def _make_branch_state(self, branch_budget: int) -> BenchmarkState:
        metadata = copy.deepcopy(self.state.metadata)
        metadata["max_budget"] = int(branch_budget)
        return BenchmarkState(
            query_count=0,
            budget_remaining=int(branch_budget),
            checkpoint_reached=False,
            attack_state={},
            metadata=metadata,
        )

    def _evaluate_branch_substitute(
        self,
        branch_attack: ActiveThief,
        spec: _BranchSpec,
        substitute,
        eval_device: str,
        *,
        query_count: Optional[int] = None,
    ) -> None:
        if substitute is None or self.victim is None:
            return

        query_step = branch_attack.state.query_count if query_count is None else int(query_count)

        track_name = f"track_b_{spec.name}"
        eval_key = (track_name, int(query_step))
        if eval_key in branch_attack._tracked_eval_points:
            return
        branch_attack._tracked_eval_points.add(eval_key)

        if branch_attack.test_loader is None:
            dataset_name = branch_attack.state.metadata.get("dataset_config", {}).get("name", "CIFAR10")
            victim_cfg = branch_attack.state.metadata.get("victim_config", {}) or {}
            input_size = victim_cfg.get("input_size")
            size = None
            if isinstance(input_size, (list, tuple)) and len(input_size) == 2:
                size = (int(input_size[0]), int(input_size[1]))
            channels = victim_cfg.get("channels")
            branch_attack.test_loader = get_test_dataloader(
                dataset_name,
                batch_size=128,
                input_size=size,
                channels=int(channels) if channels is not None else None,
            )

        metrics = evaluate_substitute(
            substitute=substitute,
            victim=self.victim,
            test_loader=branch_attack.test_loader,
            device=eval_device,
            output_mode=branch_attack.config.get("output_mode", "soft_prob"),
        )
        victim_acc_eval = compute_accuracy(self.victim, branch_attack.test_loader, eval_device)

        labeled_indices = branch_attack.state.attack_state.get("labeled_indices", [])
        labeled_total = int(len(labeled_indices))
        labeled_unique = int(len(set(int(i) for i in labeled_indices)))
        labeled_duplicates = int(max(0, labeled_total - labeled_unique))

        kl_value = metrics.get("kl_mean")
        kl_print = float(kl_value) if kl_value is not None else 0.0

        self.logger.info(
            "[TempActiveThief][%s][Evaluation] Labeled: %d (unique=%d, dup=%d), "
            "VictimAcc: %.4f, Acc: %.4f, Agreement: %.4f, KL: %.4f",
            spec.name,
            labeled_total,
            labeled_unique,
            labeled_duplicates,
            float(victim_acc_eval),
            float(metrics.get("acc_gt", 0.0)),
            float(metrics.get("agreement", 0.0)),
            kl_print,
        )

        if (
            float(victim_acc_eval) >= self._consistency_victim_acc_min
            and float(metrics.get("agreement", 0.0)) >= self._consistency_agreement_min
            and float(metrics.get("acc_gt", 1.0)) <= self._consistency_acc_max
        ):
            self.logger.warning(
                "[TempActiveThief][%s] Suspicious eval combination detected "
                "(victim_acc=%.4f, agreement=%.4f, acc_gt=%.4f).",
                spec.name,
                float(victim_acc_eval),
                float(metrics.get("agreement", 0.0)),
                float(metrics.get("acc_gt", 0.0)),
            )

        if branch_attack.ctx and branch_attack.ctx.logger is not None:
            metrics_with_counts = dict(metrics)
            metrics_with_counts.update(
                {
                    "victim_acc_eval": float(victim_acc_eval),
                    "labeled_total": float(labeled_total),
                    "labeled_unique": float(labeled_unique),
                    "labeled_duplicates": float(labeled_duplicates),
                }
            )
            branch_attack.ctx.logger.log_history(step=query_step, metrics=metrics_with_counts)
            seed = branch_attack.state.metadata.get("seed", 0)
            branch_attack.ctx.logger.log_checkpoint(
                seed=seed,
                checkpoint=query_step,
                track=track_name,
                metrics=metrics_with_counts,
            )
            branch_attack.ctx.logger.save_metrics_csv()

    def _prepare_branch(
        self,
        parent_ctx: BenchmarkContext,
        spec: _BranchSpec,
        device: str,
        pbar,
    ) -> Optional[_BranchRuntime]:
        if int(spec.budget) <= 0:
            return None

        branch_state = self._make_branch_state(int(spec.budget))
        branch_attack = ActiveThief(copy.deepcopy(self.config), branch_state)
        branch_attack.ctx = self.ctx
        branch_attack.victim = self.victim

        def _branch_eval(substitute, eval_device, *, track="track_b", query_count=None):
            _ = track
            return self._evaluate_branch_substitute(
                branch_attack,
                spec,
                substitute,
                eval_device,
                query_count=query_count,
            )

        branch_attack._evaluate_current_substitute = _branch_eval  # type: ignore[method-assign]

        branch_ctx = _BranchContext(
            parent_ctx=parent_ctx,
            branch_state=branch_state,
            branch_budget=int(spec.budget),
            query_transform=spec.query_transform,
            on_query=lambda n: pbar.update(int(n)),
        )

        self.logger.info(
            "[TempActiveThief][%s] start (budget=%d, mode=%s)",
            spec.name,
            int(spec.budget),
            spec.label,
        )

        if not branch_state.attack_state.get("initialized"):
            branch_attack._setup_datasets(branch_state)

        if spec.pool_sample_transform is not None:
            branch_attack.pool_dataset = _MappedDataset(
                branch_attack.pool_dataset,
                spec.pool_sample_transform,
            )

        branch_attack._bootstrap_seed_and_validation_sets(branch_ctx, branch_state)

        step_size = branch_attack.step_size
        if step_size is None:
            rounds = max(1, int(branch_attack.rounds))
            active_budget = int(branch_ctx.budget_remaining)
            step_size = max(1, int(torch.ceil(torch.tensor(active_budget / rounds)).item()))
            branch_attack.step_size = int(step_size)

        if branch_attack.substitute is None and branch_attack.labeled_indices:
            branch_attack._train_substitute(branch_state)
            if branch_attack.substitute is not None:
                branch_attack._evaluate_current_substitute(branch_attack.substitute, device)

        return _BranchRuntime(
            spec=spec,
            state=branch_state,
            attack=branch_attack,
            ctx=branch_ctx,
            step_size=int(step_size),
        )

    def _run_branch_round(self, runtime: _BranchRuntime, device: str) -> bool:
        attack = runtime.attack
        state = runtime.state
        branch_ctx = runtime.ctx

        if branch_ctx.budget_remaining <= 0 or not attack.unlabeled_indices:
            return False

        step_size = min(int(runtime.step_size), int(branch_ctx.budget_remaining), len(attack.unlabeled_indices))
        if step_size <= 0:
            return False

        query_batch = attack._select_query_batch(step_size, state)
        round_id = state.attack_state.get("round", 0)
        self.logger.info(
            "[TempActiveThief][%s] round=%s selected=%s (labeled=%s, unlabeled=%s)",
            runtime.spec.name,
            round_id,
            int(query_batch.x.shape[0]),
            len(attack.labeled_indices),
            len(attack.unlabeled_indices),
        )

        if int(query_batch.x.shape[0]) == 0:
            return False

        oracle_output = branch_ctx.query(query_batch.x, meta=query_batch.meta)
        attack.observe(query_batch, oracle_output, state)

        if attack.labeled_indices:
            attack._train_substitute(state)
            self.logger.info(
                "[TempActiveThief][%s] round=%s training complete (branch_budget_remaining=%s)",
                runtime.spec.name,
                round_id,
                branch_ctx.budget_remaining,
            )
            attack._evaluate_current_substitute(attack.substitute, device)

        return True

    def _finalize_branch(self, runtime: _BranchRuntime) -> None:
        model = runtime.state.attack_state.get("substitute")
        if model is None:
            model = getattr(runtime.attack, "substitute", None)
        consumed = int(runtime.spec.budget - runtime.ctx.branch_budget_remaining)

        self.state.attack_state[f"substitute_{runtime.spec.name}"] = model
        self.state.attack_state.setdefault("ablation_query_counts", {})[runtime.spec.name] = consumed
        self.logger.info(
            "[TempActiveThief][%s] done (consumed=%d)",
            runtime.spec.name,
            consumed,
        )

    def run(self, ctx: BenchmarkContext) -> None:
        self.victim = ctx.oracle.model
        device = str(self.state.metadata.get("device", "cpu"))
        total_budget = int(ctx.budget_remaining)
        raw_budget = int(total_budget // 2)
        norm_budget = int(total_budget - raw_budget)
        norm_pool_transform = self._build_norm_pool_transform()

        specs = [
            _BranchSpec(
                name="raw_query_raw_train",
                label="query=raw, train=raw",
                budget=raw_budget,
                query_transform=self._identity,
            ),
            _BranchSpec(
                name="norm_pool_query_train",
                label="pool=surrogate_norm(raw), query=pool, train=pool",
                budget=norm_budget,
                query_transform=self._identity,
                pool_sample_transform=norm_pool_transform,
            ),
        ]

        self.logger.info(
            "[TempActiveThief] Variants: raw_query_raw_train(query=raw,train=raw), "
            "norm_pool_query_train(pool=surrogate_norm(raw),query=pool,train=pool)"
        )

        pbar = self._create_progress_bar(total_budget, "[TempActiveThief] Ablation")
        runtimes = []
        for spec in specs:
            runtime = self._prepare_branch(ctx, spec, device, pbar)
            if runtime is not None:
                runtimes.append(runtime)

        while ctx.budget_remaining > 0:
            progressed = False
            for runtime in runtimes:
                if self._run_branch_round(runtime, device):
                    progressed = True
                if ctx.budget_remaining <= 0:
                    break
            if not progressed:
                break

        for runtime in runtimes:
            self._finalize_branch(runtime)
        pbar.close()

        self.state.attack_state["substitute"] = None
