"""MARICH attack implementation.

Reference: MARICH paper and official repository clone in `official_repo_clones/MARICH`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import math
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import models as tv_models

from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.data.loaders import create_dataloader
from mebench.data.preprocessing import apply_official_preprocess_batch
from mebench.models.substitute_factory import create_substitute
from mebench.training import SubstituteTrainer, TrainRequest
from mebench.utils.dataloader import (
    pool_loader_kwargs,
    resolve_pool_num_workers,
    resolve_train_num_workers,
    resolve_val_num_workers,
)


class _IndexedDataset(Dataset):
    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, _ = self.base[idx]
        return x, int(idx)


class MARICH(AttackRunner):
    """MARICH: multi-stage active model extraction.

    Official pipeline (image branch) uses staged selection:
    entropy -> loss-based proximity -> gradient-based filtering.

    Note:
    This implementation intentionally preserves the paper/original staged protocol.
    Harmonization with the benchmark's unified pool-based protocol is tracked as
    a follow-up fairness improvement task.
    """

    def __init__(self, config: Dict[str, Any], state: BenchmarkState) -> None:
        super().__init__(config, state)
        self.batch_size = int(config.get("batch_size", 128))
        self.init_points = int(config.get("init_points", 1000))
        self.rounds = int(config.get("rounds", 20))
        self.round_budget = float(config.get("budget", 300.0))
        self.budget_growth = float(config.get("budget_growth", 1.01))
        self.epochs = float(config.get("epochs", 20.0))
        self.epochs_growth = float(config.get("epochs_growth", 1.02))
        self.gamma1 = float(config.get("gamma1", 0.8))
        self.gamma2 = float(config.get("gamma2", 0.8))
        self.sampling = str(config.get("sampling", "all_elg")).strip().lower()
        self.selection_batch_size = int(config.get("selection_batch_size", self.batch_size))
        self.num_clusters = int(config.get("num_clusters", 10))
        self.validation_source = str(config.get("validation_source", "queried_holdout")).strip().lower()
        self.query_preprocess_profile = config.get("query_preprocess_profile")
        self.show_scoring_progress = bool(config.get("show_scoring_progress", True))
        self.scoring_progress_min_candidates = int(
            config.get("scoring_progress_min_candidates", self.selection_batch_size)
        )
        self.num_classes = int(
            state.metadata.get("num_classes")
            or config.get("num_classes")
            or state.metadata.get("dataset_config", {}).get("num_classes", 10)
        )
        self.lr = float(config.get("lr", 0.01))
        self.weight_decay = float(config.get("weight_decay", 5e-4))
        self.patience = int(config.get("patience", 20))

        self.pool_dataset: Optional[Dataset] = None
        self.substitute: Optional[nn.Module] = None

        self._initialize_state(state)

    def _initialize_state(self, state: BenchmarkState) -> None:
        state.attack_state.setdefault("labeled_indices", [])
        state.attack_state.setdefault("unlabeled_indices", [])
        state.attack_state.setdefault("query_data_x", [])
        state.attack_state.setdefault("query_data_y", [])
        state.attack_state.setdefault("val_query_data_x", [])
        state.attack_state.setdefault("val_query_data_y", [])
        state.attack_state.setdefault("initialized", False)
        state.attack_state.setdefault("round", 0)

    def _setup_pool(self, state: BenchmarkState) -> None:
        if bool(state.attack_state.get("initialized", False)):
            return
        dataset_config = state.metadata.get("dataset_config", {})
        loader = create_dataloader(dataset_config, batch_size=1, shuffle=False)
        self.pool_dataset = loader.dataset
        n = len(self.pool_dataset)
        state.attack_state["labeled_indices"] = []
        state.attack_state["unlabeled_indices"] = list(range(n))
        state.attack_state["initialized"] = True

    def _prepare_pool_split_validation(self, device: str) -> None:
        """Build official-style fixed validation split from the full pool.

        This mirrors MARICH notebook behavior where validation data is a fixed
        holdout split from the public pool and not part of active querying.
        """

        if self.validation_source != "pool_split":
            return
        if bool(self.state.attack_state.get("pool_validation_ready", False)):
            return
        if self.pool_dataset is None:
            return

        n = len(self.pool_dataset)
        min_train_pool = max(1, int(self.config.get("pool_validation_min_train_pool", 1)))
        if n <= min_train_pool:
            self.state.attack_state["pool_validation_ready"] = True
            return

        if "validation_seed_size" in self.config:
            val_size = int(self.config.get("validation_seed_size", 0))
        elif "validation_budget_size" in self.config:
            val_size = int(self.config.get("validation_budget_size", 0))
        else:
            ratio = float(self.config.get("validation_budget_ratio", 0.2))
            ratio = max(0.0, min(0.95, ratio))
            val_size = int(round(n * ratio))

        val_size = max(0, min(int(val_size), n - min_train_pool))
        if val_size <= 0:
            self.state.attack_state["pool_validation_ready"] = True
            return

        split_seed = int(self.config.get("validation_split_seed", self.state.metadata.get("seed", 0)))
        gen = torch.Generator().manual_seed(split_seed)
        perm = torch.randperm(n, generator=gen)
        val_idx = [int(i) for i in perm[:val_size].tolist()]
        train_idx = [int(i) for i in perm[val_size:].tolist()]

        self.state.attack_state["pool_validation_indices"] = val_idx
        self.state.attack_state["unlabeled_indices"] = train_idx

        subset = Subset(_IndexedDataset(self.pool_dataset), val_idx)
        loader = DataLoader(
            subset,
            batch_size=min(self.batch_size, len(val_idx)),
            shuffle=False,
            **self._pool_scan_loader_kwargs(device),
        )

        xs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        self.victim.eval()
        with torch.no_grad():
            for x_batch, _ in loader:
                x_query = self._apply_query_preprocess(x_batch)
                logits = self.victim(x_query.to(device))
                y = torch.argmax(logits, dim=1).detach().cpu().long()
                xs.append(x_query.detach().cpu())
                ys.append(y)

        if xs and ys:
            self.state.attack_state["val_query_data_x"] = [torch.cat(xs, dim=0)]
            self.state.attack_state["val_query_data_y"] = [torch.cat(ys, dim=0)]
            self.logger.info(
                "[MARICH] Prepared pool-split validation set: val=%d train_pool=%d",
                int(val_size),
                int(len(train_idx)),
            )

        self.state.attack_state["pool_validation_ready"] = True

    def _ensure_marich_validation_holdout(self) -> None:
        """Create a fixed validation split from currently queried data.

        Official MARICH image notebooks use a fixed validation split that is kept
        separate from training across rounds. In benchmark mode we cannot assume
        free oracle labels for the full pool, so we apply the same fixed-holdout
        idea to the queried set available at the first training call.
        """

        if self.validation_source == "pool_split":
            return

        state = self.state
        state.attack_state.setdefault("val_query_data_x", [])
        state.attack_state.setdefault("val_query_data_y", [])

        if state.attack_state.get("val_query_data_x") and state.attack_state.get("val_query_data_y"):
            return

        query_x = state.attack_state.get("query_data_x", [])
        query_y = state.attack_state.get("query_data_y", [])
        if len(query_x) == 0 or len(query_y) == 0:
            return

        x_all = torch.cat(query_x, dim=0)
        y_all = torch.cat(query_y, dim=0)
        total = int(x_all.size(0))
        min_train = max(2, int(self.config.get("validation_min_train", 2)))
        if total <= min_train:
            return

        if "validation_seed_size" in self.config:
            val_size = int(self.config.get("validation_seed_size", 0))
        elif "validation_budget_size" in self.config:
            val_size = int(self.config.get("validation_budget_size", 0))
        else:
            raw_ratio = self.config.get(
                "validation_budget_ratio",
                self.config.get("validation_seed_ratio", 0.2),
            )
            ratio = float(raw_ratio)
            ratio = max(0.0, min(0.95, ratio))
            val_size = int(round(total * ratio))

        val_size = max(0, min(int(val_size), total - min_train))
        if val_size <= 0:
            return

        split_seed = int(self.config.get("validation_split_seed", state.metadata.get("seed", 0)))
        gen = torch.Generator().manual_seed(split_seed)
        perm = torch.randperm(total, generator=gen)
        val_idx = perm[:val_size]
        train_idx = perm[val_size:]

        state.attack_state["val_query_data_x"] = [x_all[val_idx].detach().cpu()]
        state.attack_state["val_query_data_y"] = [y_all[val_idx].detach().cpu()]
        state.attack_state["query_data_x"] = [x_all[train_idx].detach().cpu()]
        state.attack_state["query_data_y"] = [y_all[train_idx].detach().cpu()]

    def _apply_query_preprocess(self, x_batch: torch.Tensor) -> torch.Tensor:
        profile = self.query_preprocess_profile
        if profile is None:
            return x_batch
        return apply_official_preprocess_batch(x_batch, str(profile))

    def _pool_scan_loader_kwargs(self, device: str) -> Dict[str, Any]:
        dataset_config = self.state.metadata.get("dataset_config", {})
        pool_workers = resolve_pool_num_workers(self.config, dataset_config)
        if pool_workers is None:
            return pool_loader_kwargs(device)
        return pool_loader_kwargs(device, {"num_workers": int(pool_workers)})

    def _create_substitute(self, device: str) -> nn.Module:
        sub_config = self.state.metadata.get("substitute_config", {})
        arch = str(sub_config.get("arch", "resnet18")).strip().lower()
        use_pretrained = bool(sub_config.get("pretrained", False))
        input_shape = self.state.metadata.get("input_shape", (3, 32, 32))
        input_channels = int(input_shape[0])

        if use_pretrained and arch == "resnet18":
            if input_channels != 3:
                raise ValueError(
                    "pretrained ResNet18 substitute requires input_channels=3, "
                    f"got {input_channels}"
                )
            try:
                model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
            except Exception as exc:  # pragma: no cover - depends on local torchvision cache
                raise RuntimeError(
                    "Failed to load torchvision pretrained ResNet18 weights. "
                    "Ensure internet/cache availability or set substitute.pretrained=false."
                ) from exc
            model.fc = nn.Linear(int(model.fc.in_features), int(self.num_classes))
            return model.to(device)

        width_mult = int(sub_config.get("width_mult", 1))
        dropout_prob = float(sub_config.get("dropout_prob", 0.0))
        return create_substitute(
            arch=arch,
            num_classes=self.num_classes,
            input_channels=input_channels,
            width_mult=width_mult,
            dropout_prob=dropout_prob,
        ).to(device)

    def _query_indices(self, ctx: BenchmarkContext, indices: List[int], *, device: str) -> None:
        if self.pool_dataset is None or not indices:
            return
        subset = Subset(_IndexedDataset(self.pool_dataset), indices)
        loader_kwargs = self._pool_scan_loader_kwargs(device)
        loader = DataLoader(
            subset,
            batch_size=min(self.batch_size, len(indices)),
            shuffle=False,
            **loader_kwargs,
        )
        for x_batch, idx_batch in loader:
            x_query = self._apply_query_preprocess(x_batch)
            qb = QueryBatch(x=x_query, meta={"indices": [int(i) for i in idx_batch.tolist()]})
            out = ctx.query(qb.x, meta=qb.meta)
            self.observe(qb, out, self.state)

    def _create_scoring_pbar(self, total: int, stage: str, budget: int):
        if not self.show_scoring_progress:
            return None
        min_candidates = max(1, int(self.scoring_progress_min_candidates))
        if int(total) < min_candidates:
            return None
        desc = f"[MARICH] Scoring {stage} ({int(total)}->{int(budget)})"
        return self._create_progress_bar(total=int(total), desc=desc)

    def _entropy_select(self, candidates: List[int], budget: int, device: str) -> List[int]:
        if self.pool_dataset is None or self.substitute is None or budget <= 0 or not candidates:
            return []
        subset = Subset(_IndexedDataset(self.pool_dataset), candidates)
        loader = DataLoader(
            subset,
            batch_size=min(self.selection_batch_size, len(candidates)),
            shuffle=False,
            **self._pool_scan_loader_kwargs(device),
        )
        self.substitute.eval()
        scored: List[tuple[int, float]] = []
        pbar = self._create_scoring_pbar(len(candidates), "entropy", min(int(budget), len(candidates)))
        try:
            with torch.no_grad():
                for x_batch, idx_batch in loader:
                    x_batch = self._apply_query_preprocess(x_batch).to(device)
                    probs = F.softmax(self.substitute(x_batch), dim=1)
                    ent = -(probs * torch.log(probs.clamp_min(1e-10))).sum(dim=1)
                    for i, idx in enumerate(idx_batch.tolist()):
                        scored.append((int(idx), float(ent[i].item())))
                    if pbar is not None:
                        pbar.update(int(x_batch.size(0)))
        finally:
            if pbar is not None:
                pbar.close()
        scored.sort(key=lambda t: t[1], reverse=True)
        return [idx for idx, _ in scored[: min(int(budget), len(scored))]]

    def _loss_dep_select(self, candidates: List[int], budget: int, device: str) -> List[int]:
        labeled = list(self.state.attack_state.get("labeled_indices", []))
        if self.pool_dataset is None or self.substitute is None or not labeled or not candidates or budget <= 0:
            return candidates[: min(int(budget), len(candidates))]

        if len(self.state.attack_state.get("query_data_x", [])) == 0:
            return candidates[: min(int(budget), len(candidates))]

        x_train = torch.cat(self.state.attack_state["query_data_x"], dim=0).to(device)
        y_train = torch.cat(self.state.attack_state["query_data_y"], dim=0).to(device)
        self.substitute.eval()
        with torch.no_grad():
            logits = self.substitute(x_train)
            losses = F.cross_entropy(logits, y_train.long(), reduction="none")
        num_centers = max(1, min(int(self.num_clusters), int(losses.numel())))
        center_idx = torch.argsort(losses, descending=True)[:num_centers]
        centers = x_train[center_idx].view(num_centers, -1).detach().cpu()

        subset = Subset(_IndexedDataset(self.pool_dataset), candidates)
        loader = DataLoader(
            subset,
            batch_size=min(self.selection_batch_size, len(candidates)),
            shuffle=False,
            **self._pool_scan_loader_kwargs(device),
        )
        cand_feats: List[np.ndarray] = []
        cand_indices: List[int] = []
        pbar = self._create_scoring_pbar(len(candidates), "loss", min(int(budget), len(candidates)))
        try:
            for x_batch, idx_batch in loader:
                x_flat = self._apply_query_preprocess(x_batch).view(x_batch.size(0), -1)
                cand_feats.append(x_flat.detach().cpu().numpy())
                cand_indices.extend([int(idx) for idx in idx_batch.tolist()])
                if pbar is not None:
                    pbar.update(int(x_batch.size(0)))
        finally:
            if pbar is not None:
                pbar.close()

        if not cand_feats:
            return []

        x = np.concatenate(cand_feats, axis=0)
        center_np = centers.numpy()
        diff = x[None, :, :] - center_np[:, None, :]
        norms = np.linalg.norm(diff, axis=2)
        sorted_indices = np.argsort(norms, axis=1)

        target = min(int(budget), len(cand_indices))
        t = max(1, int(target / max(1, num_centers)))
        while len(np.unique(sorted_indices[:, :t])) < target and t < len(cand_indices):
            t += 1
        chosen = np.unique(sorted_indices[:, :t].reshape(-1))
        return [cand_indices[int(i)] for i in chosen.tolist()]

    def _engrad_select(self, candidates: List[int], budget: int, device: str) -> List[int]:
        if self.pool_dataset is None or self.substitute is None or budget <= 0 or not candidates:
            return []
        subset = Subset(_IndexedDataset(self.pool_dataset), candidates)
        loader = DataLoader(
            subset,
            batch_size=min(self.selection_batch_size, len(candidates)),
            shuffle=False,
            **self._pool_scan_loader_kwargs(device),
        )
        grads: List[np.ndarray] = []
        cand_indices: List[int] = []
        self.substitute.eval()
        pbar = self._create_scoring_pbar(len(candidates), "engrad", min(int(budget), len(candidates)))
        try:
            for x_batch, idx_batch in loader:
                x_batch = self._apply_query_preprocess(x_batch).to(device)
                x_batch.requires_grad_(True)
                logits = self.substitute(x_batch)
                probs = F.softmax(logits, dim=0)
                ent = torch.special.entr(probs).sum()
                self.substitute.zero_grad(set_to_none=True)
                ent.backward()
                grad = x_batch.grad.view(x_batch.size(0), -1).detach().cpu().numpy()
                grads.append(grad)
                cand_indices.extend([int(idx) for idx in idx_batch.tolist()])
                if pbar is not None:
                    pbar.update(int(x_batch.size(0)))
        finally:
            if pbar is not None:
                pbar.close()

        if not grads:
            return []

        x = np.concatenate(grads, axis=0)
        n = int(x.shape[0])
        clusters = max(1, min(int(self.num_clusters), n))
        kmeans = KMeans(n_clusters=clusters, random_state=0, n_init=10)
        kmeans.fit(x)
        d = np.hstack([kmeans.transform(x), np.array(cand_indices, dtype=np.int64).reshape(n, 1)])

        indices = np.zeros((clusters, n), dtype=np.int64)
        for i in range(clusters):
            ordered = np.array(sorted(d, key=lambda row: row[i]))[:, -1].astype(np.int64)
            indices[i] = ordered

        target = min(int(budget), n)
        t = max(1, int(target / max(1, clusters)))
        while len(np.unique(indices[:, :t])) < target and t < n:
            t += 1
        chosen = np.unique(indices[:, :t].reshape(-1))
        return [int(v) for v in chosen.tolist()]

    def _select_round_indices(self, k: int, device: str) -> List[int]:
        unlabeled = list(self.state.attack_state.get("unlabeled_indices", []))
        if k <= 0 or not unlabeled:
            return []
        if self.substitute is None or self.sampling == "random":
            return np.random.choice(unlabeled, size=min(k, len(unlabeled)), replace=False).astype(int).tolist()

        if self.sampling in {"entropy", "all_elg", "all_egl", "entropy-loss", "entropy-engrad"}:
            s1_budget = int(max(1, round(float(k) / max(1e-8, self.gamma1 * self.gamma2))))
            cand = self._entropy_select(unlabeled, min(s1_budget, len(unlabeled)), device)
        elif self.sampling in {"loss", "loss-engrad"}:
            cand = unlabeled
        elif self.sampling == "engrad":
            cand = unlabeled
        else:
            cand = unlabeled

        if self.sampling in {"all_elg", "entropy-loss", "loss", "loss-engrad"}:
            s2_budget = int(max(1, round(self.gamma1 * len(cand)))) if cand else 0
            cand = self._loss_dep_select(cand, min(s2_budget, len(cand)), device)

        if self.sampling in {"all_elg", "all_egl", "engrad", "entropy-engrad", "loss-engrad"}:
            s3_budget = int(max(1, round(self.gamma2 * len(cand)))) if cand else 0
            picked = self._engrad_select(cand, min(s3_budget, len(cand)), device)
        else:
            picked = cand

        if not picked:
            return np.random.choice(unlabeled, size=min(k, len(unlabeled)), replace=False).astype(int).tolist()
        return [int(i) for i in picked]

    def _train_substitute(self, epochs: int, device: str) -> None:
        self._ensure_marich_validation_holdout()
        qx = self.state.attack_state.get("query_data_x", [])
        qy = self.state.attack_state.get("query_data_y", [])
        val_qx = self.state.attack_state.get("val_query_data_x", [])
        val_qy = self.state.attack_state.get("val_query_data_y", [])
        if len(qx) == 0 or len(qy) == 0:
            return

        if self.substitute is None:
            self.substitute = self._create_substitute(device)

        x_all = torch.cat(qx, dim=0)
        y_all = torch.cat(qy, dim=0)
        if int(x_all.size(0)) < 2:
            return

        train_size = int(x_all.size(0))
        if train_size < 1:
            return

        sub_cfg = dict(self.state.metadata.get("substitute_config", {}))
        train_batch_size = int(sub_cfg.get("batch_size", self.batch_size))
        train_workers = resolve_train_num_workers(sub_cfg, self.config, default=0)
        val_workers = resolve_val_num_workers(sub_cfg, self.config, default=train_workers)

        train_loader = DataLoader(
            TensorDataset(x_all, y_all),
            batch_size=min(max(1, train_batch_size), train_size),
            shuffle=True,
            **pool_loader_kwargs(device, {"num_workers": int(train_workers)}),
        )

        val_loader: Optional[DataLoader] = None
        if len(val_qx) > 0 and len(val_qy) > 0:
            x_val = torch.cat(val_qx, dim=0)
            y_val = torch.cat(val_qy, dim=0)
            if int(x_val.size(0)) > 0:
                val_loader = DataLoader(
                    TensorDataset(x_val, y_val),
                    batch_size=min(max(1, train_batch_size), int(x_val.size(0))),
                    shuffle=False,
                    **pool_loader_kwargs(device, {"num_workers": int(val_workers)}),
                )

        def loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return F.cross_entropy(outputs, targets.long())

        def eval_fn(model_local: nn.Module, loader_local: DataLoader) -> float:
            model_local.eval()
            total_loss = 0.0
            total_count = 0
            with torch.no_grad():
                for x_val_b, y_val_b in loader_local:
                    x_val_b = x_val_b.to(device)
                    y_val_b = y_val_b.to(device)
                    outputs = model_local(x_val_b)
                    loss = F.cross_entropy(outputs, y_val_b.long())
                    total_loss += float(loss.item()) * int(x_val_b.size(0))
                    total_count += int(x_val_b.size(0))
            return total_loss / max(1, total_count)

        train_cfg = dict(sub_cfg)
        opt_cfg = dict(train_cfg.get("optimizer", {}))
        opt_cfg.setdefault("name", "sgd")
        opt_cfg.setdefault("lr", self.lr)
        opt_cfg.setdefault("momentum", 0.9)
        opt_cfg.setdefault("weight_decay", self.weight_decay)
        train_cfg["optimizer"] = opt_cfg
        train_cfg["max_epochs"] = int(max(1, epochs))
        train_cfg["patience"] = int(max(1, train_cfg.get("patience", self.patience)))

        trainer = SubstituteTrainer(train_cfg, device=device, logger=self.logger)
        steps_per_epoch = max(1, int(math.ceil(max(1, train_size) / max(1, train_batch_size))))
        request = TrainRequest(
            model=self.substitute,
            train_loader=train_loader,
            val_loader=val_loader,
            eval_fn=eval_fn if val_loader is not None else None,
            loss_fn=loss_fn,
            max_steps=int(train_cfg["max_epochs"]) * steps_per_epoch,
            validate_every=steps_per_epoch,
            patience=int(train_cfg["patience"]) * steps_per_epoch,
            early_stop_mode="min",
            load_best=True,
        )
        trainer.train(request)
        self.state.attack_state["substitute"] = self.substitute

    def run(self, ctx: BenchmarkContext) -> None:
        self.victim = ctx.oracle.model
        device = str(self.state.metadata.get("device", "cpu"))
        self._setup_pool(self.state)
        self._prepare_pool_split_validation(device)

        if str(self.config.get("output_mode", "hard_top1")) != "hard_top1":
            raise ValueError("marich requires hard_top1 output mode")

        unlabeled = self.state.attack_state["unlabeled_indices"]
        init_k = min(int(self.init_points), int(ctx.budget_remaining), len(unlabeled))
        init_indices = np.random.choice(unlabeled, size=init_k, replace=False).astype(int).tolist() if init_k > 0 else []
        for idx in init_indices:
            unlabeled.remove(idx)
        self.state.attack_state["labeled_indices"].extend(init_indices)
        self._query_indices(ctx, init_indices, device=device)
        self._train_substitute(int(round(self.epochs)), device)
        if self.substitute is not None:
            self._evaluate_current_substitute(self.substitute, device)

        budget_now = float(self.round_budget)
        epochs_now = float(self.epochs)
        for r in range(int(self.rounds)):
            if int(ctx.budget_remaining) <= 0:
                break
            unlabeled = self.state.attack_state["unlabeled_indices"]
            if not unlabeled:
                break

            budget_now = float(budget_now) * float(self.budget_growth)
            k = min(int(max(1, round(budget_now))), len(unlabeled))
            if k <= 0:
                break

            selected = self._select_round_indices(k, device)
            if len(selected) > int(ctx.budget_remaining):
                selected = selected[: int(ctx.budget_remaining)]
            selected_set = set(selected)
            self.state.attack_state["unlabeled_indices"] = [i for i in unlabeled if i not in selected_set]
            self.state.attack_state["labeled_indices"].extend(selected)
            self._query_indices(ctx, selected, device=device)

            epochs_now = float(epochs_now) * float(self.epochs_growth)
            self._train_substitute(int(max(1, round(epochs_now))), device)
            self.state.attack_state["round"] = int(r + 1)
            if self.substitute is not None:
                self._evaluate_current_substitute(self.substitute, device)

    def observe(self, query_batch: QueryBatch, oracle_output: OracleOutput, state: BenchmarkState) -> None:
        indices = [int(i) for i in query_batch.meta.get("indices", [])]
        if len(indices) == 0:
            return
        if oracle_output.kind != "hard_top1":
            raise ValueError("marich requires hard_top1")
        state.attack_state.setdefault("query_data_x", []).append(query_batch.x.detach().cpu())
        state.attack_state.setdefault("query_data_y", []).append(oracle_output.y.detach().cpu().long())
