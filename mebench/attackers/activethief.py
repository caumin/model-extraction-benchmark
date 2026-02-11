from __future__ import annotations

import logging
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.state import BenchmarkState
from mebench.core.types import QueryBatch, OracleOutput
from mebench.models.substitute_factory import create_substitute
from mebench.training import SubstituteTrainer, TrainRequest
from mebench.data.loaders import create_dataloader
from mebench.utils.dataloader import pool_loader_kwargs


class ActiveThief(AttackRunner):
    """ActiveThief attack (Pal et al., 2020)."""

    def __init__(self, config: dict, state: BenchmarkState) -> None:
        super().__init__(config, state)
        self.dataloader = None
        self.pool_dataset = None
        self._initialize_state(state)

        self.device = str(config.get("run", {}).get("device", state.metadata.get("device", "cpu")))
        self.pool_num_workers = max(0, int(config.get("pool_num_workers", config.get("num_workers", 4))))
        self.train_num_workers = max(0, int(config.get("train_num_workers", config.get("num_workers", 0))))

    def _initialize_state(self, state: BenchmarkState) -> None:
        state.attack_state.setdefault("queried_indices", [])
        state.attack_state.setdefault("unqueried_indices", [])
        state.attack_state.setdefault("query_data_x", [])
        state.attack_state.setdefault("query_data_y", [])
        state.attack_state.setdefault("substitute", None)

    def _ensure_pool_dataset(self, state: BenchmarkState) -> None:
        if self.pool_dataset is None:
            self.dataloader = create_dataloader(
                state.metadata.get("dataset_config", {}),
                batch_size=1,
                shuffle=False,
            )
            self.pool_dataset = self.dataloader.dataset

        if not state.attack_state["unqueried_indices"] and self.pool_dataset is not None:
            dataset_config = state.metadata.get("dataset_config", {})
            seed_size = dataset_config.get("seed_size") # Used as max pool size cap if set
             
            # Usually pool is the whole surrogate dataset
            # But we respect seed_size if it implies total budget or pool cap?
            # Actually ActiveThief usually uses the whole pool.
            # We'll use the whole dataset length.
            pool_size = len(self.pool_dataset)
            
            # If explicit max_samples is set in config
            if "surrogate_max_samples" in dataset_config:
                 pool_size = min(pool_size, int(dataset_config["surrogate_max_samples"]))
            
            state.attack_state["unqueried_indices"] = list(range(pool_size))

    def run(self, ctx: BenchmarkContext) -> None:
        self.victim = ctx.oracle.model
        self._ensure_pool_dataset(self.state)
        
        # 1. Initial Seed (Random)
        initial_seed_size = int(self.config.get("initial_seed_size", 100))
        # If we haven't queried anything yet
        if len(self.state.attack_state["queried_indices"]) == 0:
            seed_k = min(initial_seed_size, ctx.budget_remaining)
            if seed_k > 0:
                self.logger.info(f"Querying initial seed of size {seed_k}")
                self._query_batch(seed_k, self.state, ctx=ctx, strategy="random")
        
        # 2. Active Loop
        step_size = int(self.config.get("step_size", self._default_step_size(ctx)))
        strategy = self.config.get("strategy", "uncertainty")
        
        pbar = self._create_progress_bar(ctx.budget_remaining, f"[{self.__class__.__name__}] Extracting")
        
        while ctx.budget_remaining > 0:
            # Train substitute on current labeled data
            substitute = self._train_substitute(self.state)
            self.state.attack_state["substitute"] = substitute
            
            # Evaluate current substitute
            self._evaluate_current_substitute(substitute, self.device)
            
            # Select next batch
            k = min(step_size, ctx.budget_remaining)
            if k <= 0:
                break
            
            # Check if pool is empty
            if not self.state.attack_state["unqueried_indices"]:
                self.logger.warning("Unlabeled pool exhausted. Stopping attack.")
                break

            self.logger.info(f"Selecting {k} samples using strategy: {strategy}")
            queried_count = self._query_batch(k, self.state, ctx=ctx, strategy=strategy, substitute=substitute)
            pbar.update(queried_count)
            
        pbar.close()
        
        # Final training
        final_substitute = self._train_substitute(self.state)
        self.state.attack_state["substitute"] = final_substitute

    def _query_batch(self, k: int, state: BenchmarkState, ctx: BenchmarkContext, strategy: str, substitute: nn.Module = None) -> int:
        indices = self._select_indices(k, state, strategy, substitute)
        
        if not indices:
            return 0
            
        x_list = [self.pool_dataset[i][0] for i in indices]
        x_batch = torch.stack(x_list)
        
        query_batch = QueryBatch(x=x_batch, meta={"strategy": strategy, "synthetic": False})
        oracle_output = ctx.query(query_batch.x, meta=query_batch.meta)
        
        # Update state
        state.attack_state["query_data_x"].append(query_batch.x.detach().cpu())
        state.attack_state["query_data_y"].append(oracle_output.y.detach().cpu())
        
        for idx in indices:
            if idx in state.attack_state["unqueried_indices"]:
                state.attack_state["unqueried_indices"].remove(idx)
            state.attack_state["queried_indices"].append(int(idx))
            
        return len(indices)

    def _query_and_update(self, indices: list[int], ctx: BenchmarkContext) -> None:
        x_list = [self.pool_dataset[i][0] for i in indices]
        x_batch = torch.stack(x_list)
        
        query_batch = QueryBatch(x=x_batch, meta={"synthetic": False})
        oracle_output = ctx.query(query_batch.x, meta=query_batch.meta)
        
        # Update state
        self.state.attack_state["query_data_x"].append(query_batch.x.detach().cpu())
        self.state.attack_state["query_data_y"].append(oracle_output.y.detach().cpu())
        
        for idx in indices:
            if idx in self.state.attack_state["unqueried_indices"]:
                self.state.attack_state["unqueried_indices"].remove(idx)
            self.state.attack_state["queried_indices"].append(int(idx))

    def _select_indices(
        self, k: int, state: BenchmarkState, strategy: str, substitute: nn.Module = None
    ) -> list[int]:
        available = state.attack_state["unqueried_indices"]
        if not available:
            return []
            
        n_take = min(int(k), len(available))
        
        if strategy == "random" or substitute is None:
            return np.random.choice(available, n_take, replace=False).tolist()
            
        # Strategy-based selection
        # Create loader for pool
        pool_subset = Subset(self.pool_dataset, available)
        # We need a mapping from subset index to original index
        # subset[i] maps to available[i]
        
        pool_loader = DataLoader(
            pool_subset, 
            batch_size=256, 
            shuffle=False, 
            **pool_loader_kwargs(self.device, {"num_workers": self.pool_num_workers}),
        )
        
        scores = []
        substitute.eval()
        output_mode = self.config.get("output_mode", "soft_prob")
        
        with torch.no_grad():
            for x, _ in pool_loader:
                x = x.to(self.device)
                outputs = substitute(x)
                
                if strategy == "uncertainty":
                    # Entropy: -sum(p * log(p))
                    if output_mode == "soft_prob":
                        probs = torch.softmax(outputs, dim=1)
                    else:
                        # If hard labels, entropy is 0?
                        # Fallback to random or some heuristic?
                        # Assume outputs are logits even if trained with hard labels?
                        probs = torch.softmax(outputs, dim=1)
                        
                    log_probs = torch.log_softmax(outputs, dim=1)
                    entropy = -torch.sum(probs * log_probs, dim=1)
                    scores.extend(entropy.cpu().numpy())
                else:
                    # Fallback to random
                    scores.extend(np.random.rand(x.size(0)))
                    
        # Select top-k indices (highest entropy)
        # scores correspond to available[0], available[1], ...
        top_k_indices = np.argsort(scores)[-n_take:]
        selected_indices = [available[i] for i in top_k_indices]
        
        return selected_indices

    def _train_substitute(self, state: BenchmarkState) -> nn.Module:
        """Train substitute model with 80/20 validation split."""
        query_x = state.attack_state.get("query_data_x", [])
        query_y = state.attack_state.get("query_data_y", [])
        
        if len(query_x) == 0:
             # Should not happen if initial seed is queried
             return None

        x_all = torch.cat(query_x, dim=0)
        y_all = torch.cat(query_y, dim=0)
        
        # Ensure sufficient data
        if x_all.size(0) < 10:
             return self._train_substitute_simple(state, x_all, y_all)

        # 80/20 Split
        total_size = x_all.size(0)
        val_size = max(1, int(0.2 * total_size))
        train_size = total_size - val_size
        
        full_dataset = TensorDataset(x_all, y_all)
        train_subset, val_subset = torch.utils.data.random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create Substitute
        sub_config = state.metadata.get("substitute_config", {}) or self.config.get("substitute", {})
        num_classes = int(state.metadata.get("num_classes", 10))
        input_channels = int(state.metadata.get("input_shape", (3, 32, 32))[0])
        
        substitute = create_substitute(
            arch=sub_config.get("arch", "resnet18"),
            num_classes=num_classes,
            input_channels=input_channels,
            dropout_prob=float(sub_config.get("dropout_prob", 0.0))
        ).to(self.device)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=int(sub_config.get("batch_size", 128)),
            shuffle=True,
            **pool_loader_kwargs(
                self.device,
                {
                    "num_workers": int(
                        sub_config.get(
                            "train_num_workers",
                            sub_config.get("num_workers", self.train_num_workers),
                        )
                    )
                },
            ),
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=int(sub_config.get("batch_size", 128)),
            shuffle=False,
            **pool_loader_kwargs(
                self.device,
                {
                    "num_workers": int(
                        sub_config.get(
                            "val_num_workers",
                            sub_config.get("num_workers", self.train_num_workers),
                        )
                    )
                },
            ),
        )
        
        output_mode = str(self.config.get("output_mode", "soft_prob"))

        def macro_f1_score(preds: torch.Tensor, targets: torch.Tensor, num_classes_local: int) -> float:
            preds = preds.view(-1)
            targets = targets.view(-1)
            f1_sum = 0.0
            for cls_idx in range(num_classes_local):
                pred_pos = preds == cls_idx
                true_pos = targets == cls_idx
                tp = int((pred_pos & true_pos).sum().item())
                fp = int((pred_pos & (~true_pos)).sum().item())
                fn = int(((~pred_pos) & true_pos).sum().item())
                denom = (2 * tp) + fp + fn
                f1_sum += (2.0 * tp / denom) if denom > 0 else 0.0
            return f1_sum / float(num_classes_local)
        
        def eval_fn(model, loader):
            model.eval()
            if output_mode == "soft_prob":
                total_loss = 0.0
                count = 0
                loss_func = nn.KLDivLoss(reduction="batchmean")
                with torch.no_grad():
                    for x, y in loader:
                        x, y = x.to(self.device), y.to(self.device)
                        outputs = model(x)
                        y = y.clamp_min(1e-10)
                        y = y / y.sum(dim=1, keepdim=True)
                        loss = loss_func(torch.log_softmax(outputs, dim=1), y)
                        total_loss += loss.item() * x.size(0)
                        count += x.size(0)
                return total_loss / count if count > 0 else float("inf")

            all_preds = []
            all_targets = []
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(self.device)
                    outputs = model(x)
                    preds = torch.argmax(outputs, dim=1).cpu()
                    all_preds.append(preds)
                    all_targets.append(y.long().cpu())

            if not all_preds:
                return 0.0
            preds_cat = torch.cat(all_preds, dim=0)
            targets_cat = torch.cat(all_targets, dim=0)
            return macro_f1_score(preds_cat, targets_cat, num_classes)

        def loss_fn(outputs, targets):
            if output_mode == "soft_prob":
                 targets = targets.clamp_min(1e-10)
                 targets = targets / targets.sum(dim=1, keepdim=True)
                 return nn.KLDivLoss(reduction="batchmean")(torch.log_softmax(outputs, dim=1), targets)
            return nn.CrossEntropyLoss()(outputs, targets.long())

        batch_size = int(sub_config.get("batch_size", 128))
        steps_per_epoch = max(1, int(math.ceil(train_size / batch_size)))
        max_epochs = int(sub_config.get("max_epochs", 1000))
        patience_epochs = int(sub_config.get("patience", 100))
        max_steps = max_epochs * steps_per_epoch
        validate_every = steps_per_epoch
        patience_steps = patience_epochs * steps_per_epoch
        
        trainer = SubstituteTrainer(dict(sub_config), device=self.device)
        request = TrainRequest(
            model=substitute,
            train_loader=train_loader,
            val_loader=val_loader,
            eval_fn=eval_fn,
            loss_fn=loss_fn,
            max_steps=max_steps,
            early_stop_mode="min" if output_mode == "soft_prob" else "max",
            load_best=True,
            patience=patience_steps,
            validate_every=validate_every,
        )
        trainer.train(request)
        return substitute

    def _train_substitute_simple(self, state, x_all, y_all):
        """Fallback for very small data."""
        # Simple training logic similar to RandomBaseline
        sub_config = state.metadata.get("substitute_config", {}) or self.config.get("substitute", {})
        num_classes = int(state.metadata.get("num_classes", 10))
        input_channels = int(state.metadata.get("input_shape", (3, 32, 32))[0])
        
        substitute = create_substitute(
            arch=sub_config.get("arch", "resnet18"),
            num_classes=num_classes,
            input_channels=input_channels,
        ).to(self.device)
        
        loader = DataLoader(
             TensorDataset(x_all, y_all),
             batch_size=32,
             shuffle=True,
             **pool_loader_kwargs(
                 self.device,
                 {
                     "num_workers": int(
                         sub_config.get(
                             "train_num_workers",
                             sub_config.get("num_workers", self.train_num_workers),
                         )
                     )
                 },
             ),
        )
        
        output_mode = str(self.config.get("output_mode", "soft_prob"))
        def loss_fn(outputs, targets):
            if output_mode == "soft_prob":
                 targets = targets.clamp_min(1e-10)
                 targets = targets / targets.sum(dim=1, keepdim=True)
                 return nn.KLDivLoss(reduction="batchmean")(torch.log_softmax(outputs, dim=1), targets)
            return nn.CrossEntropyLoss()(outputs, targets.long())

        trainer = SubstituteTrainer(dict(sub_config), device=self.device)
        request = TrainRequest(
            model=substitute,
            train_loader=loader,
            loss_fn=loss_fn,
            max_steps=200,
        )
        trainer.train(request)
        return substitute
