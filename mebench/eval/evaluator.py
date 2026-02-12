"""Track A evaluator: trains substitute from scratch at each checkpoint."""

from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from mebench.models.substitute_factory import create_substitute
from mebench.eval.metrics import evaluate_substitute
from mebench.training import SubstituteTrainer, TrainRequest


class Evaluator:
    """Evaluates benchmark state using Track A protocol."""

    def __init__(self, config: dict, state: Any, query_storage: Any):
        """Initialize evaluator.

        Args:
            config: Full experiment configuration
            state: BenchmarkState object
            query_storage: QueryStorage object
        """
        self.config = config
        self.state = state
        self.query_storage = query_storage
        run_device = config.get("run", {}).get("device")
        self.device = str(state.metadata.get("device") or run_device or "cpu")

    def _evaluate_track_b(
        self,
        victim: nn.Module,
        test_loader: DataLoader,
        checkpoint_budget: int,
    ) -> Dict[str, float]:
        """Evaluate the current Track B substitute (best-effort helper).

        Track B attacks may fail to produce a substitute in some edge cases.
        For reporting stability, return zero metrics when substitute is missing.

        Args:
            victim: Victim model
            test_loader: Test set loader
            checkpoint_budget: Query budget at which this evaluation is recorded

        Returns:
            Metrics dict (acc_gt, agreement, kl_mean, l1_mean)
        """
        _ = checkpoint_budget
        substitute = self.state.attack_state.get("substitute")
        if substitute is None:
            return {"acc_gt": 0.0, "agreement": 0.0, "kl_mean": 0.0, "l1_mean": 0.0}

        output_mode = self.config["victim"]["output_mode"]
        temperature = float(self.config["victim"].get("temperature", 1.0))
        metrics = evaluate_substitute(
            substitute=substitute,
            victim=victim,
            test_loader=test_loader,
            device=self.device,
            output_mode=output_mode,
            temperature=temperature,
        )

        # Ensure numeric defaults even if a metric is absent in a mode.
        metrics.setdefault("kl_mean", 0.0)
        metrics.setdefault("l1_mean", 0.0)
        return metrics

    def evaluate(
        self,
        victim: nn.Module,
        test_loader: DataLoader,
        checkpoint_budget: int,
    ) -> Dict[str, Dict[str, float]]:
        """Perform Track A evaluation.

        Args:
            victim: The victim model
            test_loader: DataLoader for test set
            checkpoint_budget: Current query budget

        Returns:
            Dictionary with results (track_a)
        """
        # Load query data
        query_loader = self.query_storage.get_dataloader(batch_size=128)

        # 1. Setup substitute
        sub_config = self.config["substitute"]
        num_classes = int(self.config["victim"]["num_classes"])
        input_channels = int(self.config["victim"]["channels"])

        # Reset RNG for reproducibility
        torch.manual_seed(sub_config.get("init_seed", 42))

        substitute = create_substitute(
            arch=sub_config["arch"],
            num_classes=num_classes,
            input_channels=input_channels,
        ).to(self.device)

        # 2. Train from scratch
        self._train_track_a(substitute, query_loader, checkpoint_budget)

        # 3. Evaluate
        output_mode = self.config["victim"]["output_mode"]
        temperature = self.config["victim"]["temperature"]

        metrics = evaluate_substitute(
            substitute=substitute,
            victim=victim,
            test_loader=test_loader,
            device=self.device,
            output_mode=output_mode,
            temperature=temperature,
        )

        return {"track_a": metrics}

    def _train_track_a(self, model: nn.Module, *args: Any) -> None:
        """Standard training protocol for Track A.

        Supports two call patterns:
        1) Production: (model, train_loader, checkpoint_budget)
        2) Legacy/test: (model, optimizer, num_steps, batch_size)
        """
        # Legacy/test path
        if len(args) == 3 and isinstance(args[0], optim.Optimizer):
            optimizer = args[0]
            num_steps = int(args[1])
            batch_size = int(args[2])
            train_loader = self.query_storage.get_dataloader(
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )
            checkpoint_budget = max(len(self.query_storage), 1)
            self._train_track_a_legacy(model, optimizer, train_loader, num_steps)
            return

        if len(args) != 2:
            raise TypeError(
                "_train_track_a expects (train_loader, checkpoint_budget) or (optimizer, num_steps, batch_size)"
            )

        train_loader = args[0]
        checkpoint_budget = int(args[1])

        # Get training steps: S(B) = ceil(0.2 × B)
        steps_coeff = self.config["substitute"]["trackA"]["steps_coeff_c"]
        num_steps = int(steps_coeff * checkpoint_budget + 0.9999)

        output_mode = self.config["victim"]["output_mode"]

        def loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            if output_mode == "soft_prob":
                targets = targets.to(self.device)
                targets = torch.clamp(targets, min=1e-10)
                targets = targets / targets.sum(dim=1, keepdim=True)
                log_outputs = torch.log_softmax(outputs, dim=1)
                return nn.KLDivLoss(reduction="batchmean")(log_outputs, targets)

            targets = targets.long().to(self.device)
            return nn.CrossEntropyLoss()(outputs, targets)

        trainer_config = dict(self.config["substitute"])
        if "grad_clip" not in trainer_config:
            trainer_config["grad_clip"] = 1.0

        trainer = SubstituteTrainer(trainer_config, device=self.device)
        request = TrainRequest(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            max_steps=num_steps,
            # Track A contract: run a fixed number of steps S(B)=ceil(0.2*B).
            # Do not early-stop or select best checkpoint, since that would
            # introduce an extra training-recipe degree of freedom.
            load_best=False,
        )
        result = trainer.train(request)

        if result.best_value is not None:
            print(f"Track A training complete. Best F1: {result.best_value:.4f}")
        else:
            print("Track A training complete. No validation improvement.")

    def _train_track_a_legacy(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        num_steps: int,
    ) -> None:
        """Minimal deterministic loop used by unit tests."""
        output_mode = self.config["victim"]["output_mode"]
        model.train()
        step = 0
        it = iter(train_loader)
        while step < int(num_steps):
            try:
                x_batch, y_batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                x_batch, y_batch = next(it)

            x_batch = x_batch.to(self.device)
            outputs = model(x_batch)

            if output_mode == "soft_prob":
                targets = y_batch.to(self.device)
                targets = torch.clamp(targets, min=1e-10)
                targets = targets / targets.sum(dim=1, keepdim=True)
                log_outputs = torch.log_softmax(outputs, dim=1)
                loss = nn.KLDivLoss(reduction="batchmean")(log_outputs, targets)
            else:
                loss = nn.CrossEntropyLoss()(outputs, y_batch.long().to(self.device))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1

    def _compute_f1(self, model: nn.Module, val_loader: DataLoader, output_mode: str) -> float:
        """Compute F1 score on validation set."""
        from sklearn.metrics import f1_score
        import numpy as np

        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(self.device)
                # Contract: Inputs in [0, 1].
                outputs = model(x_batch)

                # Get predictions
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)

                # Get true labels
                if output_mode == "soft_prob":
                    targets = torch.argmax(y_batch, dim=1).cpu().numpy()
                else:
                    targets = y_batch.cpu().numpy()
                all_targets.extend(targets)

        return f1_score(all_targets, all_preds, average="macro")
