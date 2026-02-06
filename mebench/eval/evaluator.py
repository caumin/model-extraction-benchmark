"""Track A evaluator: trains substitute from scratch at each checkpoint."""

from typing import Dict, Any, List
import torch
import torch.nn as nn
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
        self.device = state.metadata["device"]

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

    def _train_track_a(
        self, model: nn.Module, train_loader: DataLoader, checkpoint_budget: int
    ) -> None:
        """Standard training protocol for Track A.

        Args:
            model: Substitute model to train
            train_loader: Collected (x, y_oracle) data
            checkpoint_budget: Budget at this checkpoint
        """
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

        val_loader = train_loader
        trainer_config = dict(self.config["substitute"])
        if "grad_clip" not in trainer_config:
            trainer_config["grad_clip"] = 1.0

        trainer = SubstituteTrainer(trainer_config, device=self.device)
        request = TrainRequest(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            val_loader=val_loader,
            eval_fn=lambda m, v: self._compute_f1(m, v, output_mode),
            early_stop_mode="max",
            max_steps=num_steps,
            validate_every=100,
            load_best=True,
        )
        result = trainer.train(request)

        if result.best_value is not None:
            print(f"Track A training complete. Best F1: {result.best_value:.4f}")
        else:
            print("Track A training complete. No validation improvement.")

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
