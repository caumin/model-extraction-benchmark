"""Oracle wrapper for victim model inference."""

import torch
import torch.nn as nn
from mebench.core.types import OracleOutput
from mebench.core.state import BenchmarkState


class Oracle:
    """Wrapper for victim model that enforces budget and output constraints."""

    def __init__(self, model: nn.Module, config: dict, state: BenchmarkState):
        """Initialize oracle.

        Args:
            model: The victim model (pre-trained)
            config: Oracle/Victim configuration
            state: BenchmarkState object for budget tracking
        """
        self.model = model
        self.model.eval()
        self.config = config
        self.state = state

        self.output_mode = config.get("output_mode", "soft_prob")
        self.temperature = float(config.get("temperature", 1.0))
        # Default behavior historically returned oracle outputs on CPU. Keep that as
        # the default for backward compatibility and to avoid accidentally pinning
        # large outputs (e.g., soft_prob) on GPU.
        self.return_outputs_on_cpu = bool(config.get("return_outputs_on_cpu", True))
        self.input_shape = tuple(config.get("input_size", (3, 32, 32)))
        if len(self.input_shape) == 2:
             # Add channel if missing
             self.input_shape = (config.get("channels", 1), *self.input_shape)

    @property
    def query_count(self) -> int:
        return int(self.state.query_count)

    @property
    def budget_remaining(self) -> int:
        return int(self.state.budget_remaining)

    @torch.no_grad()
    def query(self, x_batch: torch.Tensor) -> OracleOutput:
        """Query victim model with a batch of images.

        Args:
            x_batch: Input tensor of shape (N, C, H, W) or (N, D).
                Assumed to be in [0, 1] scale.

        Returns:
            OracleOutput container
        """
        # Ensure inputs are on the victim device (avoid redundant .to when already there).
        device = next(self.model.parameters()).device
        if x_batch.device != device:
            x_batch = x_batch.to(device)

        batch_size = x_batch.size(0)

        # Update global state
        self.state.query_count += batch_size
        self.state.budget_remaining -= batch_size

        # Normalize inputs to match victim's channels/size: reshape if needed
        # Contract: Assume x_batch is in [0, 1]. No additional normalization.
        x_reshaped = x_batch.view(x_batch.size(0), *self.input_shape)

        # Forward pass
        logits = self.model(x_reshaped)

        # Apply temperature
        logits = logits / self.temperature

        if self.output_mode == "soft_prob":
            # Soft softmax probabilities
            probs = torch.softmax(logits, dim=1)
            y = probs
            kind = "soft_prob"
        elif self.output_mode == "hard_top1":
            # Hard class label (top-1)
            labels = torch.argmax(logits, dim=1)
            y = labels
            kind = "hard_top1"
        else:
            raise ValueError(f"Unsupported output mode: {self.output_mode}")

        if self.return_outputs_on_cpu:
            y = y.cpu()
        return OracleOutput(kind=kind, y=y)
