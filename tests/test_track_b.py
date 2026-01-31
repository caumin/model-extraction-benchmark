"""Test Track B evaluation with DFME."""

import pytest
import torch
from pathlib import Path
import tempfile
from mebench.core.state import BenchmarkState
from mebench.core.types import QueryBatch, OracleOutput
from mebench.attackers.dfme import DFME
from mebench.eval.evaluator import Evaluator
from mebench.core.query_storage import QueryStorage
from mebench.models.substitute_factory import create_substitute


def test_dfme_track_b_evaluation():
    """Test Track B evaluation loop with DFME."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Config for small test
        config = {
            "run": {"name": "test", "seeds": [0], "device": "cpu"},
            "victim": {
                "victim_id": "test_victim",
                "checkpoint_ref": "/path/to/ckpt.pt",  # Placeholder
                "arch": "lenet",
                "channels": 3,
                "num_classes": 10,
                "input_size": [32, 32],
                "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
                "output_mode": "soft_prob",
                "temperature": 1.0,
                "output_modes_supported": ["soft_prob", "hard_top1"]
            },
            "dataset": {
                "name": "CIFAR10",
                "data_mode": "data_free",
                "seed_size": 100
            },
            "substitute": {
                "arch": "resnet18",
                "init_seed": 1234,
                "trackA": {
                    "batch_size": 32,
                    "steps_coeff_c": 20
                },
                "optimizer": {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4},
                "patience": 5,
                "loss": {"soft": "kl", "hard": "ce"}
            },
            "attack": {
                "name": "dfme",
                "batch_size": 16,  # Small batch
                "student_lr": 0.01,  # Reduced LR
                "generator_lr": 0.005,
                "n_g_steps": 1,
                "n_s_steps": 1,
                "grad_approx_m": 1,
                "grad_approx_epsilon": 1e-3,
                "noise_dim": 100
            },
            "budget": {
                "max_budget": 500,  # Very small budget
                "checkpoints": [100, 500]
            },
            "cache": {
                "enabled": False  # Disable cache for quick test
            }
        }

        state = BenchmarkState()
        state.metadata = {"device": "cpu", "input_shape": (3, 32, 32)}

        # Initialize DFME attack
        attack = DFME(config["attack"], state)

        # Run queries through DFME (simulating 500 queries)
        print("Simulating DFME queries...")
        k = 16
        num_rounds = 500 // k

        for round_idx in range(num_rounds):
            # Propose
            query_batch = attack._select_query_batch(k, state)

            # Simulate oracle response (placeholder victim)
            x = query_batch.x
            with torch.no_grad():
                # Simple placeholder victim prediction
                victim_logits = torch.randn(x.size(0), 10)
                victim_probs = torch.softmax(victim_logits, dim=1)

            oracle_output = OracleOutput(kind="soft_prob", y=victim_probs)

            # Observe (trains DFME internally)
            attack._handle_oracle_output(query_batch, oracle_output, state)

            if (round_idx + 1) * k in [100, 500]:
                print(f"Checkpoint reached at {state.attack_state['step']} queries")
                # Track B should now have a trained substitute
                sub = state.attack_state.get("substitute")
                assert sub is not None, f"Substitute not available at checkpoint {state.attack_state['step']}"
                print(f"Substitute available at checkpoint: {sub is not None}")

        # Final check
        final_substitute = state.attack_state.get("substitute")
        assert final_substitute is not None, "Final substitute should be available"
        print("Track B test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
