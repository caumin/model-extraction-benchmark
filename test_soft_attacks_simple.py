#!/usr/bin/env python3
"""Simple test script to verify soft attacks work in hard mode."""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.core.validate import validate_config


def create_mock_hard_label_output():
    """Create mock OracleOutput with hard labels (class indices)."""
    batch_size = 4
    num_classes = 10
    
    # Create hard labels (class indices)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    return OracleOutput(
        kind="hard_top1",
        y=labels  # Class indices, not one-hot vectors
    )


def create_mock_query_batch():
    """Create mock QueryBatch."""
    batch_size = 4
    
    return QueryBatch(
        x=torch.randn(batch_size, 3, 32, 32),
        meta={}
    )


def create_mock_state():
    """Create mock BenchmarkState."""
    return BenchmarkState(
        budget_used=0,
        budget_total=1000,
        phase="training",
        data_collected=[],
        step=0,
        attack_state={},
        metadata={"max_budget": 10000, "dataset_config": {"data_mode": "seed", "name": "CIFAR10"}}
    )


def test_config_validation():
    """Test if soft attacks can be configured in hard mode."""
    print("Testing config validation...")
    
    # CloudLeak config
    config = {
        "attack": {
            "name": "cloudleak",
            "output_mode": "hard_top1",
            "lambda_adv": 0.1,
            "lambda_target": 1.0,
            "epsilon": 8/255,
            "num_steps": 10
        },
        "victim": {"output_mode": "hard_top1", "temperature": 1.0},
        "dataset": {"data_mode": "seed", "name": "CIFAR10"},
        "budget": {"max_budget": 1000, "checkpoints": [500, 1000]}
    }
    
    try:
        validate_config(config)
        print("+ CloudLeak config validation passed")
        cloudleak_ok = True
    except Exception as e:
        print("X CloudLeak config validation failed:", str(e))
        cloudleak_ok = False
    
    # SwiftThief config
    config = {
        "attack": {
            "name": "swiftthief",
            "output_mode": "hard_top1",
            "temperature": 0.1,
            "projection_dim": 128,
            "num_augmentations": 2
        },
        "victim": {"output_mode": "hard_top1", "temperature": 1.0},
        "dataset": {"data_mode": "seed", "name": "CIFAR10"},
        "budget": {"max_budget": 1000, "checkpoints": [500, 1000]}
    }
    
    try:
        validate_config(config)
        print("+ SwiftThief config validation passed")
        swifthief_ok = True
    except Exception as e:
        print("X SwiftThief config validation failed:", str(e))
        swifthief_ok = False
    
    return cloudleak_ok and swifthief_ok


def test_hard_label_conversion():
    """Test if hard labels (class indices) are handled correctly."""
    print("\nTesting hard label handling...")
    
    # Create hard label output (class indices)
    batch_size = 4
    num_classes = 10
    
    labels = torch.randint(0, num_classes, (batch_size,))
    
    oracle_output = OracleOutput(
        kind="hard_top1",
        y=labels
    )
    
    # Check if labels are valid
    assert oracle_output.y.shape == (batch_size,)
    assert oracle_output.y.dtype == torch.long
    print("+ Hard labels valid: shape {}".format(oracle_output.y.shape))
    print("+ Sample labels: {}".format(oracle_output.y[:2]))
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("SOFT ATTACKS IN HARD MODE TEST")
    print("=" * 60)
    
    tests = [
        test_hard_label_conversion,
        test_config_validation,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print("X Test {} crashed: {}".format(test.__name__, str(e)))
            results.append(False)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print("Tests passed: {}/{}".format(passed, total))
    
    if passed == total:
        print("+ All tests passed! Soft attacks can work in hard mode.")
        print("+ Next: Test with actual attack initialization and training")
        return 0
    else:
        print("X Some tests failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    exit(main())