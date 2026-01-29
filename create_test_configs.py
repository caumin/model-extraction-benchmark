#!/usr/bin/env python3
"""Create configs to test soft attacks in both soft and hard modes."""

import yaml
import os

def create_configs():
    """Create test configurations for soft attacks in both modes."""
    
    base_config = {
        "run": {
            "name": "soft_attacks_test",
            "seeds": [0],
        },
        "victim": {
            "model": "resnet18",
            "dataset": "CIFAR10",
            "pretrained": True,
            "output_mode": "soft_prob",  # Will be overridden
            "temperature": 1.0
        },
        "attack": {
            # Will be set for each attack
        },
        "dataset": {
            "data_mode": "seed",
            "name": "CIFAR10",
            "train_split": 0.8,
            "num_workers": 0
        },
        "substitute": {
            "model": "resnet18",
            "arch": "resnet18",
            "optimizer": {"name": "adam", "lr": 0.01},
            "batch_size": 128,
            "epochs": 50,
            "trackA": {"steps_coeff_c": 0.2}
        },
        "budget": {
            "max_budget": 2000,
            "checkpoints": [1000, 2000]
        },
        "metrics": ["acc_gt", "agreement", "kl_mean"],
        "device": "cpu"
    }
    
    # Attacks to test
    attacks = {
        "cloudleak": {
            "lambda_adv": 0.1,
            "lambda_target": 1.0,
            "epsilon": 8/255,
            "num_steps": 5,  # Reduced for testing
            "patience": 10
        },
        "swiftthief": {
            "temperature": 0.1,
            "projection_dim": 128,
            "num_augmentations": 2,
            "patience": 10
        }
    }
    
    modes = ["soft_prob", "hard_top1"]
    
    configs_created = []
    
    for attack_name, attack_params in attacks.items():
        for mode in modes:
            config = base_config.copy()
            
            # Set victim config
            config["victim"] = {
                "victim_id": f"cifar10_resnet18_{mode}",
                "model": "resnet18",
                "dataset": "CIFAR10",
                "pretrained": True,
                "output_mode": mode,  # Will match attack mode
                "temperature": 1.0,
                "channels": 3,
                "input_size": [32, 32],
                "num_classes": 10
            }
            
            # Set attack-specific config
            config["attack"] = attack_params.copy()
            config["attack"]["name"] = attack_name
            config["attack"]["output_mode"] = mode
            
            # Create filename
            filename = f"test_{attack_name}_{mode}.yaml"
            filepath = os.path.join("configs/debug", filename)
            
            # Ensure directory exists
            os.makedirs("configs/debug", exist_ok=True)
            
            # Write config
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            configs_created.append(filepath)
            print(f"Created: {filepath}")
    
    return configs_created


if __name__ == "__main__":
    configs = create_configs()
    print(f"\nCreated {len(configs)} test configurations:")
    for config in configs:
        print(f"  - {config}")
    
    print("\nUsage:")
    print("python -m mebench run --config configs/debug/test_cloudleak_soft_prob.yaml")
    print("python -m mebench run --config configs/debug/test_cloudleak_hard_top1.yaml")
    print("python -m mebench run --config configs/debug/test_swifthief_soft_prob.yaml")
    print("python -m mebench run --config configs/debug/test_swifthief_hard_top1.yaml")