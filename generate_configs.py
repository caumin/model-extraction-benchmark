import yaml
import os
from pathlib import Path

def generate_configs():
    # Experimental Setup
    configs = [
        {
            "id": "SET-A1",
            "victim_dataset": "MNIST",
            "victim_arch": "lenet_mnist",
            "surrogate": "EMNIST",
            "substitute_arch": "lenet_mnist",
            "channels": 1,
            "size": 28
        },
        {
            "id": "SET-A2",
            "victim_dataset": "MNIST",
            "victim_arch": "lenet_mnist",
            "surrogate": "FashionMNIST",
            "substitute_arch": "lenet_mnist",
            "channels": 1,
            "size": 28
        },
        {
            "id": "SET-B1",
            "victim_dataset": "CIFAR10",
            "victim_arch": "resnet18",
            "surrogate": "SVHN",
            "substitute_arch": "resnet18",
            "channels": 3,
            "size": 32
        },
        {
            "id": "SET-B2",
            "victim_dataset": "CIFAR10",
            "victim_arch": "resnet18",
            "surrogate": "GTSRB",
            "substitute_arch": "resnet18",
            "channels": 3,
            "size": 32
        }
    ]

    attacks = [
        "random", "activethief", "knockoff_nets", "copycatcnn",
        "dfme", "maze", "dfms", "game", "es", "blackbox_ripper",
        "cloudleak", "swiftthief", "inversenet"
    ]

    # Add hard-labeled versions for Dissecor
    attacks.append("blackbox_dissector")
    
    # Add ActiveThief variants
    attacks.append("activethief_uncertainty")
    attacks.append("activethief_dfal")

    # Define attack categories for strategy-specific budget handling
    # Category 1: AL/Phase-based (Need separate runs per budget for fair round distribution)
    AL_ATTACKS = {"activethief", "swiftthief", "cloudleak", "inversenet"}
    
    # Category 2: Batch-based (Single long run with checkpoints is fair)
    BATCH_POOL_ATTACKS = {"random", "knockoff_nets", "copycatcnn", "blackbox_dissector"}
    BATCH_FREE_ATTACKS = {"dfme", "maze", "dfms", "game", "es", "blackbox_ripper"}

    seeds = [0, 1, 2]
    budget_checkpoints = [1000, 10000, 20000]
    data_free_max = 100000
    data_free_checkpoints = [1000, 10000, 20000, 50000, 100000]

    os.makedirs("configs/matrix", exist_ok=True)

    for setup in configs:
        for attack in attacks:
            # Base attack name and strategy handling
            attack_name = attack
            attack_strategy = None
            if attack.startswith("activethief_"):
                attack_strategy = attack.split("_")[1]
                attack_name = "activethief"
            elif attack == "activethief":
                attack_strategy = "dfal_k_center"

            # Clean victim ID for filenames (e.g., mnist_lenet instead of mnist_lenet_mnist)
            victim_arch_clean = setup['victim_arch']
            if victim_arch_clean == 'lenet_mnist':
                victim_arch_clean = 'lenet'
            
            victim_id = f"{setup['victim_dataset'].lower()}_{victim_arch_clean}"
            
            output_mode = "hard_top1" if attack == "blackbox_dissector" else "soft_prob"
            data_mode = "data_free" if attack_name in BATCH_FREE_ATTACKS else "surrogate"

            # Determine execution plan based on category
            if attack_name in AL_ATTACKS:
                # Generate SEPARATE configs for each budget
                budgets_to_gen = [(b, [b]) for b in budget_checkpoints]
            elif attack_name in BATCH_POOL_ATTACKS:
                # Single run up to 20k
                budgets_to_gen = [(20000, budget_checkpoints)]
            else: # BATCH_FREE_ATTACKS
                # Single run up to 100k
                budgets_to_gen = [(data_free_max, data_free_checkpoints)]

            for max_b, checkpoints in budgets_to_gen:
                for seed in seeds:
                    budget_suffix = f"_{max_b//1000}k" if attack_name in AL_ATTACKS else ""
                    config_name = f"{setup['id']}_{attack}{budget_suffix}_seed{seed}"
                    
                    config = {
                        "run": {
                            "name": config_name,
                            "seeds": [seed],
                            "device": "cuda:0"
                        },
                        "victim": {
                            "victim_id": victim_id,
                            "arch": setup['victim_arch'],
                            "channels": setup['channels'],
                            "num_classes": 10 if setup['victim_dataset'] != "GTSRB" else 43,
                            "input_size": [setup['size'], setup['size']],
                            "checkpoint_ref": f"runs/victims/{victim_id}_seed0.pt",
                            "normalization": None,
                            "output_mode": output_mode,
                            "temperature": 1.0
                        },
                        "dataset": {
                            "name": setup['victim_dataset'],
                            "data_mode": data_mode,
                            "surrogate_name": setup['surrogate'],
                            "train_split": True
                        },
                        "attack": {
                            "name": attack_name,
                            "output_mode": output_mode,
                            "max_budget": max_b # Added for internal round calculation
                        },
                        "substitute": {
                            "arch": setup['substitute_arch'],
                            "init_seed": 1234 + seed,
                            "trackA": {
                                "batch_size": 128,
                                "steps_coeff_c": 0.2
                            },
                            "optimizer": {
                                "name": "sgd",
                                "lr": 0.01,
                                "momentum": 0.9,
                                "weight_decay": 0.0005
                            },
                            "max_epochs": 1000,
                            "patience": 100
                        },
                        "budget": {
                            "max_budget": max_b,
                            "checkpoints": checkpoints
                        },
                        "cache": {
                            "enabled": True,
                            "policy": "temporary",
                            "delete_on_finish": True
                        }
                    }

                    if attack_strategy:
                        config["attack"]["strategy"] = attack_strategy

                    if attack in ["blackbox_ripper", "game", "dfms"]:
                        config["attack"]["proxy_dataset"] = {
                            "name": setup['surrogate'],
                            "data_mode": "surrogate",
                            "surrogate_name": setup['surrogate'],
                            "train_split": True
                        }

                    filename = f"configs/matrix/{config_name}.yaml"
                    with open(filename, 'w') as f:
                        yaml.dump(config, f, sort_keys=False)

    print(f"Generated configurations in configs/matrix/")

if __name__ == "__main__":
    generate_configs()
