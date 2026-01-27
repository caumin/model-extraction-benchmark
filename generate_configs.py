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

    seeds = [0, 1, 2]
    
    os.makedirs("configs/matrix", exist_ok=True)

    for setup in configs:
        for attack in attacks:
            for seed in seeds:
                # Determine attack type and strategy
                attack_name = attack
                attack_strategy = None
                
                if attack.startswith("activethief_"):
                    attack_strategy = attack.split("_")[1]
                    attack_name = "activethief"
                elif attack == "activethief":
                    # Default ActiveThief uses dfal_k_center
                    attack_strategy = "dfal_k_center"

                # Dissector is typically hard-label
                output_mode = "hard_top1" if attack == "blackbox_dissector" else "soft_prob"
                
                # Determine data mode
                data_free_attacks = {"dfme", "maze", "dfms", "game", "es", "blackbox_ripper"}
                data_mode = "data_free" if attack_name in data_free_attacks else "surrogate"

                config = {
                    "run": {
                        "name": f"{setup['id']}_{attack}_seed{seed}",
                        "seeds": [seed],
                        "device": "cuda:0"
                    },
                    "victim": {
                        "victim_id": f"{setup['victim_dataset'].lower()}_{setup['victim_arch']}",
                        "arch": setup['victim_arch'],
                        "channels": setup['channels'],
                        "num_classes": 10 if setup['victim_dataset'] != "GTSRB" else 43,
                        "input_size": [setup['size'], setup['size']],
                        "checkpoint_ref": f"runs/victims/{setup['victim_dataset'].lower()}_{setup['victim_arch']}_seed0.pt",
                        "normalization": None, # [0, 1] scaling
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
                        "output_mode": output_mode
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
                        "max_budget": 1000,
                        "checkpoints": [1000]
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

                filename = f"configs/matrix/{setup['id']}_{attack}_seed{seed}.yaml"
                with open(filename, 'w') as f:
                    yaml.dump(config, f, sort_keys=False)

    print(f"Generated {len(configs) * len(attacks) * len(seeds)} configuration files in configs/matrix/")

if __name__ == "__main__":
    generate_configs()
