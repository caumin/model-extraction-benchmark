import yaml
import os
from pathlib import Path

def generate_configs_A1_1k():
    # Experimental Setup: SET-A1 only
    configs = [
        {
            "id": "SET-A1",
            "victim_dataset": "MNIST",
            "victim_arch": "lenet_mnist",
            "surrogate": "EMNIST",
            "substitute_arch": "lenet_mnist",
            "channels": 1,
            "size": 28
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

    # Categories
    AL_ATTACKS = {"activethief", "swiftthief", "cloudleak", "inversenet"}
    BATCH_POOL_ATTACKS = {"random", "knockoff_nets", "copycatcnn", "blackbox_dissector"}
    BATCH_FREE_ATTACKS = {"dfme", "maze", "dfms", "game", "es", "blackbox_ripper"}

    seeds = [0] # Seed 0 only
    budget_1k = 1000

    output_dir = "configs/matrix_A1_1k"
    os.makedirs(output_dir, exist_ok=True)

    for setup in configs:
        for attack in attacks:
            attack_name = attack
            attack_strategy = None
            if attack.startswith("activethief_"):
                attack_strategy = attack.split("_")[1]
                attack_name = "activethief"
            elif attack == "activethief":
                attack_strategy = "dfal_k_center"

            victim_arch_clean = setup['victim_arch']
            if victim_arch_clean == 'lenet_mnist':
                victim_arch_clean = 'lenet'
            
            victim_id = f"{setup['victim_dataset'].lower()}_{victim_arch_clean}"
            
            output_mode = "hard_top1" if attack == "blackbox_dissector" else "soft_prob"
            data_mode = "data_free" if attack_name in BATCH_FREE_ATTACKS else "surrogate"

            for seed in seeds:
                # Force budget to 1k for everything
                max_b = budget_1k
                checkpoints = [1000]
                
                config_name = f"{setup['id']}_{attack}_seed{seed}_1k"
                
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
                        "num_classes": 10,
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
                        "max_budget": max_b
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

                filename = f"{output_dir}/{config_name}.yaml"
                with open(filename, 'w') as f:
                    yaml.dump(config, f, sort_keys=False)

    print(f"Generated A1-1k configurations in {output_dir}")

if __name__ == "__main__":
    generate_configs_A1_1k()
