"""Config validation logic."""

from typing import Dict, Any


def validate_config(config: Dict[str, Any]) -> None:
    """Validate experiment configuration.

    Args:
        config: Configuration dictionary from YAML

    Raises:
        ValueError: If configuration violates contract requirements
        KeyError: If required fields are missing
    """
    # Check data mode compatibility
    attack = config["attack"]["name"]
    data_mode = config["dataset"]["data_mode"]
    seed_name = config.get("dataset", {}).get("seed_name", config.get("dataset", {}).get("name"))
    # Data-free attacks must be in data_free mode; pool-based attacks must have a valid seed/surrogate config
    data_free_attacks = {"dfme", "maze", "dfms", "game", "es", "blackbox_ripper"}
    if attack in data_free_attacks and data_mode != "data_free":
        raise ValueError(f"{attack.upper()} requires data_free mode")
    if data_mode in {"seed", "surrogate"} and seed_name not in {"CIFAR10", "MNIST", "EMNIST", "FashionMNIST", "SVHN", "GTSRB"}:
        raise ValueError(f"Dataset '{seed_name}' not supported for {data_mode} mode")

    # Check output mode compatibility
    victim_mode = config["victim"]["output_mode"]
    attack_mode = config["attack"]["output_mode"]
    if victim_mode != attack_mode:
        raise ValueError(f"Mode mismatch: victim={victim_mode}, attack={attack_mode}")

    # Attacks that can work with soft labels (also compatible with hard labels)
    # Note: CloudLeak and SwiftThief can work in hard mode with one-hot vectors
    # but may have performance degradation compared to soft mode
    # InverseNet kept as soft-only due to its inversion-based nature
    soft_only_attacks = {"inversenet", "cloudleak", "swiftthief"}
    hard_only_attacks = {"blackbox_dissector"}
    if attack in soft_only_attacks and attack_mode != "soft_prob":
        raise ValueError(f"{attack} requires soft_prob output mode")
    if attack in hard_only_attacks and attack_mode != "hard_top1":
        raise ValueError(f"{attack} requires hard_top1 output mode")
    
    # Warning for soft attacks in hard mode (for awareness)
    # if attack in {"cloudleak", "swiftthief"} and attack_mode == "hard_top1":
    #     print(f"[WARNING] {attack} running in hard_top1 mode - performance may be degraded compared to soft_prob mode")

    # Check temperature for default oracle
    if config["victim"]["temperature"] != 1.0:
        raise ValueError("Default oracle requires T=1.0 in v1.0")

    # Check budget checkpoints
    checkpoints = config["budget"]["checkpoints"]
    max_budget = config["budget"]["max_budget"]
    if any(cp > max_budget for cp in checkpoints):
        raise ValueError(f"Checkpoint exceeds max_budget: {checkpoints} vs {max_budget}")

    # Check checkpoints are increasing
    if sorted(checkpoints) != checkpoints:
        raise ValueError(f"Checkpoints must be increasing: {checkpoints}")
