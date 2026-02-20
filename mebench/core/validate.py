"""Config validation logic."""

from typing import Dict, Any
from mebench.data.preprocessing import list_official_preprocess_profiles


def _require_positive_int(config: Dict[str, Any], path: str) -> int:
    cur: Any = config
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing required config field: {path}")
        cur = cur[part]
    try:
        val = int(cur)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Config field must be int: {path}={cur!r}") from exc
    if val <= 0:
        raise ValueError(f"Config field must be > 0: {path}={val}")
    return val


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

    # Substitute supervised-training unification (pool-based attacks).
    # Enforce presence of max_epochs/patience so runs cannot silently fall back
    # to per-attack defaults.
    pool_supervised_attacks = {
        "random",
        "activethief",
        "knockoff_nets",
        "cloudleak",
        "copycatcnn",
        "inversenet",
        "blackbox_dissector",
    }
    if attack in pool_supervised_attacks:
        _require_positive_int(config, "substitute.max_epochs")
        _require_positive_int(config, "substitute.patience")

    # Output-mode capability (soft-only / hard-only / both).
    # This is a *capability* taxonomy used for fair evaluation grouping.
    # Some attacks may run in the other mode by discarding information, but we
    # enforce the intended oracle capability per attack to avoid silent misuse.
    soft_only_attacks = {
        # Soft label (probabilities) required.
        "knockoff_nets",
        "maze",
        "dfme",
        "cloudleak",
        "game",
        "blackbox_ripper",
    }
    hard_only_attacks = {
        # Hard label (top-1) only.
        "copycatcnn",
        "inversenet",
        "dfms",  # DFMS-HL
        "blackbox_dissector",
    }
    both_attacks = {
        # Works under both soft_prob and hard_top1.
        "activethief",
        "swiftthief",
        "es",
    }

    if attack in soft_only_attacks and attack_mode != "soft_prob":
        raise ValueError(f"{attack} requires soft_prob output mode")
    if attack in hard_only_attacks and attack_mode != "hard_top1":
        raise ValueError(f"{attack} requires hard_top1 output mode")
    if attack in both_attacks and attack_mode not in {"soft_prob", "hard_top1"}:
        raise ValueError(f"{attack} requires output_mode to be soft_prob or hard_top1")

    # BlackboxRipper requires a pretrained generator checkpoint (official repo behavior).
    if attack == "blackbox_ripper":
        attack_cfg = config.get("attack", {})
        ckpt = attack_cfg.get("generator_checkpoint") or attack_cfg.get("generator_ckpt")
        if not ckpt:
            raise ValueError(
                "blackbox_ripper requires attack.generator_checkpoint (path to pretrained generator weights)"
            )
    
    # Warning for soft attacks in hard mode (for awareness)
    # if attack in {"cloudleak", "swiftthief"} and attack_mode == "hard_top1":
    #     print(f"[WARNING] {attack} running in hard_top1 mode - performance may be degraded compared to soft_prob mode")

    # Check temperature for default oracle
    if config["victim"]["temperature"] != 1.0:
        raise ValueError("Default oracle requires T=1.0 in v1.0")

    # Optional victim input scaling mode for strict-paper reproduction profiles.
    input_scale_mode = str(config.get("victim", {}).get("input_scale_mode", "unit")).lower()
    valid_scale_modes = {"unit", "0_1", "01", "tanh", "neg1_1", "-1_1", "-11"}
    if input_scale_mode not in valid_scale_modes:
        raise ValueError(
            f"victim.input_scale_mode must be one of {sorted(valid_scale_modes)}, got {input_scale_mode!r}"
        )

    official_profile = config.get("victim", {}).get("official_preprocess_profile")
    if official_profile is not None:
        names = set(list_official_preprocess_profiles())
        if str(official_profile) not in names:
            raise ValueError(
                "victim.official_preprocess_profile must be one of "
                f"{sorted(names)}, got {official_profile!r}"
            )

    # Check budget checkpoints
    checkpoints = config["budget"]["checkpoints"]
    max_budget = config["budget"]["max_budget"]
    if any(cp > max_budget for cp in checkpoints):
        raise ValueError(f"Checkpoint exceeds max_budget: {checkpoints} vs {max_budget}")

    # Check checkpoints are increasing
    if sorted(checkpoints) != checkpoints:
        raise ValueError(f"Checkpoints must be increasing: {checkpoints}")
