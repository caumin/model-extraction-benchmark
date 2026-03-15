"""Config validation logic."""

from typing import Dict, Any


def _normalize_sewerml_label_mode(raw_mode: Any) -> str:
    """Normalize SewerML label mode values used by the data loader."""
    mode = str(raw_mode if raw_mode is not None else "argmax").strip().lower().replace("-", "_")
    if mode in {"argmax", "single_label", "single", "multilabel", "multiclass"}:
        return "argmax"
    if mode in {"binary", "defect", "defect_binary", "binary_label", "is_defect"}:
        return "binary"
    raise ValueError(f"Unsupported Sewerml label mode: {raw_mode!r}")


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
    data_free_attacks = {
        "dfme",
        "ds",
        "maze",
        "dfms",
        "disguide",
        "game",
        "es",
        "blackbox_ripper",
    }
    if attack in data_free_attacks and data_mode != "data_free":
        raise ValueError(f"{attack.upper()} requires data_free mode")
    if data_mode in {"seed", "surrogate"} and seed_name not in {"CIFAR10", "MNIST", "EMNIST", "FashionMNIST", "SVHN", "GTSRB", "BelgiumTSC", "SewerML"}:
        raise ValueError(f"Dataset '{seed_name}' not supported for {data_mode} mode")

    if str(seed_name) == "SewerML" and "sewerml_label_mode" in config.get("dataset", {}):
        normalized_label_mode = _normalize_sewerml_label_mode(config["dataset"]["sewerml_label_mode"])
        if normalized_label_mode == "binary":
            victim_num_classes = int(config.get("victim", {}).get("num_classes", 0) or 0)
            if victim_num_classes != 1:
                raise ValueError("SewerML binary mode requires victim.num_classes=1 (single-logit binary)")

    if data_mode == "surrogate":
        dataset_cfg = config.get("dataset", {})
        norm_mode = str(dataset_cfg.get("surrogate_normalization", "standard")).strip().lower()
        if norm_mode not in {"standard", "none", "custom"}:
            raise ValueError(
                "dataset.surrogate_normalization must be one of "
                "['standard', 'none', 'custom']"
            )
        if norm_mode == "custom":
            mean = dataset_cfg.get("surrogate_norm_mean")
            std = dataset_cfg.get("surrogate_norm_std")
            if not isinstance(mean, (list, tuple)) or not isinstance(std, (list, tuple)):
                raise ValueError(
                    "dataset.surrogate_normalization='custom' requires "
                    "dataset.surrogate_norm_mean/std lists"
                )
            if len(mean) <= 0 or len(mean) != len(std):
                raise ValueError(
                    "dataset.surrogate_norm_mean/std must be non-empty and have equal length"
                )

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
        "marich",
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
        "marich",
    }
    both_attacks = {
        # Works under both soft_prob and hard_top1.
        "activethief",
        "swiftthief",
        "disguide",
        "ds",
        "es",
    }

    if attack in soft_only_attacks and attack_mode != "soft_prob":
        raise ValueError(f"{attack} requires soft_prob output mode")
    if attack in hard_only_attacks and attack_mode != "hard_top1":
        raise ValueError(f"{attack} requires hard_top1 output mode")
    if attack in both_attacks and attack_mode not in {"soft_prob", "hard_top1"}:
        raise ValueError(f"{attack} requires output_mode to be soft_prob or hard_top1")

    victim_num_classes = int(config.get("victim", {}).get("num_classes", 0) or 0)
    if victim_num_classes == 1:
        unsupported_single_logit_attacks = {
        }
        if attack in unsupported_single_logit_attacks:
            raise ValueError(f"{attack} does not support single-logit binary victims yet")

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

    # Victim inference policy key validation.
    inference_policy = str(config.get("victim", {}).get("inference_policy", "paper_repro")).lower()
    if inference_policy not in {"benchmark", "paper_repro"}:
        raise ValueError(
            "victim.inference_policy must be 'benchmark' or 'paper_repro', "
            f"got {inference_policy!r}"
        )

    # Check budget checkpoints
    checkpoints = config["budget"]["checkpoints"]
    max_budget = config["budget"]["max_budget"]
    if any(cp > max_budget for cp in checkpoints):
        raise ValueError(f"Checkpoint exceeds max_budget: {checkpoints} vs {max_budget}")

    # Check checkpoints are increasing
    if sorted(checkpoints) != checkpoints:
        raise ValueError(f"Checkpoints must be increasing: {checkpoints}")
