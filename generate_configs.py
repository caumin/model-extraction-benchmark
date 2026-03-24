from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


@dataclass(frozen=True)
class Setup:
    set_id: str
    victim_dataset: str
    victim_arch: str
    surrogate_name: str
    substitute_arch: str
    channels: int
    size: int
    num_classes: int


@dataclass(frozen=True)
class AttackSpec:
    name: str
    kind: str  # pool | synthetic
    label_capability: str  # soft_only | hard_only | both
    extra: Optional[Dict[str, Any]] = None


def _victim_id(setup: Setup) -> str:
    return f"{setup.victim_dataset.lower()}_{setup.victim_arch}"


def _attack_output_modes(capability: str, include_both_hard: bool) -> List[str]:
    if capability == "soft_only":
        return ["soft_prob"]
    if capability == "hard_only":
        return ["hard_top1"]
    if capability == "both":
        return ["soft_prob", "hard_top1"] if include_both_hard else ["soft_prob"]
    raise ValueError(f"Unknown label_capability: {capability}")


def _budget_for_kind(kind: str, pool_budget: int, synthetic_budget: int) -> int:
    if kind == "pool":
        return int(pool_budget)
    if kind == "synthetic":
        return int(synthetic_budget)
    raise ValueError(f"Unknown attack kind: {kind}")


def _checkpoints_for_budget(max_budget: int) -> List[int]:
    """Return monotonic checkpoints within max_budget.

    These checkpoints are for monitoring/logging only (not used for cross-attack
    comparison). We generate them as a mix of:
    - canonical budget milestones (1k/10k/100k/1m) clipped to max_budget
    - ratio-based milestones relative to max_budget (1/3 and 2/3)

    This avoids invalid configs when max_budget is smaller than fixed milestones
    (e.g., max_budget=30k must not include a 50k checkpoint).
    """

    b = int(max_budget)
    if b <= 0:
        return []

    canonical = [1_000, 10_000, 100_000, 1_000_000]
    ratio_points = [int(round(b / 3.0)), int(round(2.0 * b / 3.0))]
    base = canonical + ratio_points + [b]

    checkpoints = sorted({int(cp) for cp in base if 0 < int(cp) <= b})
    if not checkpoints or checkpoints[-1] != b:
        checkpoints.append(b)
    return checkpoints


def _budget_suffix(budget: int) -> str:
    b = int(budget)
    if b <= 0:
        return str(b)
    if b % 1_000_000 == 0:
        return f"{b // 1_000_000}m"
    if b % 1_000 == 0:
        return f"{b // 1_000}k"
    return str(b)


def _clean_yaml_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("*.yaml"):
        p.unlink()


# Matrix substitute policy:
# - Keep a unified substitute training recipe per setup.
# - Do not apply per-attack LR/batch alignment overrides in matrix generation.


def generate_configs(
    out_dir: Path,
    device: str,
    seeds: List[int],
    pool_budget: Optional[int],
    synthetic_budget: Optional[int],
    set_a_pool_budget: int,
    set_a_synthetic_budget: int,
    set_b_pool_budget: int,
    set_b_synthetic_budget: int,
    set_c_pool_budget: int,
    set_c_synthetic_budget: int,
    include_both_hard: bool,
    clean: bool,
    pool_num_workers: int,
    substitute_num_workers: int,
    substitute_train_num_workers: Optional[int],
    substitute_val_num_workers: Optional[int],
    imagenet_root: str,
) -> int:
    resolved_imagenet_root = str(imagenet_root)
    set_budgets: Dict[str, Dict[str, int]] = {
        "SET-A1": {
            "pool": int(set_a_pool_budget),
            "synthetic": int(set_a_synthetic_budget),
        },
        "SET-B1": {
            "pool": int(set_b_pool_budget),
            "synthetic": int(set_b_synthetic_budget),
        },
        "SET-C1": {
            "pool": int(set_c_pool_budget),
            "synthetic": int(set_c_synthetic_budget),
        },
    }


    global_pool_budget = int(pool_budget) if pool_budget is not None else None
    global_synthetic_budget = int(synthetic_budget) if synthetic_budget is not None else None

    setups = [
        Setup(
            set_id="SET-A1",
            victim_dataset="MNIST",
            victim_arch="lenet_mnist",
            surrogate_name="ImageNet",
            substitute_arch="lenet_mnist",
            channels=1,
            size=28,
            num_classes=10,
        ),
        Setup(
            set_id="SET-B1",
            victim_dataset="CIFAR10",
            victim_arch="resnet34",
            surrogate_name="ImageNet",
            substitute_arch="resnet18",
            channels=3,
            size=32,
            num_classes=10,
        ),
        Setup(
            set_id="SET-C1",
            victim_dataset="SewerML",
            victim_arch="xie2019",
            surrogate_name="ImageNet",
            substitute_arch="xie2019",
            channels=3,
            size=224,
            num_classes=1,
        ),
    ]

    # SET-C1 is the only SewerML matrix setup.
    # SewerML publishes both a multilabel model and a defect-vs-normal binary model,
    # but the multilabel model does not fit the standard extraction-attack contract
    # used in this benchmark. We therefore benchmark the binary checkpoint only.

    # Attack taxonomy used for a single-run-per-attack evaluation.
    # - pool: selection from a public image pool (surrogate)
    # - synthetic: query synthesis / data-free generation
    attacks: List[AttackSpec] = [

        AttackSpec("random", kind="pool", label_capability="both"),
        AttackSpec("knockoff_nets", kind="pool", label_capability="soft_only"),
        AttackSpec("cloudleak", kind="pool", label_capability="soft_only"),
        AttackSpec(
            "inversenet",
            kind="pool",
            label_capability="hard_only",
            extra={
                # Paper-aligned attack semantics (2021 Gong et al.):
                # K1:K2:K3=0.45:0.45:0.1, truncation top-1, coreset seed=20,
                # HCSS xi=0.02 and DeepFool max_iter=20.
                "phase_ratios": [0.45, 0.45, 0.1],
                "truncation_k": 1,
                "coreset_seed": 20,
                "hcss_xi": 0.02,
                "hcss_max_iter": 20,
                "batch_size": 128,
            },
        ),
        AttackSpec("blackbox_dissector", kind="pool", label_capability="hard_only"),
        AttackSpec(
            "marich",
            kind="pool",
            label_capability="hard_only",
            extra={
                "sampling": "all_elg",
                "init_points": 1000,
                "budget": 300,
                "gamma1": 0.8,
                "gamma2": 0.8,
                "iterations": 20,
                "epochs": 20,
                "budget_growth": 1.01,
                "epochs_growth": 1.02,
                # SET-B1 MARICH scoring can OOM on larger selection batches during
                # full-pool scoring, so keep the generated matrix profile more
                # conservative than the previous default.
                "selection_batch_size": 32,
            },
        ),

        AttackSpec(
            "activethief",
            kind="pool",
            label_capability="both",
            extra={
                "strategy": "dfal_k_center",
                "scoring_batch_size": 512,
            },
        ),
        AttackSpec(
            "activethief",
            kind="pool",
            label_capability="both",
            extra={
                "strategy": "uncertainty",
                "variant": "uncertainty",
                "scoring_batch_size": 512,
            },
        ),
        AttackSpec(
            "activethief",
            kind="pool",
            label_capability="both",
            extra={
                "strategy": "dfal",
                "variant": "dfal",
                "scoring_batch_size": 512,
            },
        ),
        AttackSpec(
            "swiftthief",
            kind="pool",
            label_capability="both",
            extra={
                "cl_epochs": 40,
                "kd_epochs": 40,
                "patience": 50,
            },
        ),

        # [UNIFIED] All data-free attacks standardized to:
        # - SGD Student (LR via setup)
        # - Adam Generator (MAZE changed from SGD)
        # - MultiStepLR (Handled in code now, not config)
        
        AttackSpec("dfme", kind="synthetic", label_capability="soft_only"),
        AttackSpec("ds", kind="synthetic", label_capability="both"),
        AttackSpec("maze", kind="synthetic", label_capability="soft_only"),
        AttackSpec("disguide", kind="synthetic", label_capability="both"),
        # NOTE: GAME is temporarily excluded from the default matrix because its
        # TDL pretraining requires proxy labels aligned to victim classes.
        AttackSpec("blackbox_ripper", kind="synthetic", label_capability="soft_only"),
        AttackSpec(
            "dfms",
            kind="synthetic",
            label_capability="hard_only",
            extra={
                # DFMS-HL (hard-label, data-free) benefits substantially from larger
                # batches on modern GPUs. Keep this explicit in the generated configs
                # so runs are reproducible across machines.
                "batch_size": 256,
            },
        ),
    ]

    if clean:
        _clean_yaml_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    protocol_version = "1.3"
    # Heuristic default substitute profile.
    substitute_batch_size = 512

    # Unified pool-based budget protocol
    pool_initial_seed_ratio = 0.1
    pool_validation_budget_ratio = 0.2
    pool_iterations = 10

    # Unified simple substitute supervised training defaults.
    # NOTE: these are heuristic benchmark defaults applied per setup.
    unified_substitute_lr = 0.04
    set_a_unified_substitute_max_epochs = 200
    set_a_unified_substitute_patience = 20
    # SET-B1 default profile.
    set_b_substitute_batch_size = 256
    set_b_unified_substitute_lr = 0.1
    set_b_unified_substitute_max_epochs = 1000
    set_b_unified_substitute_patience = 100
    # SET-C1 uses a fixed-length substitute schedule.
    # We keep best-checkpoint selection by validation loss, but disable effective
    # early stopping by setting patience equal to the full epoch budget. The
    # generated configs are consumed by attack implementations that validate once
    # per epoch and restore the best validation-loss checkpoint after training.
    # SET-C1 runs 224x224 SewerML/Xie2019 substitutes. Keep the evaluation passes
    # conservative, but use a 128-image training batch with a linearly scaled LR
    # (0.05) relative to the previous 256->0.1 target.
    set_c_substitute_batch_size = 128
    set_c_substitute_val_batch_size = 32
    set_c_eval_batch_size = 32
    set_c_sewerml_eval_max_samples = 10_000
    set_c_sewerml_eval_subset_seed = 42
    set_c_unified_substitute_lr = 0.05
    set_c_unified_substitute_max_epochs = 90
    set_c_unified_substitute_patience = 90

    # Synthetic/data-free attacks do not depend on the surrogate dataset.
    # Generate them once per victim. Prefer an ImageNet-based SET when available
    # (useful as a canonical location for configs), otherwise keep the first.
    synthetic_canonical_by_victim: Dict[str, Setup] = {}
    for s in setups:
        key = _victim_id(s)
        if key not in synthetic_canonical_by_victim:
            synthetic_canonical_by_victim[key] = s
            continue

        prev = synthetic_canonical_by_victim[key]
        prev_is_imagenet = str(prev.surrogate_name).lower() == "imagenet"
        curr_is_imagenet = str(s.surrogate_name).lower() == "imagenet"
        if curr_is_imagenet and not prev_is_imagenet:
            synthetic_canonical_by_victim[key] = s

    resolved_sub_train_workers = (
        int(substitute_train_num_workers)
        if substitute_train_num_workers is not None
        else int(substitute_num_workers)
    )
    resolved_sub_val_workers = (
        int(substitute_val_num_workers)
        if substitute_val_num_workers is not None
        else int(substitute_num_workers)
    )

    count = 0
    for setup in setups:
        for attack in attacks:
            if attack.kind == "synthetic":
                victim_key = _victim_id(setup)
                canonical = synthetic_canonical_by_victim.get(victim_key)
                if canonical is not None and setup.set_id != canonical.set_id:
                    continue
            set_budget_cfg = set_budgets.get(setup.set_id)
            if set_budget_cfg is None:
                raise ValueError(f"Missing budget policy for setup: {setup.set_id}")

            if attack.kind == "pool":
                max_budget = (
                    int(global_pool_budget)
                    if global_pool_budget is not None
                    else int(set_budget_cfg["pool"])
                )
            elif attack.kind == "synthetic":
                max_budget = (
                    int(global_synthetic_budget)
                    if global_synthetic_budget is not None
                    else int(set_budget_cfg["synthetic"])
                )
            else:
                raise ValueError(f"Unknown attack kind: {attack.kind}")
            checkpoints = _checkpoints_for_budget(max_budget)

            victim_id = _victim_id(setup)
            data_mode = "data_free" if attack.kind == "synthetic" else "surrogate"

            for output_mode in _attack_output_modes(attack.label_capability, include_both_hard=include_both_hard):
                for seed in seeds:
                    suffix_mode = "soft" if output_mode == "soft_prob" else "hard"
                    suffix_budget = _budget_suffix(max_budget)

                    attack_variant = None
                    attack_extra = dict(attack.extra or {})
                    if "variant" in attack_extra:
                        attack_variant = str(attack_extra.pop("variant"))

                    # [ADDED] Allow attacks to override substitute config (e.g. ActiveThief needs Adam)
                    substitute_override = attack_extra.pop("substitute", {})

                    attack_name_for_filename = attack.name
                    if attack_variant:
                        attack_name_for_filename = f"{attack.name}_{attack_variant}"

                    run_name = f"{setup.set_id}_{attack_name_for_filename}_{suffix_mode}_{suffix_budget}_seed{seed}"
                    
                    setup_substitute_batch_size = substitute_batch_size
                    target_lr = unified_substitute_lr
                    setup_max_epochs = int(set_a_unified_substitute_max_epochs)
                    setup_patience = int(set_a_unified_substitute_patience)
                    setup_set_id = str(setup.set_id).strip().upper()
                    if setup_set_id == "SET-B1":
                        setup_substitute_batch_size = int(set_b_substitute_batch_size)
                        target_lr = float(set_b_unified_substitute_lr)
                        setup_max_epochs = int(set_b_unified_substitute_max_epochs)
                        setup_patience = int(set_b_unified_substitute_patience)
                    elif setup_set_id == "SET-C1":
                        setup_substitute_batch_size = int(set_c_substitute_batch_size)
                        target_lr = float(set_c_unified_substitute_lr)
                        setup_max_epochs = int(set_c_unified_substitute_max_epochs)
                        setup_patience = int(set_c_unified_substitute_patience)

                    # Default substitute config
                    substitute_config = {
                        "arch": setup.substitute_arch,
                        "init_seed": 1234 + seed,
                        "batch_size": int(setup_substitute_batch_size),
                        "num_workers": int(substitute_num_workers),
                        "train_num_workers": int(resolved_sub_train_workers),
                        "val_num_workers": int(resolved_sub_val_workers),
                        "optimizer": {
                            "name": "sgd",
                            "lr": target_lr,
                            "momentum": 0.9,
                            "weight_decay": 5e-4,
                        },
                        "scheduler": {"name": "multistep", "milestones_ratio": [0.5, 0.75], "gamma": 0.1},
                        "max_epochs": int(setup_max_epochs),
                        "patience": int(setup_patience),
                    }
                    
                    # Merge override recursively (simple 1-level merge for optimizer)
                    if substitute_override:
                        for k, v in substitute_override.items():
                            if k == "optimizer" and isinstance(v, dict) and "optimizer" in substitute_config:
                                substitute_config["optimizer"].update(v)
                            else:
                                substitute_config[k] = v

                    victim_config: Dict[str, Any] = {
                        "victim_id": victim_id,
                        "arch": setup.victim_arch,
                        "channels": setup.channels,
                        "num_classes": setup.num_classes,
                        "input_size": [setup.size, setup.size],
                        "checkpoint_ref": f"runs/victims/{victim_id}_seed0.pt",
                        "normalization": None,
                        "output_mode": output_mode,
                        "temperature": 1.0,
                    }

                    if setup.set_id == "SET-B1":
                        victim_config.update(
                            {
                                "inference_policy": "benchmark",
                                "victim_id": "cifar10_resnet34_8x_official",
                                "arch": "resnet34",
                                "width_mult": 1,
                                "official_preprocess_profile": "dfme_cifar10_test",
                                "checkpoint_ref": "runs/victims/cifar10-resnet34_8x.pt",
                            }
                        )
                    elif setup.set_id == "SET-C1":
                        victim_config.update(
                            {
                                "victim_id": "sewerml_xie2019_binary_defect_normal",
                                "arch": "xie2019",
                                "num_classes": 1,
                                "channels": 3,
                                "input_size": [224, 224],
                                "checkpoint_ref": "runs/victims/xie2019_binary-binary-version_1.pth",
                                "inference_policy": "benchmark",
                            }
                        )

                    cfg: Dict[str, Any] = {
                        "run": {"name": run_name, "seeds": [seed], "device": device},
                        "benchmark": {"protocol_version": protocol_version},
                        "victim": victim_config,
                        "dataset": {
                            "name": setup.victim_dataset,
                            "data_mode": data_mode,
                            "surrogate_name": setup.surrogate_name,
                            "surrogate_normalization": "none" if attack.kind == "synthetic" else "standard",
                            "num_workers": int(pool_num_workers),
                            "train_split": True,
                            "channels": setup.channels,
                            "input_size": [setup.size, setup.size],
                        },
                        "attack": {
                            "name": attack.name,
                            "output_mode": output_mode,
                            "pool_num_workers": int(pool_num_workers),
                            **attack_extra,
                        },
                        "substitute": substitute_config,
                        "budget": {"max_budget": max_budget, "checkpoints": checkpoints},
                        "cache": {
                            "enabled": True,
                            "policy": "temporary",
                            "delete_on_finish": True,
                        },
                    }

                    if setup.set_id == "SET-C1":
                        cfg["benchmark"]["eval_batch_size"] = int(set_c_eval_batch_size)
                        cfg["dataset"]["sewerml_label_mode"] = "binary"
                        cfg["dataset"]["sewerml_ann_root"] = "D:/Sewer-ML"
                        cfg["dataset"]["sewerml_data_root"] = "D:/Sewer-ML"
                        cfg["dataset"]["sewerml_eval_split"] = "Valid"
                        cfg["dataset"]["sewerml_max_samples"] = int(set_c_sewerml_eval_max_samples)
                        cfg["dataset"]["sewerml_subset_seed"] = int(set_c_sewerml_eval_subset_seed)
                        cfg["substitute"]["val_batch_size"] = int(set_c_substitute_val_batch_size)
                        # SET-C1 uses 224x224 SewerML inputs, so keep large scoring /
                        # selection passes more conservative than the CIFAR-style cells.
                        if attack.name == "activethief":
                            cfg["attack"]["scoring_batch_size"] = min(
                                int(cfg["attack"].get("scoring_batch_size", 512)),
                                128,
                            )
                        elif attack.name == "blackbox_dissector":
                            cfg["attack"]["selection_batch_size"] = min(
                                int(cfg["attack"].get("selection_batch_size", 512)),
                                128,
                            )

                    if attack.kind == "pool":
                        cfg["attack"].setdefault("initial_seed_ratio", pool_initial_seed_ratio)
                        cfg["attack"].setdefault("validation_budget_ratio", pool_validation_budget_ratio)
                        # Canonical loop-count key for generated configs.
                        # Backward compatibility for legacy `rounds` is handled in
                        # attack implementations via fallback parsing.
                        cfg["attack"].setdefault("iterations", pool_iterations)

                    # GAME canonical budget keys (future-ready).
                    # GAME is currently excluded from matrix generation, so this
                    # has no effect on today's outputs; it keeps naming consistent
                    # when GAME is re-enabled.
                    if attack.name == "game":
                        cfg["attack"].setdefault("query_budget", int(max_budget))
                        cfg["attack"].setdefault("nominal_query_budget", 0)

                    # CloudLeak: disable pretrained substitutes by default for fairness.
                    if attack.name == "cloudleak":
                        cfg["attack"].setdefault("use_pretrained", False)
                        # Official-style FeatureFool settings (optimize.py): bounded feature matching.
                        cfg["attack"].setdefault("featurefool_objective", "euclidean")
                        cfg["attack"].setdefault("max_thres", 10.0 / 255.0)
                        cfg["attack"].setdefault("lbfgs_iters", 10)

                    def _maybe_add_imagenet_imagefolder_keys(d: Dict[str, Any]) -> None:
                        if str(d.get("data_mode")).lower() != "surrogate":
                            return
                        if str(d.get("surrogate_name")).lower() != "imagenet":
                            return
                        # ImageNet surrogate (ImageFolder format): use set-specific deterministic subset size.
                        # Default local path is configurable via --imagenet-root.
                        setup_id_upper = str(setup.set_id).strip().upper()
                        surrogate_max_samples = 100_000 if setup_id_upper in {"SET-B1", "SET-C1"} else 50_000
                        d.update(
                            {
                                "surrogate_root": resolved_imagenet_root,
                                "surrogate_resize": [setup.size, setup.size],
                                "surrogate_max_samples": int(surrogate_max_samples),
                                "surrogate_subset_seed": 42,
                            }
                        )

                    _maybe_add_imagenet_imagefolder_keys(cfg["dataset"])

                    if attack.name == "knockoff_nets":
                        cfg["attack"]["offline_train_epochs"] = cfg["substitute"]["max_epochs"]

                    if attack.name in {"blackbox_ripper", "dfms"}:
                        cfg["attack"]["proxy_dataset"] = {
                            "name": setup.surrogate_name,
                            "data_mode": "surrogate",
                            "surrogate_name": setup.surrogate_name,
                            "surrogate_normalization": "none",
                            "num_workers": int(pool_num_workers),
                            "train_split": True,
                            "channels": setup.channels,
                            "input_size": [setup.size, setup.size],
                        }
                        _maybe_add_imagenet_imagefolder_keys(cfg["attack"]["proxy_dataset"])

                    # DFME/DFMS consume oracle outputs immediately inside GPU training
                    # loops. Keep oracle outputs on-device to avoid unnecessary
                    # GPU->CPU->GPU transfers while preserving the 1-query=1-image
                    # contract.
                    if attack.name in {"dfms", "dfme", "ds", "disguide"}:
                        cfg["victim"]["return_outputs_on_cpu"] = False

                    if attack.name == "blackbox_ripper":
                        if setup.victim_dataset == "CIFAR10":
                            gen_name = "cifar_100_6_classes_gan"
                            gen_ckpt = f"checkpoints/blackbox_ripper/official/{gen_name}"
                        else:
                            gen_name = "cifar_10_gan"
                            gen_ckpt = f"checkpoints/blackbox_ripper/official/{gen_name}.pth"
                        cfg["attack"]["generator_name"] = gen_name
                        cfg["attack"]["generator_checkpoint"] = gen_ckpt

                    out_path = out_dir / f"{run_name}.yaml"
                    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
                    count += 1

    return count


def generate_paperlike_configs(
    out_dir: Path,
    device: str,
    seeds: List[int],
    clean: bool,
    pool_num_workers: int,
    substitute_num_workers: int,
    substitute_train_num_workers: Optional[int],
    substitute_val_num_workers: Optional[int],
    imagenet_root: str,
) -> int:
    resolved_imagenet_root = str(imagenet_root)
    mnist_setup = Setup(
        set_id="SET-A1",
        victim_dataset="MNIST",
        victim_arch="classifier",
        surrogate_name="EMNIST",
        substitute_arch="cnn32",
        channels=1,
        size=28,
        num_classes=10,
    )

    cifar10_setup = Setup(
        set_id="SET-B1",
        victim_dataset="CIFAR10",
        victim_arch="resnet18",
        surrogate_name="ImageNet",
        substitute_arch="resnet18",
        channels=3,
        size=32,
        num_classes=10,
    )

    if clean:
        _clean_yaml_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    protocol_version = "1.3"
    substitute_batch_size = 128
    unified_substitute_max_epochs = 200
    unified_substitute_patience = 20

    resolved_sub_train_workers = (
        int(substitute_train_num_workers)
        if substitute_train_num_workers is not None
        else int(substitute_num_workers)
    )
    resolved_sub_val_workers = (
        int(substitute_val_num_workers)
        if substitute_val_num_workers is not None
        else int(substitute_num_workers)
    )

    def _paper_substitute_config(
        *,
        arch: str,
        seed: int,
        optimizer_lr: float,
        width_mult: int = 1,
        scheduler_name: str = "multistep",
    ) -> Dict[str, Any]:
        cfg = {
            "arch": str(arch),
            "init_seed": 1234 + int(seed),
            "batch_size": substitute_batch_size,
            "num_workers": int(substitute_num_workers),
            "train_num_workers": int(resolved_sub_train_workers),
            "val_num_workers": int(resolved_sub_val_workers),
            "optimizer": {
                "name": "sgd",
                "lr": float(optimizer_lr),
                "momentum": 0.9,
                "weight_decay": 5e-4,
            },
            "max_epochs": unified_substitute_max_epochs,
            "patience": unified_substitute_patience,
        }
        if int(width_mult) != 1:
            cfg["width_mult"] = int(width_mult)

        if scheduler_name == "cosine":
            cfg["scheduler"] = {"name": "cosine", "t_max_epochs": unified_substitute_max_epochs}
        else:
            cfg["scheduler"] = {
                "name": "multistep",
                "milestones_ratio": [0.5, 0.75],
                "gamma": 0.1,
            }
        return cfg

    def _paper_base_config(
        *,
        run_name: str,
        seed: int,
        victim_arch: str,
        victim_output_mode: str,
        max_budget: int,
        attack_cfg: Dict[str, Any],
        substitute_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "run": {"name": run_name, "seeds": [int(seed)], "device": device},
            "benchmark": {"protocol_version": protocol_version},
            "victim": {
                "victim_id": f"cifar10_{victim_arch}",
                "arch": str(victim_arch),
                "channels": cifar10_setup.channels,
                "num_classes": cifar10_setup.num_classes,
                "input_size": [cifar10_setup.size, cifar10_setup.size],
                "checkpoint_ref": f"runs/victims/cifar10_{victim_arch}_seed0.pt",
                "normalization": None,
                "output_mode": victim_output_mode,
                "temperature": 1.0,
            },
            "dataset": {
                "name": cifar10_setup.victim_dataset,
                "data_mode": "data_free",
                "surrogate_name": cifar10_setup.surrogate_name,
                "num_workers": int(pool_num_workers),
                "train_split": True,
                "channels": cifar10_setup.channels,
                "input_size": [cifar10_setup.size, cifar10_setup.size],
            },
            "attack": dict(attack_cfg),
            "substitute": dict(substitute_cfg),
            "budget": {"max_budget": int(max_budget), "checkpoints": _checkpoints_for_budget(int(max_budget))},
            "cache": {
                "enabled": True,
                "policy": "temporary",
                "delete_on_finish": True,
            },
        }
        return cfg

    def _paper_surrogate_dataset(base_name: str, seed: int, *, name: str) -> Dict[str, Any]:
        dataset_cfg: Dict[str, Any] = {
            "name": str(name),
            "data_mode": "surrogate",
            "surrogate_name": str(base_name),
            "surrogate_normalization": "standard",
            "num_workers": int(pool_num_workers),
            "train_split": True,
            "channels": cifar10_setup.channels,
            "input_size": [cifar10_setup.size, cifar10_setup.size],
        }
        if str(base_name).lower() == "imagenet":
            dataset_cfg.update(
                {
                    "surrogate_root": resolved_imagenet_root,
                    "surrogate_resize": [cifar10_setup.size, cifar10_setup.size],
                    "surrogate_max_samples": 100_000,
                    "surrogate_subset_seed": int(seed),
                }
            )
        return dataset_cfg

    def _paper_pool_base_config(
        *,
        run_name: str,
        seed: int,
        victim_arch: str,
        victim_output_mode: str,
        max_budget: int,
        attack_cfg: Dict[str, Any],
        substitute_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "run": {"name": run_name, "seeds": [int(seed)], "device": device},
            "benchmark": {"protocol_version": protocol_version},
            "victim": {
                "victim_id": f"cifar10_{victim_arch}",
                "arch": str(victim_arch),
                "channels": cifar10_setup.channels,
                "num_classes": cifar10_setup.num_classes,
                "input_size": [cifar10_setup.size, cifar10_setup.size],
                "checkpoint_ref": f"runs/victims/cifar10_{victim_arch}_seed0.pt",
                "normalization": None,
                "output_mode": victim_output_mode,
                "temperature": 1.0,
            },
            "dataset": _paper_surrogate_dataset(
                base_name=cifar10_setup.surrogate_name,
                seed=int(seed),
                name=cifar10_setup.victim_dataset,
            ),
            "attack": dict(attack_cfg),
            "substitute": dict(substitute_cfg),
            "budget": {
                "max_budget": int(max_budget),
                "checkpoints": _checkpoints_for_budget(int(max_budget)),
            },
            "cache": {
                "enabled": True,
                "policy": "temporary",
                "delete_on_finish": True,
            },
        }
        return cfg

    def _blackbox_ripper_generator_cfg(victim_dataset: str) -> Tuple[str, str]:
        if str(victim_dataset).upper() == "CIFAR10":
            gen_name = "cifar_100_6_classes_gan"
            gen_ckpt = f"checkpoints/blackbox_ripper/official/{gen_name}"
            return gen_name, gen_ckpt
        gen_name = "cifar_10_gan"
        gen_ckpt = f"checkpoints/blackbox_ripper/official/{gen_name}.pth"
        return gen_name, gen_ckpt

    count = 0
    for seed in seeds:
        # InverseNet paper-like profile (MNIST hard-label):
        # - victim: Classifier
        # - query pool: EMNIST Letters
        # - substitute: CNN32
        # - budget: 30k
        # - K1:K2:K3 = 0.45:0.45:0.1
        inversenet_budget = 30_000
        inversenet_run_name = (
            f"{mnist_setup.set_id}_inversenet_paper_hard_{_budget_suffix(inversenet_budget)}_seed{seed}"
        )
        inversenet_attack = {
            "name": "inversenet",
            "output_mode": "hard_top1",
            "max_budget": int(inversenet_budget),
            "pool_num_workers": int(pool_num_workers),
            "batch_size": 128,
            "phase_ratios": [0.45, 0.45, 0.1],
            "truncation_k": 1,
            "coreset_seed": 20,
            "hcss_xi": 0.02,
            "hcss_max_iter": 20,
        }
        inversenet_substitute = _paper_substitute_config(
            arch="cnn32",
            seed=int(seed),
            optimizer_lr=0.01,
            scheduler_name="multistep",
        )
        inversenet_cfg: Dict[str, Any] = {
            "run": {"name": inversenet_run_name, "seeds": [int(seed)], "device": device},
            "benchmark": {"protocol_version": protocol_version},
            "victim": {
                "victim_id": "mnist_classifier",
                "arch": "classifier",
                "dropout_prob": 0.2,
                "channels": mnist_setup.channels,
                "num_classes": mnist_setup.num_classes,
                "input_size": [mnist_setup.size, mnist_setup.size],
                "checkpoint_ref": "runs/victims/mnist_classifier_seed0.pt",
                "normalization": None,
                "output_mode": "hard_top1",
                "temperature": 1.0,
            },
            "dataset": {
                "name": mnist_setup.victim_dataset,
                "data_mode": "surrogate",
                "surrogate_name": mnist_setup.surrogate_name,
                "surrogate_normalization": "standard",
                "surrogate_split": "letters",
                "num_workers": int(pool_num_workers),
                "train_split": True,
                "channels": mnist_setup.channels,
                "input_size": [mnist_setup.size, mnist_setup.size],
            },
            "attack": inversenet_attack,
            "substitute": inversenet_substitute,
            "budget": {
                "max_budget": int(inversenet_budget),
                "checkpoints": [1_000, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000],
            },
            "cache": {
                "enabled": True,
                "policy": "temporary",
                "delete_on_finish": True,
            },
        }
        (out_dir / f"{inversenet_run_name}.yaml").write_text(
            yaml.safe_dump(inversenet_cfg, sort_keys=False),
            encoding="utf-8",
        )
        count += 1

        # MARICH paper-like profile (CIFAR-10 hard-label):
        # - victim: ResNet-34-8x official checkpoint
        # - query pool: ImageNet surrogate
        # - staged selector (all_elg)
        # - budget: 30k
        marich_budget = 30_000
        marich_run_name = (
            f"{cifar10_setup.set_id}_marich_paper_hard_{_budget_suffix(marich_budget)}_seed{seed}"
        )
        marich_attack = {
            "name": "marich",
            "output_mode": "hard_top1",
            "max_budget": int(marich_budget),
            "pool_num_workers": int(pool_num_workers),
            "sampling": "all_elg",
            "init_points": 1_000,
            "budget": 300,
            "gamma1": 0.8,
            "gamma2": 0.8,
            "rounds": 20,
            "epochs": 20,
            "budget_growth": 1.01,
            "epochs_growth": 1.02,
            "batch_size": 128,
            "selection_batch_size": 32,
            "lr": 0.01,
            "weight_decay": 5e-4,
            "patience": 20,
        }
        marich_substitute = _paper_substitute_config(
            arch="resnet18",
            seed=int(seed),
            optimizer_lr=0.01,
            width_mult=1,
            scheduler_name="multistep",
        )
        marich_cfg: Dict[str, Any] = {
            "run": {"name": marich_run_name, "seeds": [int(seed)], "device": device},
            "benchmark": {"protocol_version": protocol_version},
            "victim": {
                "victim_id": "cifar10_resnet34_8x_official",
                "arch": "resnet34",
                "width_mult": 1,
                "channels": cifar10_setup.channels,
                "num_classes": cifar10_setup.num_classes,
                "input_size": [cifar10_setup.size, cifar10_setup.size],
                "checkpoint_ref": "runs/victims/cifar10-resnet34_8x.pt",
                "output_mode": "hard_top1",
                "temperature": 1.0,
                "inference_policy": "benchmark",
                "official_preprocess_profile": "dfme_cifar10_test",
            },
            "dataset": {
                "name": cifar10_setup.victim_dataset,
                "data_mode": "surrogate",
                "surrogate_name": "ImageNet",
                "surrogate_normalization": "standard",
                "surrogate_root": resolved_imagenet_root,
                "surrogate_resize": [cifar10_setup.size, cifar10_setup.size],
                "surrogate_max_samples": 100_000,
                "surrogate_subset_seed": 42,
                "num_workers": int(pool_num_workers),
                "train_split": True,
                "channels": cifar10_setup.channels,
                "input_size": [cifar10_setup.size, cifar10_setup.size],
            },
            "attack": marich_attack,
            "substitute": marich_substitute,
            "budget": {
                "max_budget": int(marich_budget),
                "checkpoints": [1_000, 5_000, 10_000, 20_000, 30_000],
            },
            "cache": {
                "enabled": True,
                "policy": "temporary",
                "delete_on_finish": True,
            },
        }
        (out_dir / f"{marich_run_name}.yaml").write_text(
            yaml.safe_dump(marich_cfg, sort_keys=False),
            encoding="utf-8",
        )
        count += 1

        # BlackboxDissector paper-like profile (CIFAR-10 hard-label):
        # - victim: ResNet-34-8x official checkpoint
        # - query pool: ImageNet surrogate
        # - erase variants N=10, erase_rate=0.25
        # - iterative budgets up to 30k
        dissector_budget = 30_000
        dissector_run_name = (
            f"{cifar10_setup.set_id}_blackbox_dissector_paper_hard_{_budget_suffix(dissector_budget)}_seed{seed}"
        )
        dissector_attack = {
            "name": "blackbox_dissector",
            "output_mode": "hard_top1",
            "max_budget": int(dissector_budget),
            "pool_num_workers": int(pool_num_workers),
            "batch_size": 128,
            "n_variants": 10,
            "erase_rate": 0.25,
            "iterative_budgets": [1_000, 5_000, 10_000, 20_000, 30_000],
            "lr": 0.02,
            "momentum": 0.9,
            "max_epochs": 200,
            "patience": 20,
            "l2_reg": 5e-4,
            "selection_batch_size": 256,
            "sl": 0.02,
            "sh": 0.4,
            "r1": 0.3,
            "r2": 3.3,
            "fill_min": 0.0,
            "fill_max": 1.0,
        }
        dissector_substitute = _paper_substitute_config(
            arch="resnet34",
            seed=int(seed),
            optimizer_lr=0.02,
            width_mult=1,
            scheduler_name="multistep",
        )
        dissector_cfg: Dict[str, Any] = {
            "run": {"name": dissector_run_name, "seeds": [int(seed)], "device": device},
            "benchmark": {"protocol_version": protocol_version},
            "victim": {
                "victim_id": "cifar10_resnet34_8x_official",
                "arch": "resnet34",
                "width_mult": 1,
                "channels": cifar10_setup.channels,
                "num_classes": cifar10_setup.num_classes,
                "input_size": [cifar10_setup.size, cifar10_setup.size],
                "checkpoint_ref": "runs/victims/cifar10-resnet34_8x.pt",
                "output_mode": "hard_top1",
                "temperature": 1.0,
                "inference_policy": "benchmark",
                "official_preprocess_profile": "dfme_cifar10_test",
            },
            "dataset": {
                "name": cifar10_setup.victim_dataset,
                "data_mode": "surrogate",
                "surrogate_name": "ImageNet",
                "surrogate_normalization": "standard",
                "surrogate_root": "<FILL_ME>",
                "surrogate_resize": [cifar10_setup.size, cifar10_setup.size],
                "surrogate_max_samples": 100_000,
                "surrogate_subset_seed": 42,
                "num_workers": int(pool_num_workers),
                "train_split": True,
                "channels": cifar10_setup.channels,
                "input_size": [cifar10_setup.size, cifar10_setup.size],
            },
            "attack": dissector_attack,
            "substitute": dissector_substitute,
            "budget": {
                "max_budget": int(dissector_budget),
                "checkpoints": [1_000, 5_000, 10_000, 20_000, 30_000],
            },
            "cache": {
                "enabled": True,
                "policy": "temporary",
                "delete_on_finish": True,
            },
        }
        (out_dir / f"{dissector_run_name}.yaml").write_text(
            yaml.safe_dump(dissector_cfg, sort_keys=False),
            encoding="utf-8",
        )
        count += 1

        # BlackboxRipper paper-like profile (CIFAR-10 soft-label):
        # - victim: ResNet-34-8x official checkpoint
        # - data-free evolutionary latent optimization loop
        # - budget: 30k
        ripper_budget = 30_000
        ripper_run_name = (
            f"{cifar10_setup.set_id}_blackbox_ripper_paper_soft_{_budget_suffix(ripper_budget)}_seed{seed}"
        )
        ripper_attack = {
            "name": "blackbox_ripper",
            "output_mode": "soft_prob",
            "max_budget": int(ripper_budget),
            "generator_name": "cifar_progan",
            "generator_checkpoint": "checkpoints/blackbox_ripper/official/cifar_100_6_classes_gan",
            "generator_output_range": "tanh",
            "population_size": 30,
            "elite_size": 10,
            "latent_bound": 3.0,
            "mutation_scale": 1.0,
            "fitness_threshold": 0.02,
            "max_evolve_iters": 10,
            "substitute_epochs": 200,
            "train_batch_size": 64,
            "batches_per_epoch": 1_000,
            "substitute_lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "log_interval": 25,
        }
        ripper_substitute = _paper_substitute_config(
            arch="resnet18",
            seed=int(seed),
            optimizer_lr=0.01,
            width_mult=1,
            scheduler_name="multistep",
        )
        ripper_cfg = _paper_base_config(
            run_name=ripper_run_name,
            seed=int(seed),
            victim_arch="resnet34",
            victim_output_mode="soft_prob",
            max_budget=int(ripper_budget),
            attack_cfg=ripper_attack,
            substitute_cfg=ripper_substitute,
        )
        ripper_cfg["victim"]["victim_id"] = "cifar10_resnet34_8x_official"
        ripper_cfg["victim"]["width_mult"] = 1
        ripper_cfg["victim"]["checkpoint_ref"] = "runs/victims/cifar10-resnet34_8x.pt"
        ripper_cfg["victim"]["inference_policy"] = "benchmark"
        ripper_cfg["victim"]["normalization"] = None
        ripper_cfg["victim"]["official_preprocess_profile"] = "dfme_cifar10_test"

        (out_dir / f"{ripper_run_name}.yaml").write_text(
            yaml.safe_dump(ripper_cfg, sort_keys=False),
            encoding="utf-8",
        )
        count += 1

        # MAZE paper-like profile (CIFAR-10):
        # - victim: ResNet-20 (paper setting)
        # - clone: WideResNet-22
        # - budget: 30M
        # - B=128, NG=1, NC=5, NR=10, m=10
        maze_budget = 30_000_000
        maze_run_name = f"{cifar10_setup.set_id}_maze_paper_soft_{_budget_suffix(maze_budget)}_seed{seed}"
        maze_attack = {
            "name": "maze",
            "output_mode": "soft_prob",
            "max_budget": int(maze_budget),
            "pool_num_workers": int(pool_num_workers),
            "batch_size": 128,
            "n_g_steps": 1,
            "n_c_steps": 5,
            "n_r_steps": 10,
            "grad_approx_m": 10,
            "grad_approx_epsilon": 1e-3,
            "generator_lr": 1e-4,
            "generator_momentum": 0.0,
            "lr_schedule": "cosine",
        }
        maze_substitute = _paper_substitute_config(
            arch="wideresnet22",
            width_mult=2,
            seed=int(seed),
            optimizer_lr=0.1,
            scheduler_name="cosine",
        )
        maze_cfg = _paper_base_config(
            run_name=maze_run_name,
            seed=int(seed),
            victim_arch="resnet20",
            victim_output_mode="soft_prob",
            max_budget=int(maze_budget),
            attack_cfg=maze_attack,
            substitute_cfg=maze_substitute,
        )
        (out_dir / f"{maze_run_name}.yaml").write_text(
            yaml.safe_dump(maze_cfg, sort_keys=False),
            encoding="utf-8",
        )
        count += 1

        # DisGUIDE paper-like profile (CIFAR-10 soft-label):
        # - victim: ResNet-34-8x (paper/default script target)
        # - budget: 20M
        # - batch_size=256, ensemble_size=2
        # - d_iter=1, replay=Classic, rep_iter=3, grayscale=8
        disguide_budget = 20_000_000
        disguide_run_name = (
            f"{cifar10_setup.set_id}_disguide_paper_soft_{_budget_suffix(disguide_budget)}_seed{seed}"
        )
        disguide_attack = {
            "name": "disguide",
            "output_mode": "soft_prob",
            "max_budget": int(disguide_budget),
            "pool_num_workers": int(pool_num_workers),
            "batch_size": 256,
            "g_iter": 1,
            "d_iter": 1,
            "rep_iter": 3,
            "replay": "Classic",
            "replay_size": 1_000_000,
            "loss": "l1",
            "student_lr": 0.03,
            "generator_lr": 1e-4,
            "ensemble_size": 2,
            "grayscale_freq": 8,
            "lambda_div": 0.2,
            "strict_iteration_budget": True,
            "lr_decay_milestones_ratio": [0.4, 0.8],
            "lr_decay_gamma": 0.3,
        }
        disguide_substitute = _paper_substitute_config(
            arch="resnet18",
            seed=int(seed),
            optimizer_lr=0.03,
            width_mult=1,
            scheduler_name="multistep",
        )
        disguide_cfg = _paper_base_config(
            run_name=disguide_run_name,
            seed=int(seed),
            victim_arch="resnet34",
            victim_output_mode="soft_prob",
            max_budget=int(disguide_budget),
            attack_cfg=disguide_attack,
            substitute_cfg=disguide_substitute,
        )
        disguide_cfg["victim"]["victim_id"] = "cifar10_resnet34_8x_official"
        disguide_cfg["victim"]["width_mult"] = 1
        disguide_cfg["victim"]["checkpoint_ref"] = "runs/victims/cifar10-resnet34_8x.pt"
        disguide_cfg["victim"]["return_outputs_on_cpu"] = False

        (out_dir / f"{disguide_run_name}.yaml").write_text(
            yaml.safe_dump(disguide_cfg, sort_keys=False),
            encoding="utf-8",
        )
        count += 1

        # Dual Students paper-like profile (CIFAR-10 soft-label):
        # - victim: ResNet-34-8x official checkpoint
        # - budget: 20M
        # - batch_size=256, num_students=2
        # - g_iter=1, d_iter=5
        ds_budget = 20_000_000
        ds_run_name = f"{cifar10_setup.set_id}_ds_paper_soft_{_budget_suffix(ds_budget)}_seed{seed}"
        ds_attack = {
            "name": "ds",
            "output_mode": "soft_prob",
            "max_budget": int(ds_budget),
            "pool_num_workers": int(pool_num_workers),
            "batch_size": 256,
            "noise_dim": 256,
            "g_iter": 1,
            "d_iter": 5,
            "num_students": 2,
            "loss": "l1",
            "generator_loss": "l1",
            "student_lr": 0.3,
            "generator_lr": 1e-4,
            "strict_iteration_budget": True,
            "lr_decay_milestones_ratio": [0.1, 0.3, 0.5],
            "lr_decay_gamma": 0.3,
        }
        ds_substitute = _paper_substitute_config(
            arch="resnet18",
            seed=int(seed),
            optimizer_lr=0.3,
            width_mult=1,
            scheduler_name="multistep",
        )
        ds_cfg = _paper_base_config(
            run_name=ds_run_name,
            seed=int(seed),
            victim_arch="resnet34",
            victim_output_mode="soft_prob",
            max_budget=int(ds_budget),
            attack_cfg=ds_attack,
            substitute_cfg=ds_substitute,
        )
        ds_cfg["victim"]["victim_id"] = "cifar10_resnet34_8x_official"
        ds_cfg["victim"]["width_mult"] = 1
        ds_cfg["victim"]["checkpoint_ref"] = "runs/victims/cifar10-resnet34_8x.pt"
        ds_cfg["victim"]["return_outputs_on_cpu"] = False

        (out_dir / f"{ds_run_name}.yaml").write_text(
            yaml.safe_dump(ds_cfg, sort_keys=False),
            encoding="utf-8",
        )
        count += 1

        # DFMS-HL paper-like profile (CIFAR-10 hard-label):
        # - budget: 8M
        # - proxy variants: CIFAR100(40 classes), CIFAR100(10 classes)
        # - compare setting with ResNet-34 victim and ResNet-18 clone
        dfms_budget = 8_000_000
        dfms_profiles = [
            {
                "variant": "c100_40c_resnet18",
                "victim_arch": "resnet18",
                "substitute_arch": "resnet18",
                "surrogate_class_subset_size": 40,
            },
            {
                "variant": "c100_10c_resnet18",
                "victim_arch": "resnet18",
                "substitute_arch": "resnet18",
                "surrogate_class_subset_size": 10,
            },
            {
                "variant": "c100_40c_resnet34",
                "victim_arch": "resnet34",
                "substitute_arch": "resnet18",
                "surrogate_class_subset_size": 40,
            },
        ]

        for profile in dfms_profiles:
            run_name = (
                f"{cifar10_setup.set_id}_dfms_paper_{profile['variant']}_hard_"
                f"{_budget_suffix(dfms_budget)}_seed{seed}"
            )
            proxy_cfg = {
                "name": "CIFAR10",
                "data_mode": "surrogate",
                "surrogate_name": "CIFAR100",
                "surrogate_normalization": "none",
                "num_workers": int(pool_num_workers),
                "train_split": True,
                "channels": cifar10_setup.channels,
                "input_size": [cifar10_setup.size, cifar10_setup.size],
                "surrogate_class_subset_size": int(profile["surrogate_class_subset_size"]),
                "surrogate_class_subset_seed": 42,
                "surrogate_max_samples": 100_000,
                "surrogate_subset_seed": 42,
            }
            dfms_attack = {
                "name": "dfms",
                "output_mode": "hard_top1",
                "max_budget": int(dfms_budget),
                "pool_num_workers": int(pool_num_workers),
                "batch_size": 64,
                "oracle_batch_size": 64,
                "use_official_stages": True,
                "dcgan_epochs": 200,
                "student_init_epochs": 200,
                "degan_epochs": 100,
                "student_degan_epochs": 200,
                "alternate_epochs": 800,
                "student_init_lr": 0.1,
                "student_alt_lr": 0.01,
                "clone_lr": 0.01,
                "generator_lr": 2e-4,
                "discriminator_lr": 2e-4,
                "student_batch_size": 128,
                "proxy_data_ratio": 1.0,
                "dcgan_data_ratio": 0.8,
                "div_gan_data_ratio": 0.8,
                "student_pad_crop": True,
                "alternate_pad_crop": False,
                "proxy_dataset": proxy_cfg,
            }
            dfms_substitute = _paper_substitute_config(
                arch=str(profile["substitute_arch"]),
                seed=int(seed),
                optimizer_lr=0.01,
                scheduler_name="multistep",
            )
            dfms_cfg = _paper_base_config(
                run_name=run_name,
                seed=int(seed),
                victim_arch=str(profile["victim_arch"]),
                victim_output_mode="hard_top1",
                max_budget=int(dfms_budget),
                attack_cfg=dfms_attack,
                substitute_cfg=dfms_substitute,
            )
            dfms_cfg["victim"]["return_outputs_on_cpu"] = False

            (out_dir / f"{run_name}.yaml").write_text(
                yaml.safe_dump(dfms_cfg, sort_keys=False),
                encoding="utf-8",
            )
            count += 1

    return count


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate configs for the fairness benchmark runs")
    parser.add_argument("--out", type=str, default="configs/matrix")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--pool-budget",
        type=int,
        default=None,
        help="Global pool budget override for all sets (default: use per-set budgets)",
    )
    parser.add_argument(
        "--synthetic-budget",
        type=int,
        default=None,
        help="Global synthetic budget override for all sets (default: use per-set budgets)",
    )
    parser.add_argument("--set-a-pool-budget", type=int, default=10_000)
    parser.add_argument("--set-a-synthetic-budget", type=int, default=10_000_000)
    parser.add_argument("--set-b-pool-budget", type=int, default=20_000)
    parser.add_argument("--set-b-synthetic-budget", type=int, default=20_000_000)
    parser.add_argument("--set-c-pool-budget", type=int, default=20_000)
    parser.add_argument("--set-c-synthetic-budget", type=int, default=20_000_000)
    parser.add_argument(
        "--imagenet-root",
        type=str,
        default="D:/imagenet",
        help="Local ImageNet root path used as dataset.surrogate_root in generated configs",
    )
    parser.add_argument(
        "--include-both-hard",
        action="store_true",
        help="Generate hard_top1 variants for 'both' attacks (in addition to soft_prob)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete existing *.yaml in output dir before generation",
    )
    parser.add_argument(
        "--pool-num-workers",
        type=int,
        default=8,
        help="Default workers for pool scanning/loaders (dataset/attack pool workers)",
    )
    parser.add_argument(
        "--substitute-num-workers",
        type=int,
        default=4,
        help="Default substitute DataLoader workers (substitute.num_workers)",
    )
    parser.add_argument(
        "--substitute-train-num-workers",
        type=int,
        default=None,
        help="Override substitute train loader workers (defaults to --substitute-num-workers)",
    )
    parser.add_argument(
        "--substitute-val-num-workers",
        type=int,
        default=None,
        help="Override substitute val loader workers (defaults to --substitute-num-workers)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_dir = Path(args.out)
    count = generate_configs(
        out_dir=out_dir,
        device=str(args.device),
        seeds=list(args.seeds),
        pool_budget=(int(args.pool_budget) if args.pool_budget is not None else None),
        synthetic_budget=(int(args.synthetic_budget) if args.synthetic_budget is not None else None),
        set_a_pool_budget=int(args.set_a_pool_budget),
        set_a_synthetic_budget=int(args.set_a_synthetic_budget),
        set_b_pool_budget=int(args.set_b_pool_budget),
        set_b_synthetic_budget=int(args.set_b_synthetic_budget),
        set_c_pool_budget=int(args.set_c_pool_budget),
        set_c_synthetic_budget=int(args.set_c_synthetic_budget),
        include_both_hard=bool(args.include_both_hard),
        clean=(not args.no_clean),
        pool_num_workers=int(args.pool_num_workers),
        substitute_num_workers=int(args.substitute_num_workers),
        substitute_train_num_workers=(
            int(args.substitute_train_num_workers)
            if args.substitute_train_num_workers is not None
            else None
        ),
        substitute_val_num_workers=(
            int(args.substitute_val_num_workers)
            if args.substitute_val_num_workers is not None
            else None
        ),
        imagenet_root=str(args.imagenet_root),
    )
    print(f"Generated {count} configs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
