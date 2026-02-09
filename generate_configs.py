from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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
    # Monitoring-only checkpoints (not used for cross-attack comparison).
    if max_budget <= 20_000:
        return [5_000, 10_000, max_budget]
    if max_budget <= 100_000:
        return [10_000, 50_000, max_budget]
    return [100_000, 500_000, 1_000_000, max_budget]


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


def generate_configs(
    out_dir: Path,
    device: str,
    seeds: List[int],
    pool_budget: int,
    synthetic_budget: int,
    include_both_hard: bool,
    clean: bool,
) -> int:
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
            victim_arch="resnet18",
            surrogate_name="ImageNet",
            substitute_arch="resnet18",
            channels=3,
            size=32,
            num_classes=10,
        ),
    ]

    # Attack taxonomy used for a single-run-per-attack evaluation.
    # - pool: selection from a public image pool (surrogate)
    # - synthetic: query synthesis / data-free generation
    attacks: List[AttackSpec] = [
        AttackSpec("random", kind="pool", label_capability="both"),
        AttackSpec("knockoff_nets", kind="pool", label_capability="soft_only"),
        AttackSpec("cloudleak", kind="pool", label_capability="soft_only"),
        AttackSpec("copycatcnn", kind="pool", label_capability="hard_only"),
        AttackSpec("inversenet", kind="pool", label_capability="hard_only"),
        AttackSpec("blackbox_dissector", kind="pool", label_capability="hard_only"),

        AttackSpec(
            "activethief",
            kind="pool",
            label_capability="both",
            extra={
                "strategy": "dfal_k_center", 
                "batch_size": 150,
                "substitute": {
                    "batch_size": 150,
                    "optimizer": {"name": "adam", "lr": 0.001, "weight_decay": 5e-4}
                }
            },
        ),
        AttackSpec(
            "activethief",
            kind="pool",
            label_capability="both",
            extra={
                "strategy": "uncertainty", 
                "variant": "uncertainty",
                "batch_size": 150,
                "substitute": {
                    "batch_size": 150,
                    "optimizer": {"name": "adam", "lr": 0.001, "weight_decay": 5e-4}
                }
            },
        ),
        AttackSpec(
            "activethief",
            kind="pool",
            label_capability="both",
            extra={
                "strategy": "dfal", 
                "variant": "dfal",
                "batch_size": 150,
                "substitute": {
                    "batch_size": 150,
                    "optimizer": {"name": "adam", "lr": 0.001, "weight_decay": 5e-4}
                }
            },
        ),
        AttackSpec("swiftthief", kind="pool", label_capability="both"),

        # [UNIFIED] All data-free attacks standardized to:
        # - SGD Student (LR via setup)
        # - Adam Generator (MAZE changed from SGD)
        # - MultiStepLR (Handled in code now, not config)
        
        AttackSpec("dfme", kind="synthetic", label_capability="soft_only"),
        AttackSpec("maze", kind="synthetic", label_capability="soft_only"),
        AttackSpec("game", kind="synthetic", label_capability="soft_only"),
        AttackSpec("blackbox_ripper", kind="synthetic", label_capability="soft_only"),
        AttackSpec("dfms", kind="synthetic", label_capability="hard_only"),
        AttackSpec("es", kind="synthetic", label_capability="both"),
    ]

    if clean:
        _clean_yaml_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    protocol_version = "1.2"
    substitute_batch_size = 256

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

    count = 0
    for setup in setups:
        for attack in attacks:
            if attack.kind == "synthetic":
                victim_key = _victim_id(setup)
                canonical = synthetic_canonical_by_victim.get(victim_key)
                if canonical is not None and setup.set_id != canonical.set_id:
                    continue
            max_budget = _budget_for_kind(attack.kind, pool_budget=pool_budget, synthetic_budget=synthetic_budget)
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
                    
                    # [LR Logic]
                    # SET-A (MNIST/LeNet): 0.01
                    # SET-B (CIFAR10/ResNet): 0.1
                    # Default: 0.1 (Data-Free Standard) if undetermined
                    if "SET-A" in setup.set_id:
                        target_lr = 0.01
                    elif "SET-B" in setup.set_id:
                        target_lr = 0.1
                    else:
                        target_lr = 0.1

                    # Default substitute config
                    substitute_config = {
                        "arch": setup.substitute_arch,
                        "init_seed": 1234 + seed,
                        "batch_size": substitute_batch_size,
                        "trackA": {"batch_size": substitute_batch_size, "steps_coeff_c": 0.2},
                        "optimizer": {
                            "name": "sgd",
                            "lr": target_lr,
                            "momentum": 0.9,
                            "weight_decay": 5e-4,
                        },
                        "max_epochs": 1000,
                        "patience": 100,
                    }
                    
                    # Merge override recursively (simple 1-level merge for optimizer)
                    if substitute_override:
                        for k, v in substitute_override.items():
                            if k == "optimizer" and isinstance(v, dict) and "optimizer" in substitute_config:
                                substitute_config["optimizer"].update(v)
                            else:
                                substitute_config[k] = v

                    cfg: Dict[str, Any] = {
                        "run": {"name": run_name, "seeds": [seed], "device": device},
                        "benchmark": {"protocol_version": protocol_version},
                        "victim": {
                            "victim_id": victim_id,
                            "arch": setup.victim_arch,
                            "channels": setup.channels,
                            "num_classes": setup.num_classes,
                            "input_size": [setup.size, setup.size],
                            "checkpoint_ref": f"runs/victims/{victim_id}_seed0.pt",
                            "normalization": None,
                            "output_mode": output_mode,
                            "temperature": 1.0,
                        },
                        "dataset": {
                            "name": setup.victim_dataset,
                            "data_mode": data_mode,
                            "surrogate_name": setup.surrogate_name,
                            "train_split": True,
                            "channels": setup.channels,
                            "input_size": [setup.size, setup.size],
                        },
                        "attack": {
                            "name": attack.name,
                            "output_mode": output_mode,
                            "max_budget": max_budget,
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

                    def _maybe_add_imagenet_imagefolder_keys(d: Dict[str, Any]) -> None:
                        if str(d.get("data_mode")).lower() != "surrogate":
                            return
                        if str(d.get("surrogate_name")).lower() != "imagenet":
                            return
                        # ImageNet surrogate (ImageFolder format): use a deterministic 100k subset.
                        # Local path should be provided by the user later via `surrogate_root`
                        # or env var `MEBENCH_IMAGENET_ROOT`.
                        d.update(
                            {
                                "surrogate_root": "<FILL_ME>",
                                "surrogate_resize": [setup.size, setup.size],
                                "surrogate_max_samples": 100_000,
                                "surrogate_subset_seed": 42,
                            }
                        )

                    _maybe_add_imagenet_imagefolder_keys(cfg["dataset"])

                    if attack.name == "knockoff_nets":
                        cfg["attack"]["offline_train_epochs"] = cfg["substitute"]["max_epochs"]

                    if attack.name in {"blackbox_ripper", "game", "dfms"}:
                        cfg["attack"]["proxy_dataset"] = {
                            "name": setup.surrogate_name,
                            "data_mode": "surrogate",
                            "surrogate_name": setup.surrogate_name,
                            "train_split": True,
                            "channels": setup.channels,
                            "input_size": [setup.size, setup.size],
                        }
                        _maybe_add_imagenet_imagefolder_keys(cfg["attack"]["proxy_dataset"])

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


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate configs for the fairness benchmark runs")
    parser.add_argument("--out", type=str, default="configs/matrix")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--pool-budget", type=int, default=20_000)
    parser.add_argument("--synthetic-budget", type=int, default=20_000_000)
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
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_dir = Path(args.out)
    count = generate_configs(
        out_dir=out_dir,
        device=str(args.device),
        seeds=list(args.seeds),
        pool_budget=int(args.pool_budget),
        synthetic_budget=int(args.synthetic_budget),
        include_both_hard=bool(args.include_both_hard),
        clean=(not args.no_clean),
    )
    print(f"Generated {count} configs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
