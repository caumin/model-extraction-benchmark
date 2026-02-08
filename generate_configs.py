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
    arch_clean = setup.victim_arch
    if arch_clean == "lenet_mnist":
        arch_clean = "lenet"
    return f"{setup.victim_dataset.lower()}_{arch_clean}"


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
            surrogate_name="EMNIST",
            substitute_arch="lenet_mnist",
            channels=1,
            size=28,
            num_classes=10,
        ),
        Setup(
            set_id="SET-A2",
            victim_dataset="MNIST",
            victim_arch="lenet_mnist",
            surrogate_name="FashionMNIST",
            substitute_arch="lenet_mnist",
            channels=1,
            size=28,
            num_classes=10,
        ),
        Setup(
            set_id="SET-B1",
            victim_dataset="CIFAR10",
            victim_arch="resnet18",
            surrogate_name="SVHN",
            substitute_arch="resnet18",
            channels=3,
            size=32,
            num_classes=10,
        ),
        Setup(
            set_id="SET-B2",
            victim_dataset="CIFAR10",
            victim_arch="resnet18",
            surrogate_name="GTSRB",
            substitute_arch="resnet18",
            channels=3,
            size=32,
            num_classes=10,
        ),
        Setup(
            set_id="SET-B3",
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
            extra={"strategy": "dfal_k_center"},
        ),
        AttackSpec(
            "activethief",
            kind="pool",
            label_capability="both",
            extra={"strategy": "uncertainty", "variant": "uncertainty"},
        ),
        AttackSpec(
            "activethief",
            kind="pool",
            label_capability="both",
            extra={"strategy": "dfal", "variant": "dfal"},
        ),
        AttackSpec("swiftthief", kind="pool", label_capability="both"),

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

    count = 0
    for setup in setups:
        for attack in attacks:
            max_budget = _budget_for_kind(attack.kind, pool_budget=pool_budget, synthetic_budget=synthetic_budget)
            checkpoints = _checkpoints_for_budget(max_budget)

            victim_id = _victim_id(setup)
            data_mode = "data_free" if attack.kind == "synthetic" else "surrogate"

            for output_mode in _attack_output_modes(attack.label_capability, include_both_hard=include_both_hard):
                for seed in seeds:
                    suffix_mode = "soft" if output_mode == "soft_prob" else "hard"
                    suffix_budget = "20k" if attack.kind == "pool" else "2m"

                    attack_variant = None
                    attack_extra = dict(attack.extra or {})
                    if "variant" in attack_extra:
                        attack_variant = str(attack_extra.pop("variant"))

                    attack_name_for_filename = attack.name
                    if attack_variant:
                        attack_name_for_filename = f"{attack.name}_{attack_variant}"

                    run_name = f"{setup.set_id}_{attack_name_for_filename}_{suffix_mode}_{suffix_budget}_seed{seed}"
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
                        },
                        "attack": {
                            "name": attack.name,
                            "output_mode": output_mode,
                            "max_budget": max_budget,
                            **attack_extra,
                        },
                        "substitute": {
                            "arch": setup.substitute_arch,
                            "init_seed": 1234 + seed,
                            "batch_size": substitute_batch_size,
                            "trackA": {"batch_size": substitute_batch_size, "steps_coeff_c": 0.2},
                            "optimizer": {
                                "name": "sgd",
                                "lr": 0.01,
                                "momentum": 0.9,
                                "weight_decay": 5e-4,
                            },
                            "max_epochs": 1000,
                            "patience": 100,
                        },
                        "budget": {"max_budget": max_budget, "checkpoints": checkpoints},
                        "cache": {
                            "enabled": True,
                            "policy": "temporary",
                            "delete_on_finish": True,
                        },
                    }

                    # ImageNet surrogate (ImageFolder format): use a deterministic 100k subset.
                    # Local path should be provided by the user later via `dataset.surrogate_root`
                    # or env var `MEBENCH_IMAGENET_ROOT`.
                    if setup.surrogate_name == "ImageNet":
                        cfg["dataset"].update(
                            {
                                "surrogate_root": "<FILL_ME>",
                                "surrogate_resize": [setup.size, setup.size],
                                "surrogate_max_samples": 100_000,
                                "surrogate_subset_seed": 42,
                            }
                        )

                    if attack.name == "knockoff_nets":
                        cfg["attack"]["offline_train_epochs"] = cfg["substitute"]["max_epochs"]

                    if attack.name in {"blackbox_ripper", "game", "dfms"}:
                        cfg["attack"]["proxy_dataset"] = {
                            "name": setup.surrogate_name,
                            "data_mode": "surrogate",
                            "surrogate_name": setup.surrogate_name,
                            "train_split": True,
                        }

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
    parser.add_argument("--synthetic-budget", type=int, default=2_000_000)
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
