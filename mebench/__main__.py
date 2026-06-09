"""CLI entry point for mebench.

Usage:
    python -m mebench run --config <YAML> [--device <device>]

Examples:
    python -m mebench run --config configs/smoke/dfme_mnist_200.yaml
    python -m mebench run --config configs/matrix/SET-A1_random_hard_10k_seed0.yaml \
        --device cuda:0

See `configs/matrix/` for full SET-A/B/C profiles produced by
`generate_configs.py`. The contract these YAMLs follow is described in
`Model_Extraction_Benchmark_v1.0.1_Contract_and_Implementation_Guide.md`
and `THREAT_MODEL.md`.
"""

import argparse
import sys
import yaml

from mebench.core.engine import run_experiment


_EXAMPLE_CONFIG = """\
# Minimal mebench config (random baseline on CIFAR-10).
# See configs/matrix/*.yaml for fully-specified SET-A/B/C profiles.
run:
  name: random_baseline_cifar10_smoke
  seeds: [0]
  device: cuda:0

benchmark:
  protocol_version: '1.3'

victim:
  victim_id: cifar10_resnet18
  arch: resnet18
  channels: 3
  num_classes: 10
  input_size: [32, 32]
  checkpoint_ref: runs/victims/cifar10_resnet18_seed0.pt
  output_mode: soft_prob
  temperature: 1.0

dataset:
  name: CIFAR10
  data_mode: surrogate
  surrogate_name: ImageNet
  surrogate_root: ${IMAGENET_ROOT}
  surrogate_resize: [32, 32]
  surrogate_max_samples: 10000
  channels: 3
  input_size: [32, 32]

attack:
  name: random_baseline
  output_mode: soft_prob

substitute:
  arch: resnet18

budget:
  max_budget: 1000
  checkpoints: [200, 500, 1000]
"""


def main():
    """Main CLI entry point."""
    from mebench.core.logging import setup_console_logging
    setup_console_logging()

    parser = argparse.ArgumentParser(
        prog="python -m mebench",
        description="Model Extraction Benchmark — run a single attack from a YAML config.",
        epilog=(
            "Examples in configs/matrix/. See README.md, THREAT_MODEL.md, "
            "and docs/METRICS.md for the contract and metric definitions."
        ),
    )
    parser.add_argument("command", choices=["run"], help="Command to run")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (required unless --example-config is given)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, e.g. cuda:0 or cpu (defaults to run.device in config)",
    )
    parser.add_argument(
        "--example-config",
        action="store_true",
        help="Print a minimal working YAML to stdout and exit",
    )

    args = parser.parse_args()

    if args.example_config:
        sys.stdout.write(_EXAMPLE_CONFIG)
        return

    if args.command == "run":
        if args.config is None:
            parser.error("--config is required for 'run' (or pass --example-config)")
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        if args.device is None:
            args.device = config.get("run", {}).get("device", "cpu")

        run_experiment(config, args.device)


if __name__ == "__main__":
    main()
