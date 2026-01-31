"""CLI entry point for mebench."""

import argparse
import sys
import yaml
import logging
from mebench.core.engine import run_experiment


def main():
    """Main CLI entry point."""
    # Configure logging to show INFO level and above to console
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    parser = argparse.ArgumentParser(description="Model Extraction Benchmark")
    parser.add_argument("command", choices=["run"], help="Command to run")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None, help="Device to use")

    args = parser.parse_args()

    if args.command == "run":
        # Load config
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        if args.device is None:
            args.device = config.get("run", {}).get("device", "cpu")

        # Run experiment
        run_experiment(config, args.device)


if __name__ == "__main__":
    main()
