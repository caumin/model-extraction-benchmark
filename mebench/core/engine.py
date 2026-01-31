"""Benchmark engine core."""

from pathlib import Path
from typing import Any, Dict

from mebench.core.context import BenchmarkContext
from mebench.core.logging import ArtifactLogger, create_run_dir, setup_console_logging
from mebench.core.seed import set_seed
from mebench.core.state import BenchmarkState
from mebench.core.validate import validate_config
from mebench.oracles.oracle import Oracle
from mebench.oracles.victim_loader import load_victim_from_config
from mebench.attackers.runner import AttackRunner
from mebench.attackers.activethief import ActiveThief
from mebench.attackers.blackbox_dissector import BlackboxDissector
from mebench.attackers.blackbox_ripper import BlackboxRipper
from mebench.attackers.cloudleak import CloudLeak
from mebench.attackers.copycatcnn import CopycatCNN
from mebench.attackers.dfme import DFME
from mebench.attackers.dfms import DFMSHL
from mebench.attackers.es_attack import ESAttack
from mebench.attackers.game import GAME
from mebench.attackers.inversenet import InverseNet
from mebench.attackers.knockoff_nets import KnockoffNets
from mebench.attackers.maze import MAZE
from mebench.attackers.random_baseline import RandomBaseline
from mebench.attackers.swiftthief import SwiftThief


def create_runner(
    attack_name: str,
    config: Dict[str, Any],
    state: BenchmarkState,
) -> AttackRunner:
    """Create attack instance from name.

    Args:
        attack_name: Attack name (random, activethief, dfme)
        config: Attack configuration
        state: Global benchmark state

    Returns:
        Attack runner instance
    """
    if attack_name == "activethief":
        return ActiveThief(config["attack"], state)
    elif attack_name == "dfme":
        return DFME(config["attack"], state)
    elif attack_name == "maze":
        return MAZE(config["attack"], state)
    elif attack_name == "dfms":
        return DFMSHL(config["attack"], state)
    elif attack_name == "game":
        return GAME(config["attack"], state)
    elif attack_name == "es":
        return ESAttack(config["attack"], state)
    elif attack_name == "random":
        return RandomBaseline(config["attack"], state)
    elif attack_name == "swiftthief":
        return SwiftThief(config["attack"], state)
    elif attack_name == "blackbox_dissector":
        return BlackboxDissector(config["attack"], state)
    elif attack_name == "cloudleak":
        return CloudLeak(config["attack"], state)
    elif attack_name == "blackbox_ripper":
        return BlackboxRipper(config["attack"], state)
    elif attack_name == "copycatcnn":
        return CopycatCNN(config["attack"], state)
    elif attack_name == "inversenet":
        return InverseNet(config["attack"], state)
    elif attack_name == "knockoff_nets":
        return KnockoffNets(config["attack"], state)
    else:
        raise ValueError(f"Unknown attack: {attack_name}")


def run_experiment(
    config: Dict[str, Any],
    device: str = "cpu",
) -> None:
    """Run benchmark experiment.

    Args:
        config: Experiment configuration
        device: Device to use
    """
    # Validate config
    validate_config(config)

    # Setup logging
    setup_console_logging()

    # Run for each seed
    for seed in config["run"]["seeds"]:
        print(f"\n{'='*60}")
        print(f"Running seed {seed}")
        print(f"{'='*60}")

        # Set seed for reproducibility
        set_seed(seed)

        # Create run directory
        base_dir = Path("runs")
        run_dir = create_run_dir(base_dir, config["run"]["name"], seed)
        print(f"Run directory: {run_dir}")

        # Initialize logger
        logger = ArtifactLogger(run_dir)
        logger.set_run_metadata(config)
        logger.save_config(config)

        # Initialize state
        state = BenchmarkState(
            budget_remaining=config["budget"]["max_budget"],
            metadata={
                "device": device,
                "input_shape": (
                    int(config["victim"]["channels"]),
                    *config["victim"].get("input_size", [32, 32]),
                ),
                "dataset_config": config.get("dataset", {}),
                "substitute_config": config.get("substitute", {}),
                "victim_config": config.get("victim", {}),
                "max_budget": config["budget"]["max_budget"],
            },
        )

        # Load victim model from checkpoint or placeholder
        victim = load_victim_from_config(config["victim"], device)

        # Initialize oracle
        oracle = Oracle(victim, config["victim"], state)

        # Initialize attack
        attack = create_runner(config["attack"]["name"], config, state)

        # Context (Track B only)
        ctx = BenchmarkContext(state=state, oracle=oracle, logger=logger, config=config)

        print("\nStarting attack run (Track B only)")
        attack.run(ctx)
        
        # FINAL EVALUATION for Track B
        substitute = state.attack_state.get("substitute")
        if substitute is not None:
            # Ensure victim is set in case run() finished early or skipped it
            if attack.victim is None:
                attack.victim = victim
            attack._evaluate_current_substitute(substitute, device)

        print("\nAttack run complete!")

        # Finalize logging
        logger.finalize()

    print(f"\n{'='*60}")
    print("Experiment completed!")
    print(f"{'='*60}")
