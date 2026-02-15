"""Benchmark engine core."""

from pathlib import Path
from typing import Any, Dict

from mebench.core.context import BenchmarkContext
from mebench.core.logging import ArtifactLogger, create_run_dir, setup_console_logging
from mebench.core.seed import set_seed
from mebench.core.state import BenchmarkState
from mebench.core.validate import validate_config
from mebench.core.query_storage import create_query_storage
from mebench.oracles.oracle import Oracle
from mebench.oracles.victim_loader import load_victim_from_config
from mebench.data.loaders import get_test_dataloader
from mebench.eval.metrics import compute_accuracy
from mebench.eval.unified_pool import train_unified_pool_substitute
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
                "seed": seed,  # [ADDED] Track current seed
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

        # Optional: verify victim accuracy on the public test set.
        # Disabled by default to avoid accidental dataset downloads in CI/unit tests.
        if bool(config.get("benchmark", {}).get("verify_victim_accuracy", False)):
            dataset_name = config.get("dataset", {}).get("name", "CIFAR10")
            victim_cfg = config.get("victim", {})
            input_size = victim_cfg.get("input_size")
            size = None
            if isinstance(input_size, (list, tuple)) and len(input_size) == 2:
                size = (int(input_size[0]), int(input_size[1]))
            channels = victim_cfg.get("channels")

            test_loader = get_test_dataloader(
                dataset_name,
                batch_size=128,
                input_size=size,
                channels=int(channels) if channels is not None else None,
            )
            victim_acc = compute_accuracy(victim, test_loader, device)
            print(f"[VERIFY] Victim Test Accuracy: {victim_acc*100:.2f}%")

        # Initialize oracle
        oracle = Oracle(victim, config["victim"], state)

        # Initialize attack
        attack = create_runner(config["attack"]["name"], config, state)

        # Context (Track B only)
        # For pool/surrogate runs, record oracle queries so we can do a unified
        # from-scratch retrain at the end for fair cross-attack comparison.
        data_mode = str(config.get("dataset", {}).get("data_mode", "")).lower()
        record_queries = data_mode == "surrogate"
        query_storage = None
        if record_queries:
            query_storage = create_query_storage(
                run_dir,
                output_mode=str(config.get("victim", {}).get("output_mode", "soft_prob")),
            )
        ctx = BenchmarkContext(
            state=state,
            oracle=oracle,
            logger=logger,
            config=config,
            query_storage=query_storage,
            record_queries=record_queries,
        )
        
        # [ADDED] Inject context into attack runner for metric logging
        attack.ctx = ctx

        print("\nStarting attack run (Track B only)")
        attack.run(ctx)

        # Unified pool evaluation: train a substitute from scratch using the
        # collected transfer set with a fixed train/val split + early stopping.
        if record_queries and query_storage is not None and int(len(query_storage)) > 0:
            substitute_unified = train_unified_pool_substitute(
                config=config,
                state=state,
                query_dataset=query_storage,
                device=device,
            )
            if substitute_unified is not None:
                state.attack_state["substitute_attack"] = state.attack_state.get("substitute")
                state.attack_state["substitute_unified"] = substitute_unified
                state.attack_state["substitute"] = substitute_unified
 
        
        # FINAL EVALUATION for Track B
        # Skip evaluation for placeholder victims used by unit tests.
        victim_ref = str(config.get("victim", {}).get("checkpoint_ref") or "")
        is_placeholder_victim = victim_ref == "" or victim_ref == "/path/to/ckpt.pt"
        substitute = state.attack_state.get("substitute")
        if substitute is not None and not is_placeholder_victim:
            # Ensure victim is set in case run() finished early or skipped it
            if attack.victim is None:
                attack.victim = victim
            track = "track_b_unified" if record_queries else "track_b"
            qc = int(len(query_storage)) if (record_queries and query_storage is not None) else None
            attack._evaluate_current_substitute(substitute, device, track=track, query_count=qc)

        # Cleanup temporary query cache if configured.
        cache_cfg = config.get("cache", {})
        if record_queries and query_storage is not None:
            if bool(cache_cfg.get("enabled", False)) and bool(cache_cfg.get("delete_on_finish", False)):
                query_storage.cleanup()

        print("\nAttack run complete!")

        # Finalize logging
        logger.finalize()

    print(f"\n{'='*60}")
    print("Experiment completed!")
    print(f"{'='*60}")
