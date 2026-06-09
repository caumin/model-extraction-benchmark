"""Benchmark engine core."""

import logging
from pathlib import Path
from typing import Any, Dict

from mebench.core.context import BenchmarkContext
from mebench.core.logging import ArtifactLogger, create_run_dir, setup_console_logging
from mebench.core.seed import set_seed
from mebench.core.state import BenchmarkState
from mebench.core.validate import validate_config
from mebench.oracles.oracle import Oracle
from mebench.oracles.victim_loader import load_victim_from_config
from mebench.data.loaders import get_test_dataloader
from mebench.eval.metrics import compute_accuracy
from mebench.attackers.runner import AttackRunner
from mebench.attackers.activethief import ActiveThief
from mebench.attackers.blackbox_dissector import BlackboxDissector
from mebench.attackers.blackbox_ripper import BlackboxRipper
from mebench.attackers.marich import MARICH
from mebench.attackers.cloudleak import CloudLeak
from mebench.attackers.copycatcnn import CopycatCNN
from mebench.attackers.dfme import DFME
from mebench.attackers.ds import DualStudents
from mebench.attackers.dfms import DFMSHL
from mebench.attackers.disguide import DisGUIDE
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
    elif attack_name == "ds":
        return DualStudents(config["attack"], state)
    elif attack_name == "dfms":
        return DFMSHL(config["attack"], state)
    elif attack_name == "disguide":
        return DisGUIDE(config["attack"], state)
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
    elif attack_name == "marich":
        return MARICH(config["attack"], state)
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
    logger = logging.getLogger(__name__)

    # Run for each seed
    for seed in config["run"]["seeds"]:
        logger.info("\n%s", "=" * 60)
        logger.info("Running seed %s", seed)
        logger.info("%s", "=" * 60)

        # Set seed for reproducibility
        set_seed(seed)

        # Create run directory
        base_dir = Path("runs")
        run_dir = create_run_dir(base_dir, config["run"]["name"], seed)

        # Initialize logger
        artifact_logger = ArtifactLogger(run_dir)
        try:
            artifact_logger.set_run_metadata(config)
            artifact_logger.save_config(config)
            logger.info("Run directory: %s", run_dir)
            logger.info("Experiment log file: %s", artifact_logger.log_path)

            # Initialize state
            state = BenchmarkState(
                budget_remaining=config["budget"]["max_budget"],
                metadata={
                    "seed": seed,  # [ADDED] Track current seed
                    "device": device,
                    "benchmark_config": config.get("benchmark", {}),
                    "num_classes": int(config["victim"]["num_classes"]),
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
            artifact_logger.log_resource_snapshot(0, "seed_setup", device=device, reset_peak=True)

            # Load victim model from checkpoint or placeholder
            victim = load_victim_from_config(config["victim"], device)
            victim_ref = str(config.get("victim", {}).get("checkpoint_ref") or "")
            is_placeholder_victim = victim_ref == "" or victim_ref == "/path/to/ckpt.pt"
            artifact_logger.log_resource_snapshot(0, "victim_loaded", device=device)

            # Optional: verify victim accuracy on the public test set.
            # Disabled by default to avoid accidental dataset downloads in CI/unit tests.
            if bool(config.get("benchmark", {}).get("verify_victim_accuracy", True)):
                dataset_name = config.get("dataset", {}).get("name", "CIFAR10")
                victim_cfg = config.get("victim", {})
                input_size = victim_cfg.get("input_size")
                size = None
                if isinstance(input_size, (list, tuple)) and len(input_size) == 2:
                    size = (int(input_size[0]), int(input_size[1]))
                channels = victim_cfg.get("channels")
                eval_batch_size = max(1, int(config.get("benchmark", {}).get("eval_batch_size", 128)))

                test_loader = get_test_dataloader(
                    dataset_name,
                    batch_size=eval_batch_size,
                    input_size=size,
                    channels=int(channels) if channels is not None else None,
                    sewerml_label_mode=str(config.get("dataset", {}).get("sewerml_label_mode", "argmax")),
                    sewerml_ann_root=config.get("dataset", {}).get("sewerml_ann_root"),
                    sewerml_data_root=config.get("dataset", {}).get("sewerml_data_root"),
                    sewerml_eval_split=config.get("dataset", {}).get("sewerml_eval_split"),
                    sewerml_max_samples=int(config.get("dataset", {}).get("sewerml_max_samples", 0)),
                    sewerml_subset_seed=int(config.get("dataset", {}).get("sewerml_subset_seed", 42)),
                )
                victim_acc = compute_accuracy(victim, test_loader, device)
                logger.info("[VERIFY] Victim Test Accuracy: %.2f%%", victim_acc * 100.0)

            # Initialize oracle
            oracle = Oracle(victim, config["victim"], state)

            # Initialize attack
            attack = create_runner(config["attack"]["name"], config, state)

            ctx = BenchmarkContext(
                state=state,
                oracle=oracle,
                logger=artifact_logger,
                config=config,
                checkpoint_callback=None,
            )

            # [ADDED] Inject context into attack runner for metric logging
            attack.ctx = ctx

            logger.info("Starting attack run (Track B only)")
            ctx.log_resource_snapshot("attack_run_start", {"seed": int(seed)}, reset_peak=True)
            attack.run(ctx)
            ctx.log_resource_snapshot("attack_run_end", {"seed": int(seed)})

            # FINAL EVALUATION for Track B
            substitute = state.attack_state.get("substitute")
            if substitute is not None and not is_placeholder_victim:
                # Ensure victim is attached for metric computation.
                if attack.victim is None:
                    attack.victim = victim
                attack._drain_deferred_track_b_checkpoints(device)
                attack._evaluate_current_substitute(
                    substitute,
                    device,
                    track="track_b",
                    query_count=state.query_count,
                )

            logger.info("Attack run complete")

            # Finalize logging
            artifact_logger.finalize()
        except KeyboardInterrupt:
            artifact_logger.log_resource_snapshot(0, "seed_interrupted", device=device, payload={"seed": int(seed)})
            logger.exception("Seed %s interrupted", seed)
            raise
        except Exception:
            artifact_logger.log_resource_snapshot(0, "seed_failed", device=device, payload={"seed": int(seed)})
            logger.exception("Seed %s failed", seed)
            raise
        finally:
            artifact_logger.close()

    logger.info("\n%s", "=" * 60)
    logger.info("Experiment completed!")
    logger.info("%s", "=" * 60)
