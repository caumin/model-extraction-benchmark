"""Generate 1k-budget matrix configs for quick validation."""

from pathlib import Path

import yaml


def generate_matrix_1k() -> None:
    input_dir = Path("configs/matrix")
    output_dir = Path("configs/matrix_1k")
    output_dir.mkdir(parents=True, exist_ok=True)

    for config_path in input_dir.glob("*.yaml"):
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        config.setdefault("budget", {})
        config["budget"]["max_budget"] = 1000
        config["budget"]["checkpoints"] = [1000]

        if "attack" in config:
            config["attack"]["max_budget"] = 1000

        victim_cfg = config.setdefault("victim", {})
        # victim_cfg["checkpoint_ref"] = "runs/victims/mnist_lenet_seed0.pt"

        run_cfg = config.setdefault("run", {})
        run_name = str(run_cfg.get("name", config_path.stem))
        if not run_name.endswith("_1k"):
            run_cfg["name"] = f"{run_name}_1k"

        output_name = f"{config_path.stem}_1k.yaml"
        output_path = output_dir / output_name
        with output_path.open("w", encoding="utf-8") as handle:
            yaml.dump(config, handle, sort_keys=False)

    print(f"Generated 1k configs in {output_dir}")


if __name__ == "__main__":
    generate_matrix_1k()
