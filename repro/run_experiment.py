"""Reproduction pipeline runner for paper folders under repro/papers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent.parent
REPRO_ROOT = ROOT / "repro"
PAPERS_ROOT = REPRO_ROOT / "papers"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _ensure_file(path: Path, content: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _bootstrap_paper(paper_id: str, pdf_path: str) -> None:
    paper_dir = PAPERS_ROOT / paper_id
    (paper_dir / "configs").mkdir(parents=True, exist_ok=True)
    (paper_dir / "logs").mkdir(parents=True, exist_ok=True)
    (paper_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (paper_dir / "results" / "plots").mkdir(parents=True, exist_ok=True)

    _ensure_file(
        paper_dir / "paper_meta.yaml",
        (
            f'title: ""\n'
            "authors: []\n"
            'venue: ""\n'
            "year: null\n"
            f"paper_id: {paper_id}\n"
            f"pdf_file: {pdf_path}\n"
            'sha256: ""\n'
        ),
    )

    _ensure_file(
        paper_dir / "extracted_spec.yaml",
        (
            "paper:\n"
            f"  paper_id: {paper_id}\n"
            '  title: ""\n'
            "  authors: []\n"
            "  year: null\n"
            '  venue: ""\n'
            "pdf:\n"
            f"  path: /{pdf_path}\n"
            '  sha256: ""\n'
            "datasets: []\n"
            "victim: {}\n"
            "attack: {}\n"
            "evaluation: {}\n"
            "reported_results:\n"
            "  tables: []\n"
            "  figures: []\n"
            "assumptions: []\n"
            "evidence_index: []\n"
        ),
    )

    _ensure_file(
        paper_dir / "evidence.md",
        (
            "# Evidence Log\n\n"
            "- item_path: ``\n"
            "- pdf: ``\n"
            "- page: ``\n"
            "- ref: ``\n"
            "- quote: ``\n"
            "- interpretation: ``\n"
        ),
    )

    _ensure_file(
        paper_dir / "mapping.md",
        "# Paper-to-Code Mapping\n\n| Paper item | Paper value | Code target | Mapping |\n|---|---|---|---|\n",
    )

    _ensure_file(
        paper_dir / "REPRODUCTION_REPORT.md",
        (
            "# REPRODUCTION_REPORT\n\n"
            "## Scope\n\n"
            f"- paper_id: `{paper_id}`\n"
            "- status: pending\n"
        ),
    )

    _ensure_file(paper_dir / "configs" / "victim_train.yaml", "# TODO\n")
    _ensure_file(paper_dir / "configs" / "victim_eval.yaml", "# TODO\n")
    _ensure_file(paper_dir / "configs" / "attack.yaml", "# TODO\n")
    _ensure_file(paper_dir / "configs" / "experiment.yaml", "# TODO\n")
    _ensure_file(
        paper_dir / "results" / "reproduced_metrics.csv",
        "paper_id,run_id,seed,query_budget,oracle_type,metric_name,metric_value,track,timestamp\n",
    )
    _ensure_file(paper_dir / "results" / "reproduced_metrics.json", "[]\n")
    _ensure_file(
        paper_dir / "results" / "comparison_table.md",
        "# Comparison Table\n\n| condition | paper_value | reproduced_mean | reproduced_std | delta_abs_pp | verdict |\n|---|---:|---:|---:|---:|---|\n",
    )


def bootstrap_from_index(index_path: Path) -> None:
    cfg = _load_yaml(index_path)
    papers = cfg.get("papers", [])
    if not isinstance(papers, list):
        raise ValueError(f"`papers` must be a list in: {index_path}")
    for item in papers:
        if not isinstance(item, dict):
            continue
        paper_id = str(item.get("paper_id", "")).strip()
        pdf = str(item.get("pdf", "")).strip()
        if paper_id == "" or pdf == "":
            continue
        _bootstrap_paper(paper_id, pdf)


def _run_command(cmd: list[str], log_path: Path, dry_run: bool) -> None:
    shown = shlex.join(cmd)
    print(f"$ {shown}")
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n$ {shown}\n")

    if dry_run:
        return

    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    with log_path.open("a", encoding="utf-8") as logf:
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed (exit {rc}): {shown}")


def _find_latest_seed_dir(run_name: str, seed: int) -> Path:
    base = ROOT / "runs" / run_name
    if not base.exists():
        raise FileNotFoundError(f"Run directory not found: {base}")

    candidates = [p for p in base.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No timestamp directories under: {base}")

    candidates.sort(key=lambda p: p.name, reverse=True)
    seed_dir_name = f"seed_{int(seed)}"
    for ts_dir in candidates:
        seed_dir = ts_dir / seed_dir_name
        if (seed_dir / "metrics.csv").exists():
            return seed_dir
    raise FileNotFoundError(f"No metrics found for seed {seed} under {base}")


def _collect_reproduced_metrics(
    paper_id: str,
    paper_dir: Path,
    experiment_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    run_name = str(experiment_cfg.get("run", {}).get("name", "")).strip()
    seeds = experiment_cfg.get("run", {}).get("seeds", [])
    output_mode = str(experiment_cfg.get("victim", {}).get("output_mode", ""))
    if run_name == "":
        raise ValueError("experiment.yaml must define run.name")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("experiment.yaml must define non-empty run.seeds")

    rows: list[dict[str, Any]] = []
    now = datetime.now().isoformat()
    for seed in seeds:
        seed_int = int(seed)
        seed_dir = _find_latest_seed_dir(run_name, seed_int)
        run_id = f"{run_name}/{seed_dir.parent.name}"
        metrics_path = seed_dir / "metrics.csv"
        with metrics_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                q = int(raw.get("checkpoint_B", 0))
                track = str(raw.get("track", ""))
                for metric_name in ("acc_gt", "agreement", "kl_mean", "l1_mean"):
                    v = raw.get(metric_name)
                    if v is None or str(v).strip() == "":
                        continue
                    value = float(v)
                    rows.append(
                        {
                            "paper_id": paper_id,
                            "run_id": run_id,
                            "seed": seed_int,
                            "query_budget": q,
                            "oracle_type": output_mode,
                            "metric_name": metric_name,
                            "metric_value": value,
                            "track": track,
                            "timestamp": now,
                        }
                    )

    results_dir = paper_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "reproduced_metrics.csv"
    json_path = results_dir / "reproduced_metrics.json"

    fields = [
        "paper_id",
        "run_id",
        "seed",
        "query_budget",
        "oracle_type",
        "metric_name",
        "metric_value",
        "track",
        "timestamp",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _safe_mean(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def _build_comparison_table(
    paper_id: str,
    paper_dir: Path,
    experiment_cfg: dict[str, Any],
    reproduced_rows: list[dict[str, Any]],
) -> None:
    spec = _load_yaml(paper_dir / "extracted_spec.yaml")
    comparison_path = paper_dir / "results" / "comparison_table.md"
    targets = spec.get("reported_results", {}).get("targets", [])
    if not isinstance(targets, list) or len(targets) == 0:
        comparison_path.write_text(
            "# Comparison Table\n\n- status: INCOMPLETE\n- reason: `reported_results.targets` is missing or empty in extracted_spec.yaml\n",
            encoding="utf-8",
        )
        return

    table_rows: list[str] = []
    missing_reasons: list[str] = []

    for target in targets:
        if not isinstance(target, dict):
            continue
        condition = str(target.get("condition", target.get("id", "target")))
        metric_name = str(target.get("metric_name", "agreement"))
        query_budget = int(target.get("query_budget", 0))
        paper_value = float(target.get("paper_value"))
        track = target.get("track", None)
        tolerance_pp = float(target.get("tolerance_pp", 1.0))
        paper_std = target.get("paper_std", None)

        matched = [
            r
            for r in reproduced_rows
            if r.get("metric_name") == metric_name and int(r.get("query_budget", 0)) == query_budget
        ]
        if track is not None:
            matched = [r for r in matched if str(r.get("track", "")) == str(track)]

        if len(matched) == 0:
            available = sorted(
                {
                    int(r.get("query_budget", 0))
                    for r in reproduced_rows
                    if r.get("metric_name") == metric_name
                }
            )
            missing_reasons.append(
                f"- no reproduced `{metric_name}` at query_budget={query_budget} (track={track}); available={available}"
            )
            continue

        vals = [float(r["metric_value"]) for r in matched]
        reproduced_mean = _safe_mean(vals)
        reproduced_std = _safe_std(vals)

        delta_abs_pp = abs((reproduced_mean - paper_value) * 100.0)
        criterion = ""
        if paper_std is not None:
            paper_std_float = float(paper_std)
            verdict = "PASS" if abs(reproduced_mean - paper_value) <= paper_std_float else "FAIL"
            criterion = f"|delta| <= paper_std={paper_std_float:.4f}"
        else:
            verdict = "PASS" if delta_abs_pp <= tolerance_pp else "FAIL"
            criterion = f"|delta_pp| <= {tolerance_pp:.2f}"

        table_rows.append(
            f"| {condition} | {metric_name} | {query_budget} | {paper_value:.4f} | {reproduced_mean:.4f} | {reproduced_std:.4f} | {delta_abs_pp:.2f} | {criterion} | {verdict} |"
        )

    if len(table_rows) == 0:
        reason_block = "\n".join(missing_reasons) if missing_reasons else "- no valid targets matched"
        comparison_path.write_text(
            f"# Comparison Table\n\n- status: INCOMPLETE\n{reason_block}\n",
            encoding="utf-8",
        )
        return

    md = (
        "# Comparison Table\n\n"
        "| condition | metric | query_budget | paper_value | reproduced_mean | reproduced_std | delta_abs_pp | criterion | verdict |\n"
        "|---|---|---:|---:|---:|---:|---:|---|---|\n"
        + "\n".join(table_rows)
        + "\n"
    )

    if len(missing_reasons) > 0:
        md += "\n## Missing Targets\n" + "\n".join(missing_reasons) + "\n"

    comparison_path.write_text(md, encoding="utf-8")


def _capture_environment(paper_dir: Path, device: str) -> None:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else "none"
        torch_version = str(torch.__version__)
    except Exception:
        cuda_available = False
        gpu_name = "unknown"
        torch_version = "unknown"

    env = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "device_requested": device,
        "torch": torch_version,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
    }
    out = paper_dir / "results" / "environment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(env, indent=2), encoding="utf-8")


def _resolve_experiment_path(paper_dir: Path, profile: str) -> Path:
    if profile == "smoke":
        smoke = paper_dir / "configs" / "experiment_smoke.yaml"
        if smoke.exists():
            return smoke
    return paper_dir / "configs" / "experiment.yaml"


def run_pipeline(args: argparse.Namespace) -> None:
    paper_dir = PAPERS_ROOT / args.paper_id
    if not paper_dir.exists():
        raise FileNotFoundError(f"Paper folder not found: {paper_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = paper_dir / "logs" / f"pipeline_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    _capture_environment(paper_dir, args.device)

    experiment_path = _resolve_experiment_path(paper_dir, args.profile)
    experiment_cfg = _load_yaml(experiment_path)

    if "victim_train" in stages:
        cmd = [
            sys.executable,
            "scripts/train_victim.py",
            "--config",
            str(paper_dir / "configs" / "victim_train.yaml"),
            "--device",
            args.device,
        ]
        if args.profile == "smoke" and args.smoke_epochs is not None:
            cmd.extend(["--epochs", str(int(args.smoke_epochs))])
        if args.profile == "smoke" and args.smoke_batch_size is not None:
            cmd.extend(["--batch-size", str(int(args.smoke_batch_size))])
        _run_command(cmd, log_path, args.dry_run)

    if "victim_eval" in stages:
        cmd = [
            sys.executable,
            "scripts/eval_victim.py",
            "--config",
            str(paper_dir / "configs" / "victim_eval.yaml"),
            "--device",
            args.device,
        ]
        _run_command(cmd, log_path, args.dry_run)

    if "attack" in stages:
        cmd = [
            sys.executable,
            "-m",
            "mebench",
            "run",
            "--config",
            str(experiment_path),
            "--device",
            args.device,
        ]
        _run_command(cmd, log_path, args.dry_run)

    reproduced_rows: list[dict[str, Any]] = []
    if "collect" in stages:
        if args.dry_run:
            print("[dry-run] skip collect")
        else:
            reproduced_rows = _collect_reproduced_metrics(args.paper_id, paper_dir, experiment_cfg)

    if "compare" in stages:
        if args.dry_run:
            print("[dry-run] skip compare")
        else:
            if not reproduced_rows:
                reproduced_rows = _collect_reproduced_metrics(args.paper_id, paper_dir, experiment_cfg)
            _build_comparison_table(args.paper_id, paper_dir, experiment_cfg, reproduced_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduction set/pipeline helper")
    sub = parser.add_subparsers(dest="command", required=True)

    p_bootstrap = sub.add_parser("bootstrap", help="Bootstrap paper folder layout from index")
    p_bootstrap.add_argument(
        "--index",
        type=str,
        default=str(PAPERS_ROOT / "index.yaml"),
        help="Path to repro/papers/index.yaml",
    )

    p_run = sub.add_parser("run", help="Run staged pipeline for a paper")
    p_run.add_argument("--paper-id", type=str, required=True)
    p_run.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    p_run.add_argument("--device", type=str, default="cuda:0")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument(
        "--stages",
        type=str,
        default="victim_train,victim_eval,attack,collect,compare",
        help="Comma-separated stages",
    )
    p_run.add_argument(
        "--smoke-epochs",
        type=int,
        default=2,
        help="Epoch override used only when profile=smoke",
    )
    p_run.add_argument(
        "--smoke-batch-size",
        type=int,
        default=64,
        help="Batch-size override used only when profile=smoke victim training",
    )

    args = parser.parse_args()
    if args.command == "bootstrap":
        bootstrap_from_index(Path(args.index))
        return
    if args.command == "run":
        run_pipeline(args)
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
