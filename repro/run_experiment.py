"""Reproduction pipeline runner for paper folders under repro/papers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
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

DEFAULT_STAGES = "victim_train,victim_eval,attack,collect,compare"

REPRO_PREFLIGHT_ATTACKS: dict[str, dict[str, Any]] = {
    "2023_karmakar_marich": {
        "attack_name": "marich",
        "output_mode": "hard_top1",
        "requires_surrogate": True,
        "requires_generator_ckpt": False,
    },
    "2021_wang_blackbox_dissector": {
        "attack_name": "blackbox_dissector",
        "output_mode": "hard_top1",
        "requires_surrogate": True,
        "requires_generator_ckpt": False,
    },
    "2020_barbalau_blackbox_ripper": {
        "attack_name": "blackbox_ripper",
        "output_mode": "soft_prob",
        "requires_surrogate": False,
        "requires_generator_ckpt": True,
    },
    "2023_tan_disguide": {
        "attack_name": "disguide",
        "output_mode": "soft_prob",
        "requires_surrogate": False,
        "requires_generator_ckpt": False,
    },
    "2021_truong_dfme": {
        "attack_name": "dfme",
        "output_mode": "soft_prob",
        "requires_surrogate": False,
        "requires_generator_ckpt": False,
    },
    "2023_beetham_dual_students": {
        "attack_name": "ds",
        "output_mode": "soft_prob",
        "requires_surrogate": False,
        "requires_generator_ckpt": False,
    },
}


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


def _run_command(cmd: list[str], log_path: Path, dry_run: bool, live_output: bool) -> None:
    shown = shlex.join(cmd)
    print(f"$ {shown}")
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n$ {shown}\n")

    if dry_run:
        return

    if live_output:
        rc = subprocess.run(cmd, cwd=ROOT, check=False).returncode
        with log_path.open("a", encoding="utf-8") as logf:
            logf.write("[live-output] subprocess output streamed directly to terminal\n")
    else:
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


def _resolve_from_root(raw: str) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return ROOT / path


def _resolve_checkpoint_like(checkpoint_path: str) -> Path | None:
    base = _resolve_from_root(checkpoint_path)
    if base.exists():
        return base
    if base.suffix:
        return None
    for suffix in (".pth", ".pt"):
        candidate = base.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def _preflight_single(paper_id: str, profile: str) -> tuple[bool, list[str]]:
    if paper_id not in REPRO_PREFLIGHT_ATTACKS:
        return False, [
            f"[FAIL] unsupported paper_id for preflight: {paper_id}",
            f"       supported: {sorted(REPRO_PREFLIGHT_ATTACKS.keys())}",
        ]

    req = REPRO_PREFLIGHT_ATTACKS[paper_id]
    paper_dir = PAPERS_ROOT / paper_id
    messages: list[str] = [f"[{paper_id}] profile={profile}"]
    ok = True

    if not paper_dir.exists():
        return False, messages + [f"[FAIL] paper folder missing: {paper_dir}"]

    experiment_path = _resolve_experiment_path(paper_dir, profile, "pair1")
    if not experiment_path.exists():
        return False, messages + [f"[FAIL] experiment config missing: {experiment_path}"]

    cfg = _load_yaml(experiment_path)
    attack_cfg = cfg.get("attack", {}) if isinstance(cfg.get("attack", {}), dict) else {}
    victim_cfg = cfg.get("victim", {}) if isinstance(cfg.get("victim", {}), dict) else {}
    budget_cfg = cfg.get("budget", {}) if isinstance(cfg.get("budget", {}), dict) else {}
    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset", {}), dict) else {}

    expected_attack = str(req["attack_name"])
    attack_name = str(attack_cfg.get("name", ""))
    if attack_name != expected_attack:
        ok = False
        messages.append(f"[FAIL] attack.name={attack_name!r} (expected {expected_attack!r})")
    else:
        messages.append(f"[OK] attack.name={attack_name}")

    expected_mode = str(req["output_mode"])
    attack_mode = str(attack_cfg.get("output_mode", ""))
    victim_mode = str(victim_cfg.get("output_mode", ""))
    if attack_mode != expected_mode or victim_mode != expected_mode:
        ok = False
        messages.append(
            "[FAIL] output mode mismatch "
            f"(victim={victim_mode!r}, attack={attack_mode!r}, expected={expected_mode!r})"
        )
    else:
        messages.append(f"[OK] output_mode={expected_mode}")

    attack_budget = int(attack_cfg.get("max_budget", 0) or 0)
    top_budget = int(budget_cfg.get("max_budget", 0) or 0)
    if attack_budget <= 0 or top_budget <= 0 or attack_budget != top_budget:
        ok = False
        messages.append(
            "[FAIL] budget mismatch "
            f"(attack.max_budget={attack_budget}, budget.max_budget={top_budget})"
        )
    else:
        messages.append(f"[OK] budget.max_budget={top_budget}")

    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run", {}), dict) else {}
    seeds = run_cfg.get("seeds", [])
    if not isinstance(seeds, list) or len(seeds) == 0:
        ok = False
        messages.append("[FAIL] run.seeds must be a non-empty list")
    else:
        messages.append(f"[OK] run.seeds={seeds}")

    victim_ckpt = str(victim_cfg.get("checkpoint_ref", "")).strip()
    if victim_ckpt == "":
        ok = False
        messages.append("[FAIL] victim.checkpoint_ref is missing")
    else:
        victim_path = _resolve_from_root(victim_ckpt)
        if not victim_path.exists():
            ok = False
            messages.append(f"[FAIL] victim checkpoint missing: {victim_path}")
        else:
            messages.append(f"[OK] victim checkpoint: {victim_path}")

    if bool(req["requires_surrogate"]):
        surrogate_name = str(dataset_cfg.get("surrogate_name", ""))
        if surrogate_name not in {"IMAGENET", "ImageNet", "imagenet", "ILSVRC", "ILSVRC2012"}:
            ok = False
            messages.append(f"[FAIL] dataset.surrogate_name={surrogate_name!r} (expected ImageNet family)")
        else:
            messages.append(f"[OK] dataset.surrogate_name={surrogate_name}")

        surrogate_root_raw = str(dataset_cfg.get("surrogate_root", "")).strip()
        if surrogate_root_raw == "":
            surrogate_root_raw = str(os.environ.get("MEBENCH_IMAGENET_ROOT", "")).strip()
        if surrogate_root_raw == "":
            ok = False
            messages.append(
                "[FAIL] missing ImageNet root (set dataset.surrogate_root or MEBENCH_IMAGENET_ROOT)"
            )
        else:
            surrogate_root = _resolve_from_root(surrogate_root_raw)
            split_dir = surrogate_root / ("train" if bool(dataset_cfg.get("train_split", True)) else "val")
            if not split_dir.exists():
                ok = False
                messages.append(f"[FAIL] surrogate split dir missing: {split_dir}")
            else:
                messages.append(f"[OK] surrogate split dir: {split_dir}")

    if bool(req["requires_generator_ckpt"]):
        generator_ckpt = str(attack_cfg.get("generator_checkpoint", "")).strip()
        if generator_ckpt == "":
            ok = False
            messages.append("[FAIL] attack.generator_checkpoint is missing")
        else:
            resolved = _resolve_checkpoint_like(generator_ckpt)
            if resolved is None:
                ok = False
                messages.append(
                    "[FAIL] generator checkpoint missing: "
                    f"{_resolve_from_root(generator_ckpt)} (.pth/.pt fallback not found)"
                )
            else:
                messages.append(f"[OK] generator checkpoint: {resolved}")

                # Paper-faithful guard for Black-Box Ripper: use official CIFAR100-6-class
                # generator artifact expected by paper reproduction docs/config generation.
                if paper_id == "2020_barbalau_blackbox_ripper":
                    stem = str(resolved.stem).strip().lower()
                    if stem != "cifar_100_6_classes_gan":
                        ok = False
                        messages.append(
                            "[FAIL] blackbox_ripper paper profile requires official generator "
                            "checkpoint stem 'cifar_100_6_classes_gan' "
                            f"(got '{resolved.stem}')"
                        )
                    else:
                        messages.append("[OK] blackbox_ripper official generator stem verified")

    extracted_spec_path = paper_dir / "extracted_spec.yaml"
    spec = _load_yaml(extracted_spec_path)
    targets = spec.get("reported_results", {}).get("targets", []) if isinstance(spec, dict) else []
    if not isinstance(targets, list) or len(targets) == 0:
        ok = False
        messages.append(f"[FAIL] extracted_spec targets missing: {extracted_spec_path}")
    else:
        messages.append(f"[OK] extracted_spec targets={len(targets)}")

    return ok, messages


def run_preflight(args: argparse.Namespace) -> None:
    if args.paper_id == "all":
        paper_ids = list(REPRO_PREFLIGHT_ATTACKS.keys())
    else:
        paper_ids = [str(args.paper_id)]

    overall_ok = True
    for paper_id in paper_ids:
        ok, messages = _preflight_single(paper_id=paper_id, profile=str(args.profile))
        for line in messages:
            print(line)
        print("-" * 72)
        overall_ok = overall_ok and ok

    if not overall_ok and bool(args.strict):
        raise SystemExit(1)


def _resolve_config_path(
    paper_dir: Path,
    base_name: str,
    pair: str,
    profile: str,
) -> Path:
    """Resolve config file with optional pair-specific overrides.

    Existing pair-1 behavior is preserved using the legacy filenames:
      victim_train.yaml, victim_eval.yaml, attack.yaml, experiment.yaml.

    Pair-2 can use pair-suffixed filenames:
      victim_train_pair2.yaml, victim_eval_pair2.yaml, attack_pair2.yaml,
      experiment_pair2.yaml.
    If no explicit pair-2 file exists for an attack or experiment, we fall back to
    the best-effort existing alias pattern.
    """

    config_dir = paper_dir / "configs"

    # Pair-1 keeps current pathing to avoid regressions.
    if pair == "pair1":
        if base_name == "experiment":
            if profile == "smoke":
                smoke = config_dir / "experiment_smoke.yaml"
                if smoke.exists():
                    return smoke
            return config_dir / "experiment.yaml"
        return config_dir / f"{base_name}.yaml"

    pair_suffix = f"{base_name}_pair2.yaml"
    explicit_pair = config_dir / pair_suffix
    if explicit_pair.exists():
        return explicit_pair

    if base_name == "experiment":
        # Prefer explicit smoke profile pair-2 file when available.
        if profile == "smoke":
            smoke_pair = config_dir / "experiment_smoke_pair2.yaml"
            if smoke_pair.exists():
                return smoke_pair

        # Some generated configs use the paper-specific naming pattern.
        for candidate in config_dir.glob("experiment_paper_pair2*.yaml"):
            return candidate

        # Fallback to a generic legacy pair-2 file if it already exists.
        pair2_full = config_dir / "experiment_pair2.yaml"
        if pair2_full.exists():
            return pair2_full

    # Last-resort fallback keeps the pipeline runnable if a specific pair file is missing.
    return config_dir / f"{base_name}.yaml"


def _resolve_experiment_path(paper_dir: Path, profile: str, pair: str) -> Path:
    return _resolve_config_path(paper_dir, "experiment", pair, profile)


def _apply_repro_stage_toggles(
    requested_stages: list[str], experiment_cfg: dict[str, Any]
) -> list[str]:
    """Apply stage toggles defined in experiment YAML.

    Supported schema:
      repro:
        victim_eval:
          enabled: true|false
    """

    stages = list(requested_stages)
    repro_cfg = experiment_cfg.get("repro", {})
    if not isinstance(repro_cfg, dict):
        return stages

    victim_eval_cfg = repro_cfg.get("victim_eval")
    enabled: Any = None
    if isinstance(victim_eval_cfg, dict):
        enabled = victim_eval_cfg.get("enabled")
    elif isinstance(victim_eval_cfg, bool):
        enabled = victim_eval_cfg

    if enabled is True and "victim_eval" not in stages:
        if "victim_train" in stages:
            insert_at = stages.index("victim_train") + 1
            stages.insert(insert_at, "victim_eval")
        else:
            stages.insert(0, "victim_eval")
    elif enabled is False and "victim_eval" in stages:
        stages = [stage for stage in stages if stage != "victim_eval"]

    return stages


def _resolve_requested_stages(stages_arg: str | None) -> list[str]:
    raw = "" if stages_arg is None else str(stages_arg).strip()
    if raw != "":
        return [s.strip() for s in raw.split(",") if s.strip()]

    return [s.strip() for s in DEFAULT_STAGES.split(",") if s.strip()]


def _resolve_victim_train_output(victim_train_cfg: dict[str, Any]) -> Path | None:
    raw = victim_train_cfg.get("out")
    if not isinstance(raw, str):
        return None
    out_str = raw.strip()
    if out_str == "":
        return None
    out_path = Path(out_str)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    return out_path


def _prefer_existing_victim_checkpoint(
    stages: list[str],
    *,
    paper_dir: Path,
    profile: str,
    pair: str,
    stages_arg: str | None,
) -> list[str]:
    """Checkpoint-first behavior for implicit default stages.

    If user did not pass --stages and victim checkpoint already exists, skip victim_train.
    Explicit --stages is respected as-is.
    """

    raw = "" if stages_arg is None else str(stages_arg).strip()
    if raw != "":
        return stages

    victim_train_cfg_path = _resolve_config_path(paper_dir, "victim_train", pair, profile)
    victim_train_cfg = _load_yaml(victim_train_cfg_path)
    if not victim_train_cfg:
        return stages

    out_path = _resolve_victim_train_output(victim_train_cfg)
    if out_path is None or not out_path.exists():
        return stages

    filtered = [stage for stage in stages if stage != "victim_train"]
    print(f"[INFO] Reusing existing victim checkpoint: {out_path}")
    return filtered


def run_pipeline(args: argparse.Namespace) -> None:
    paper_dir = PAPERS_ROOT / args.paper_id
    if not paper_dir.exists():
        raise FileNotFoundError(f"Paper folder not found: {paper_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = paper_dir / "logs" / f"pipeline_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    requested_stages = _resolve_requested_stages(args.stages)
    _capture_environment(paper_dir, args.device)

    experiment_path = _resolve_experiment_path(paper_dir, args.profile, args.pair)
    experiment_cfg = _load_yaml(experiment_path)
    stages = _apply_repro_stage_toggles(requested_stages, experiment_cfg)
    stages = _prefer_existing_victim_checkpoint(
        stages,
        paper_dir=paper_dir,
        profile=args.profile,
        pair=args.pair,
        stages_arg=args.stages,
    )

    if "victim_train" in stages:
        victim_train_cfg = _resolve_config_path(paper_dir, "victim_train", args.pair, args.profile)
        cmd = [
            sys.executable,
            "scripts/train_victim.py",
            "--config",
            str(victim_train_cfg),
            "--device",
            args.device,
        ]
        if args.profile == "smoke" and args.smoke_epochs is not None:
            cmd.extend(["--epochs", str(int(args.smoke_epochs))])
        if args.profile == "smoke" and args.smoke_batch_size is not None:
            cmd.extend(["--batch-size", str(int(args.smoke_batch_size))])
        _run_command(cmd, log_path, args.dry_run, args.live_output)

    if "victim_eval" in stages:
        victim_eval_cfg = _resolve_config_path(paper_dir, "victim_eval", args.pair, args.profile)
        cmd = [
            sys.executable,
            "scripts/eval_victim.py",
            "--config",
            str(victim_eval_cfg),
            "--device",
            args.device,
        ]
        _run_command(cmd, log_path, args.dry_run, args.live_output)

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
        _run_command(cmd, log_path, args.dry_run, args.live_output)

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
    p_run.add_argument(
        "--pair",
        choices=["pair1", "pair2"],
        default="pair1",
        help="Dataset pair for reproduction profiles (pair1 or pair2)",
    )
    p_run.add_argument("--device", type=str, default="cuda:0")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument(
        "--stages",
        type=str,
        default="",
        help=(
            "Comma-separated stages. If omitted, defaults to "
            "victim_train,victim_eval,attack,collect,compare, but victim_train is "
            "auto-skipped when an existing victim checkpoint is found."
        ),
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
    p_run.add_argument(
        "--live-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream child process output directly to terminal (recommended for tqdm/pbar)",
    )

    p_preflight = sub.add_parser("preflight", help="Validate repro readiness before long runs")
    p_preflight.add_argument(
        "--paper-id",
        type=str,
        default="all",
        help=(
            "Paper id to validate. Use 'all' for the tracked attacks: "
            "2023_karmakar_marich, 2021_wang_blackbox_dissector, "
            "2020_barbalau_blackbox_ripper, 2023_tan_disguide, "
            "2021_truong_dfme, 2023_beetham_dual_students"
        ),
    )
    p_preflight.add_argument("--profile", choices=["smoke", "full"], default="full")
    p_preflight.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit non-zero when any preflight check fails",
    )

    args = parser.parse_args()
    if args.command == "bootstrap":
        bootstrap_from_index(Path(args.index))
        return
    if args.command == "run":
        run_pipeline(args)
        return
    if args.command == "preflight":
        run_preflight(args)
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
