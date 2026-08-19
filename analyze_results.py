import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Run-name classification
# ---------------------------------------------------------------------------
# Run names follow these conventions:
#   Main  :  SET-X_<attack>_<label>_<budget>_seed<N>
#   Abl.  :  SET-X_<attack>_<label>_<budget>_seed<N>_ablation_<variant>
#   Repro :  SET-X_<attack>_<label>_<budget>_seed<N>_official[_<sub>]
#   Legacy:  <base_run_name>__<tag>     (double underscore tag suffix)
# Anything else falls into "other".

_BASE_RE = re.compile(
    r"^(SET-[A-Z]\d+)_(.+?)_(soft|hard)_(\d+[km])_seed(\d+)$"
)
_ABL_RE = re.compile(
    r"^(SET-[A-Z]\d+)_(.+?)_(soft|hard)_(\d+[km])_seed(\d+)_ablation_(.+)$"
)
_OFF_RE = re.compile(
    r"^(SET-[A-Z]\d+)_(.+?)_(soft|hard)_(\d+[km])_seed(\d+)_official(?:_(.+))?$"
)

# Optimizer suffixes that distinguish canonical main-tier runs by the
# substitute optimizer used (e.g. SET-B1_random_soft_20k_seed0_adamw).
# Default optimizer when no suffix is present is "sgd". Only stripped when
# the remainder matches _BASE_RE — ablation/sweep variants stay in their own
# category.
_OPTIMIZER_SUFFIX_RE = re.compile(r"_(adamw)$")

# Substitute architecture markers. Used to split SET-B (and any future set)
# results between the legacy substitute (no marker) and a newer surrogate-
# victim unified substitute. For SET-B specifically, `_sub_resnet34` runs are
# routed to the virtual set_id "SET-B1-main" while bare SET-B1 runs (resnet18)
# become "SET-B1-legacy" and appear in the paper appendix.
_SUBSTITUTE_SUFFIX_RE = re.compile(r"_sub_(resnet34|resnet18|xie2019|lenet_mnist)$")
_LEGACY_SET_REMAP = {"SET-B1": "SET-B1-legacy"}  # bare SET-B1 → legacy
_MAIN_SET_REMAP = {
    ("SET-B1", "resnet34"): "SET-B1-main",
}

# Augmentation marker suffixes. Two variants emit different downstream
# columns:
#   _aug       → 'strong' (SwiftThief-style: RRC scale 0.2 + HFlip + CJ + Gray)
#   _aug_soft  → 'soft'   (MNIST-appropriate: RandomCrop pad=2 only)
# Longest first so `_aug_soft` is matched before bare `_aug`.
_AUG_VARIANT_SUFFIXES: List[Tuple[str, str]] = [
    ("_aug_soft", "soft"),
    ("_aug",      "strong"),
]


def _classify_run(run_name: str) -> Optional[dict]:
    """Classify a run directory name into one of: main / ablation / official /
    legacy / other. Returns a dict of parsed fields or None if unparsable.

    The classifier strips a trailing ``__<tag>`` suffix first (legacy/version
    variants) and reclassifies the base name, attaching the tag as ``variant``.
    """
    legacy_tag = None
    base_name = run_name
    if "__" in run_name:
        base_name, legacy_tag = run_name.rsplit("__", 1)
        base_name = base_name.strip()
        legacy_tag = legacy_tag.strip() or None

    m = _ABL_RE.match(base_name)
    if m:
        set_id, attack, label, budget, seed, variant = m.groups()
        return {
            "category": "ablation",
            "run_name": run_name,
            "set_id": set_id,
            "attack": attack,
            "label": label,
            "budget_tag": budget,
            "seed": int(seed),
            "variant": variant,
            "legacy_tag": legacy_tag,
            "optimizer": "sgd",
            "augmentation": "",
        }

    m = _OFF_RE.match(base_name)
    if m:
        set_id, attack, label, budget, seed, variant = m.groups()
        return {
            "category": "official",
            "run_name": run_name,
            "set_id": set_id,
            "attack": attack,
            "label": label,
            "budget_tag": budget,
            "seed": int(seed),
            "variant": variant or "default",
            "legacy_tag": legacy_tag,
            "optimizer": "sgd",
            "augmentation": "",
        }

    # Strip optional aug-variant suffix FIRST so the optimizer detection below
    # can still recognise canonical names. Variants are scanned longest-first
    # (_aug_soft before _aug). Only strip when the remainder (with or without
    # optimizer suffix and substitute suffix) matches _BASE_RE.
    augmentation = ""  # "" | "strong" | "soft"
    candidate = base_name

    def _is_canonical_after_optional_opt_and_sub(name: str) -> bool:
        """True if `name` ends in optimizer+/sub_+/base, with each strip optional."""
        s = name
        m_opt = _OPTIMIZER_SUFFIX_RE.search(s)
        if m_opt:
            s = s[: m_opt.start()]
        m_sub = _SUBSTITUTE_SUFFIX_RE.search(s)
        if m_sub:
            s = s[: m_sub.start()]
        return bool(_BASE_RE.match(s))

    for suffix, label in _AUG_VARIANT_SUFFIXES:
        if candidate.endswith(suffix):
            stripped_aug = candidate[: -len(suffix)]
            if _is_canonical_after_optional_opt_and_sub(stripped_aug):
                candidate = stripped_aug
                augmentation = label
                break

    # Optimizer suffix may sit between the seed (or sub_X) and end.
    optimizer = "sgd"
    m_opt = _OPTIMIZER_SUFFIX_RE.search(candidate)
    if m_opt:
        stripped = candidate[: m_opt.start()]
        # Stripped form must be canonical, with sub_X allowed in tail position.
        check = stripped
        m_sub_check = _SUBSTITUTE_SUFFIX_RE.search(check)
        if m_sub_check:
            check = check[: m_sub_check.start()]
        if _BASE_RE.match(check):
            candidate = stripped
            optimizer = m_opt.group(1)

    # Substitute suffix (e.g. _sub_resnet34): strip + remember.
    substitute = ""
    m_sub = _SUBSTITUTE_SUFFIX_RE.search(candidate)
    if m_sub and _BASE_RE.match(candidate[: m_sub.start()]):
        candidate = candidate[: m_sub.start()]
        substitute = m_sub.group(1)

    m = _BASE_RE.match(candidate)
    if m:
        set_id, attack, label, budget, seed = m.groups()
        # Apply set_id remap so SET-B legacy/main split into virtual sets.
        if substitute and (set_id, substitute) in _MAIN_SET_REMAP:
            set_id = _MAIN_SET_REMAP[(set_id, substitute)]
        elif not substitute and set_id in _LEGACY_SET_REMAP:
            set_id = _LEGACY_SET_REMAP[set_id]
        return {
            "category": "legacy" if legacy_tag else "main",
            "run_name": run_name,
            "set_id": set_id,
            "attack": attack,
            "label": label,
            "budget_tag": budget,
            "seed": int(seed),
            "variant": legacy_tag,
            "legacy_tag": legacy_tag,
            "optimizer": optimizer,
            "augmentation": augmentation,
            "substitute": substitute or "default",
        }

    return {
        "category": "other",
        "run_name": run_name,
        "set_id": None,
        "attack": None,
        "label": None,
        "budget_tag": None,
        "seed": None,
        "variant": legacy_tag or run_name,
        "legacy_tag": legacy_tag,
        "optimizer": "sgd",
        "augmentation": "",
    }


# Folder suffix convention for versioned comparisons:
#   <run_name>__prealign
#   <run_name>__<custom_tag>
# This keeps the original run_name parsable while exposing variant labels.
_LEGACY_VARIANT_TAGS = {
    "legacy",
    "prealign",
    "pre-align",
    "pre_align",
    "v0",
    "old",
    "before-align",
}


def _split_variant_tag(run_name: str) -> Tuple[str, Optional[str]]:
    if "__" not in run_name:
        return run_name, None
    base, tag = run_name.rsplit("__", 1)
    base = base.strip()
    tag = tag.strip()
    if not base or not tag:
        return run_name, None
    return base, tag


def _annotate_attack_name(attack_name: str, variant_tag: Optional[str]) -> str:
    if variant_tag is None:
        return attack_name
    tag_norm = str(variant_tag).strip().lower()
    if tag_norm in _LEGACY_VARIANT_TAGS:
        return f"{attack_name} (pre-align)"
    return f"{attack_name} ({variant_tag})"


# ---------------------------------------------------------------------------
# Paper table display settings
# ---------------------------------------------------------------------------

CANONICAL_SETS = {"SET-A1", "SET-B1-legacy", "SET-B1-main", "SET-C1"}
LOW_BUDGET_MAX = 100_000  # queries; above this = data-free / high-budget tier

# Tables with fewer rows than this are too sparse to read as a benchmark and
# are skipped from the per-(dataset, optimizer) output. Adjust if a sparser
# table becomes informative.
MIN_TABLE_ROWS = 2

# Combined paper-tables ordering. Each item is (set_id, optimizer); pairs that
# are missing from the data (or fall under MIN_TABLE_ROWS) are silently
# skipped. Order here is the legacy "baseline only" order used for the
# auxiliary `paper_tables_baseline.tex` file.
COMBINED_PAPER_ORDER: List[Tuple[str, str]] = [
    ("SET-A1", "sgd"),
    ("SET-A1", "adamw"),
    ("SET-B1-main", "adamw"),
    ("SET-B1-main", "sgd"),
    ("SET-B1-legacy", "adamw"),  # appendix
    ("SET-B1-legacy", "sgd"),    # appendix
    ("SET-C1", "sgd"),
    ("SET-C1", "adamw"),
]

# Per-Set master ordering: for each canonical Set, the list of
# (optimizer, augmentation) tables to include, in display order. Augmentation
# value is "" (baseline), "strong" (SwiftThief-style), or "soft" (mild).
# Tables missing from the data are silently skipped. SET-A1 now exposes
# all three variants (baseline / strong / soft) since the comparison is
# directly informative.
PER_SET_PAPER_ORDER: Dict[str, List[Tuple[str, str]]] = {
    # SET-A1: 5 col — baseline / +Aug / +Aug-soft / AdamW / AdamW+Aug.
    "SET-A1": [
        ("sgd", ""),
        ("sgd", "strong"),
        ("sgd", "soft"),
        ("adamw", ""),
        ("adamw", "strong"),
    ],
    # SET-B1-main: 4 col with resnet34 substitute (surrogate-victim unified).
    # SGD column reported as negative finding (~44–47%, learning insufficient
    # for resnet34 at 20k budget; see PROJECT_COMPENDIUM.md §2.8).
    "SET-B1-main": [
        ("sgd",   ""),
        ("sgd",   "strong"),
        ("adamw", ""),
        ("adamw", "strong"),
    ],
    # SET-B1-legacy: 4 col with resnet18 substitute (paper appendix only).
    "SET-B1-legacy": [
        ("sgd",   ""),
        ("sgd",   "strong"),
        ("adamw", ""),
        ("adamw", "strong"),
    ],
    # SET-C1: 4 col — added AdamW + AdamW+Aug under 4-col unification.
    "SET-C1": [
        ("sgd", ""),
        ("sgd", "strong"),
        ("adamw", ""),
        ("adamw", "strong"),
    ],
}

# Order in which Sets appear in the main `paper_tables.tex` master.
# SET-B1-legacy is intentionally placed last so it reads as appendix material.
MAIN_MASTER_SETS: List[str] = ["SET-A1", "SET-B1-main", "SET-C1", "SET-B1-legacy"]

# Attacks excluded from the public v1 benchmark tables. See top-of-file notes
# in mebench/attackers/{game,es_attack}.py for the parity status.
EXCLUDED_FROM_PUBLIC = {"GAME", "ES_ATTACK", "ESATTACK"}

# Maps CSV attack-name prefix → (display_name, query_type).
# Order matters: longer/more-specific prefixes must come first.
_ATTACK_DISPLAY_MAP: List[Tuple[str, str, str]] = [
    ("RANDOM_HARD",                  "Random",            "Hard"),
    ("RANDOM_SOFT",                  "Random",            "Soft"),
    ("KNOCKOFF_NETS_SOFT",           "KnockoffNets",      "Soft"),
    ("CLOUDLEAK_SOFT",               "CloudLeak",         "Soft"),
    ("SWIFTTHIEF_HARD",              "SwiftThief",        "Hard"),
    ("SWIFTTHIEF_SOFT",              "SwiftThief",        "Soft"),
    # ActiveThief variants: use the names from the original paper directly
    # (Uncertainty / DFAL / DFAL+k-Center). The AT- prefix is dropped since
    # these strategy names already identify the method family.
    ("ACTIVETHIEF_DFAL_HARD",        "DFAL",              "Hard"),
    ("ACTIVETHIEF_DFAL_SOFT",        "DFAL",              "Soft"),
    ("ACTIVETHIEF_UNCERTAINTY_HARD", "Uncertainty",       "Hard"),
    ("ACTIVETHIEF_UNCERTAINTY_SOFT", "Uncertainty",       "Soft"),
    ("ACTIVETHIEF_HARD",             "DFAL+k-Center",     "Hard"),
    ("ACTIVETHIEF_SOFT",             "DFAL+k-Center",     "Soft"),
    ("MARICH_HARD",                  "MARICH",            "Hard"),
    ("BLACKBOX_DISSECTOR_HARD",      "BlackBoxDissector", "Hard"),
    ("INVERSENET_HARD",              "InverseNet",        "Hard"),
    ("DFME_SOFT",                    "DFME",              "Soft"),
    ("DFMS_HARD",                    "DFMS",              "Hard"),
    ("DS_SOFT",                      "DualStudents",      "Soft"),
    ("DS_HARD",                      "DualStudents",      "Hard"),
    ("MAZE_SOFT",                    "MAZE",              "Soft"),
    ("DISGUIDE_SOFT",                "DisGuide",          "Soft"),
    ("DISGUIDE_HARD",                "DisGuide",          "Hard"),
]

# Per-table sort: rows within each tier are split into Soft and Hard groups
# (in that order) and sorted by descending Accuracy *within* that table. See
# `_row_sort_key` in `_generate_per_dataset_tables`.


def _parse_attack_key(csv_attack: str) -> Optional[Tuple[str, str]]:
    """Map a CSV attack name → (display_name, query_type), or None to skip."""
    if "(" in csv_attack:  # skip parenthetical variants like '(LR003)'
        return None
    upper = csv_attack.upper()
    if any(upper.startswith(prefix) for prefix in EXCLUDED_FROM_PUBLIC):
        return None
    for prefix, display, qtype in _ATTACK_DISPLAY_MAP:
        if upper.startswith(prefix):
            return display, qtype
    return None


def _extract_mean(value_str: str) -> Optional[float]:
    if not isinstance(value_str, str) or not value_str.strip():
        return None
    try:
        return float(value_str.split("±")[0].strip())
    except ValueError:
        return None


def _is_single_seed(value_str: str) -> bool:
    """True when no standard deviation is present (single run)."""
    return isinstance(value_str, str) and "±" not in value_str


def _fmt_pct(mean: float, std: Optional[float] = None, *, latex: bool, decimals: int = 1) -> str:
    """Format 0..1 mean (and optional std) as percent string.

    LaTeX:    "92.1$_{\\pm 0.5}$"  (subscript std — compact, top-tier-venue style)
    Markdown: "92.1 ± 0.5"
    """
    pct = float(mean) * 100.0
    if std is None:
        return f"{pct:.{decimals}f}"
    spct = float(std) * 100.0
    if latex:
        return f"{pct:.{decimals}f}$_{{\\pm{spct:.{decimals}f}}}$"
    return f"{pct:.{decimals}f} ± {spct:.{decimals}f}"


def _parse_mean_std(value_str: str) -> Optional[Tuple[float, Optional[float]]]:
    if not isinstance(value_str, str) or not value_str.strip():
        return None
    try:
        if "±" in value_str:
            parts = value_str.split("±")
            return float(parts[0].strip()), float(parts[1].strip())
        return float(value_str.strip()), None
    except ValueError:
        return None


def _fmt_cell(value_str: str, bold: bool, decimals: int = 1) -> str:
    """LaTeX-formatted percent cell. Bold via \\textbf{}."""
    if not isinstance(value_str, str) or not value_str.strip():
        return "--"
    parsed = _parse_mean_std(value_str)
    if parsed is None:
        return value_str
    m, s = parsed
    cell = _fmt_pct(m, s, latex=True, decimals=decimals)
    return r"\textbf{" + cell + r"}" if bold else cell


def _fmt_cell_md(value_str: str, bold: bool, decimals: int = 1) -> str:
    """Markdown-formatted percent cell. Bold via **...**."""
    if not isinstance(value_str, str) or not value_str.strip():
        return "--"
    parsed = _parse_mean_std(value_str)
    if parsed is None:
        return value_str
    m, s = parsed
    cell = _fmt_pct(m, s, latex=False, decimals=decimals)
    return f"**{cell}**" if bold else cell


def _best_mean(sub_df: pd.DataFrame, col: str) -> Optional[float]:
    means = sub_df[col].apply(_extract_mean).dropna()
    return float(means.max()) if not means.empty else None


def _is_best(value_str: str, best: Optional[float], tol: float = 1e-9) -> bool:
    if best is None:
        return False
    m = _extract_mean(value_str)
    return m is not None and abs(m - best) < tol


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def _format_mean_std(values: pd.Series, fmt: str = "{:.4f}") -> str:
    """Format a numeric series as 'mean ± std'; bare mean if single sample."""
    values = values.dropna()
    if len(values) == 0:
        return ""
    if len(values) == 1:
        return fmt.format(float(values.iloc[0]))
    return f"{fmt.format(float(values.mean()))} ± {fmt.format(float(values.std()))}"


def _write_variant_report(
    rows: List[dict],
    md_path: Path,
    csv_path: Path,
    title: str,
) -> None:
    """Write an aggregated report for category=ablation/official.

    Groups by (Set, Attack, Label, BudgetTag, Variant) and reduces seeds to
    mean ± std. One Markdown section per (Set, Attack, Label, Budget) so all
    variants of the same experimental unit appear in a single table.
    """
    if not rows:
        return
    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["Set", "Attack", "Label", "BudgetTag", "Variant"])
        .agg(
            n_seeds=("Seed", "nunique"),
            Acc=("Final_Accuracy", lambda s: _format_mean_std(s)),
            Agreement=("Final_Agreement", lambda s: _format_mean_std(s)),
            Acc_mean=("Final_Accuracy", "mean"),
            Agr_mean=("Final_Agreement", "mean"),
        )
        .reset_index()
        .sort_values(["Set", "Attack", "Label", "BudgetTag", "Variant"])
    )
    grouped.to_csv(csv_path, index=False)

    lines = [f"# {title}\n"]
    for (set_id, attack, label, budget), block in grouped.groupby(
        ["Set", "Attack", "Label", "BudgetTag"], sort=True
    ):
        # `attack` already encodes the label (e.g. RANDOM_SOFT), so just use it.
        lines.append(f"\n## {set_id} — {attack.upper()} ({budget})\n")
        lines.append("| Variant | n_seeds | Accuracy | Agreement |")
        lines.append("|---|---:|---|---|")
        block_sorted = block.sort_values("Acc_mean", ascending=False)
        for _, r in block_sorted.iterrows():
            lines.append(
                f"| {r['Variant']} | {int(r['n_seeds'])} | {r['Acc']} | {r['Agreement']} |"
            )
    md_path.write_text("\n".join(lines) + "\n")


def _write_other_report(rows: List[dict], csv_path: Path) -> None:
    """Dump unclassified runs to a plain CSV so they're not silently lost."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)


# Canonical budgets reported in the paper. Attacks whose iteration scheduling
# leaves a small remainder short of these targets (e.g. DS uses
# `(budget // query_batch_size) * query_batch_size = 9_999_360` for the 10M tier)
# are snapped to the nearest canonical bucket when within `BUDGET_SNAP_TOL`.
_CANONICAL_BUDGETS: List[int] = [
    10_000, 20_000, 50_000, 100_000,
    1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000,
]
BUDGET_SNAP_TOL = 0.01  # 1% — DS 9_999_360 → 10_000_000 (diff ~0.006%)


def _canonical_budget(raw_budget: int) -> int:
    """Snap raw budget to the nearest canonical bucket if within BUDGET_SNAP_TOL.

    Returns the original integer when no canonical bucket is close enough,
    so unexpected/legacy budgets stay distinguishable rather than being
    silently merged.
    """
    if raw_budget <= 0:
        return raw_budget
    for canon in _CANONICAL_BUDGETS:
        if abs(raw_budget - canon) / canon <= BUDGET_SNAP_TOL:
            return canon
    return raw_budget


_VICTIM_ACC_RE = re.compile(r"\[VERIFY\]\s+Victim Test Accuracy:\s*([0-9.]+)\s*%")


def _extract_victim_acc(seed_dir: Path) -> Optional[float]:
    """Read the verified victim test accuracy line that engine.py emits at
    experiment start. Returns the value as a 0..1 float, or None if missing.
    """
    log_path = seed_dir / "experiment.log"
    if not log_path.exists():
        return None
    try:
        with open(log_path) as f:
            for line in f:
                m = _VICTIM_ACC_RE.search(line)
                if m:
                    return float(m.group(1)) / 100.0
    except Exception:
        return None
    return None


def _load_seed_metrics(seed_dir: Path, preferred_tracks: List[str]) -> Optional[dict]:
    """Read summary.json under a seed_<N>/ dir; return final-checkpoint metrics."""
    summary_path = seed_dir / "summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        data = json.load(f)
    checkpoints = data.get("checkpoints", {})
    if not checkpoints:
        return None
    max_cp = max(int(cp) for cp in checkpoints.keys())
    tracks = checkpoints[str(max_cp)]
    track_to_use = next((t for t in preferred_tracks if t in tracks), None)
    if track_to_use not in tracks:
        return None
    metrics = tracks[track_to_use]
    return {
        "Track": track_to_use,
        "Budget": _canonical_budget(max_cp),
        "Final_Accuracy": metrics.get("acc_gt", np.nan),
        "Final_Agreement": metrics.get("agreement", np.nan),
        "Victim_Accuracy": _extract_victim_acc(seed_dir),
    }


def analyze_results(root_dir="runs", output_dir="analysis_results"):
    print(f"Loading results from {root_dir}...")
    root = Path(root_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    main_dir = out_root / "main"
    analysis_dir = out_root / "analysis"
    main_dir.mkdir(exist_ok=True)
    analysis_dir.mkdir(exist_ok=True)

    preferred_tracks = ["track_b"]

    records: Dict[str, List[dict]] = {
        "main": [],
        "ablation": [],
        "official": [],
        "legacy": [],
        "other": [],
    }

    # 1. Load Final Metrics, classifying by run-name structure
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        cls = _classify_run(run_dir.name)
        if cls is None:
            continue

        # Latest timestamp dir
        timestamps = sorted(d for d in run_dir.iterdir() if d.is_dir())
        if not timestamps:
            continue
        latest_run = timestamps[-1]

        for seed_dir in latest_run.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            try:
                seed_value = int(seed_dir.name.replace("seed_", ""))
            except ValueError:
                continue

            m = _load_seed_metrics(seed_dir, preferred_tracks)
            if m is None:
                continue

            # Compose attack name for backwards-compat with per-dataset tables:
            # `<ATTACK>_<LABEL>` (e.g. RANDOM_SOFT) is the key in _ATTACK_DISPLAY_MAP.
            attack_base = (cls["attack"] or "").upper()
            label_upper = (cls["label"] or "").upper()
            attack_name = (
                f"{attack_base}_{label_upper}" if label_upper else attack_base
            )
            if cls["category"] == "legacy" and cls["variant"]:
                attack_name = _annotate_attack_name(attack_name, cls["variant"])

            records[cls["category"]].append(
                {
                    "Set": cls["set_id"],
                    "Attack": attack_name,
                    "Label": cls["label"],
                    "BudgetTag": cls["budget_tag"],
                    "Variant": cls["variant"] or "default",
                    "Optimizer": cls.get("optimizer", "sgd"),
                    "Augmentation": str(cls.get("augmentation", "")),
                    "Seed": cls["seed"] if cls["seed"] is not None else seed_value,
                    "RunName": cls["run_name"],
                    **m,
                }
            )

    n_total = sum(len(v) for v in records.values())
    if n_total == 0:
        print("No results found.")
        return
    print(f"Loaded {n_total} runs: " + ", ".join(
        f"{k}={len(v)}" for k, v in records.items() if v
    ))

    # ------------------------------------------------------------------
    # Main pipeline — paper-tier benchmark results
    # ------------------------------------------------------------------
    main_records = records["main"] + records["legacy"]
    if not main_records:
        print("[WARN] No main runs found; skipping main report.")
        return

    df = pd.DataFrame(main_records)
    output_dir = str(main_dir)  # rebind for downstream code
    df.to_csv(f"{output_dir}/final_metrics_raw.csv", index=False)

    # Ensure column exists even when classifier didn't emit it (legacy rows).
    if "Augmentation" not in df.columns:
        df["Augmentation"] = ""
    df["Augmentation"] = df["Augmentation"].fillna("").astype(str)

    # Keep a single track per (Set, Attack, Budget, Optimizer, Augmentation):
    #   Track-B-only benchmark: select the canonical runtime track.
    track_rank = {name: i for i, name in enumerate(preferred_tracks)}
    selected_groups = []
    for (set_id, attack, budget, optimizer, augmentation), group in df.groupby(
        ["Set", "Attack", "Budget", "Optimizer", "Augmentation"]
    ):
        tracks_available = sorted(group["Track"].unique(), key=lambda t: track_rank.get(t, 999))
        selected_track = tracks_available[0]
        if len(tracks_available) > 1:
            print(
                f"[WARN] Mixed tracks for {set_id}/{attack}/{budget}/{optimizer}"
                f"/aug={augmentation}: {tracks_available} -> using {selected_track}"
            )
        selected_groups.append(group[group["Track"] == selected_track])

    if selected_groups:
        df = pd.concat(selected_groups, ignore_index=True)
    # `final_metrics_selected.csv` intentionally not written — it is a strict
    # subset of `final_metrics_raw.csv` (one track per key) and adds no new
    # information; keep it only in memory for the aggregate step below.

    # 2. Aggregate Mean/Std (Safe Explicit Method)
    grouped = df.groupby(["Set", "Attack", "Budget", "Optimizer", "Augmentation"])

    table_rows = []
    for (set_id, attack, budget, optimizer, augmentation), group in grouped:
        track_label = group["Track"].iloc[0]
        acc_mean = group["Final_Accuracy"].mean()
        acc_std = group["Final_Accuracy"].std()
        agr_mean = group["Final_Agreement"].mean()
        agr_std = group["Final_Agreement"].std()

        acc_str = f"{acc_mean:.4f}"
        if pd.notna(acc_std):
            acc_str += f" ± {acc_std:.4f}"

        agr_str = f"{agr_mean:.4f}"
        if pd.notna(agr_std):
            agr_str += f" ± {agr_std:.4f}"

        # Victim accuracy is a property of the (Set, victim_id) pair; should be
        # constant across attacks and seeds within a Set. Take any non-null
        # value from the group; mismatches across the same Set are caught
        # downstream when we aggregate over the Set.
        if "Victim_Accuracy" in group.columns:
            v_series = group["Victim_Accuracy"].dropna()
            victim_acc = float(v_series.iloc[0]) if not v_series.empty else None
        else:
            victim_acc = None

        table_rows.append(
            {
                "Set": str(set_id),
                "Attack": str(attack),
                "Budget": int(budget),
                "Optimizer": str(optimizer),
                "Augmentation": str(augmentation),
                "Track": str(track_label),
                "Accuracy": acc_str,
                "Agreement": agr_str,
                "Victim_Accuracy": victim_acc,
            }
        )

    summary_df = pd.DataFrame(table_rows)
    summary_df.to_csv(f"{output_dir}/final_summary.csv", index=False)

    unique_budgets = sorted(summary_df["Budget"].unique())

    def _latex_pm(value: str) -> str:
        if not isinstance(value, str):
            return value
        return value.replace(" ± ", " $\\pm$ ")

    unique_optimizers = sorted(summary_df["Optimizer"].unique())
    aug_variants = sorted({str(v) for v in summary_df.get("Augmentation", [""])})

    # Variant label helper: matches paper_tables formatting.
    #   ""       → "SGD" / "AdamW"
    #   "strong" → "SGD+Aug" / "AdamW+Aug"
    #   "soft"   → "SGD+Aug(soft)" / "AdamW+Aug(soft)"
    def _variant_label(opt_: str, aug_: str) -> str:
        opt_disp = {"sgd": "SGD", "adamw": "AdamW"}.get(opt_, str(opt_).upper())
        a = str(aug_ or "")
        if a == "strong":
            return f"{opt_disp}+Aug"
        if a == "soft":
            return f"{opt_disp}+Aug(soft)"
        return opt_disp

    summary_df = summary_df.copy()
    summary_df["Variant"] = summary_df.apply(
        lambda r: _variant_label(r["Optimizer"], str(r["Augmentation"] or "")), axis=1
    )

    # Order SETs by canonical paper order (SET-A1 → SET-B1 → SET-C1) but only
    # include those that have data.
    _SET_ORDER = ["SET-A1", "SET-B1-main", "SET-C1", "SET-B1-legacy"]

    # Console preview of per-Set × per-Budget accuracy pivot (readable only;
    # no side-effect files). All persisted matrices are consolidated in
    # `paper_tables.tex` / `paper_tables.md` produced downstream.
    def _pivot_to_md(pivot: pd.DataFrame) -> str:
        disp = pivot.copy()
        for col in disp.columns:
            disp[col] = disp[col].apply(
                lambda v: _fmt_cell_md(v, bold=False) if isinstance(v, str) else "--"
            )
        return disp.to_markdown()

    for set_id in [s for s in _SET_ORDER if s in summary_df["Set"].unique()]:
        entries = PER_SET_PAPER_ORDER.get(set_id, [])
        if not entries:
            continue
        ordered_vars = [_variant_label(o, a) for o, a in entries]
        set_df = summary_df[summary_df["Set"] == set_id]
        if set_df.empty:
            continue
        print(f"\n=== {set_id} ===")
        print(f"Variants (in column order): {ordered_vars}")
        for b in sorted(set_df["Budget"].unique()):
            subset = set_df[set_df["Budget"] == b]
            if subset.empty:
                continue
            print(f"\n--- {set_id} | Budget: {b} ---")
            try:
                pivot_acc = subset.pivot(index="Attack", columns="Variant", values="Accuracy")
                ordered_present = [c for c in ordered_vars if c in pivot_acc.columns]
                extras = [c for c in pivot_acc.columns if c not in ordered_present]
                pivot_acc = pivot_acc[ordered_present + extras]
                print("[Accuracy %]")
                print(_pivot_to_md(pivot_acc))
            except Exception as e:
                print(f"ERROR: pivot failed for {set_id}/budget={b}: {e}")

    # 3. Paper-quality per-dataset tables (final .tex + .md)
    _generate_per_dataset_tables(summary_df, output_dir)

    print(f"\nSaved main analysis to {output_dir}/")

    # ------------------------------------------------------------------
    # Analysis pipeline — ablation / official-repro / other (non-paper)
    # ------------------------------------------------------------------
    _write_variant_report(
        records["ablation"],
        analysis_dir / "ablation_report.md",
        analysis_dir / "ablation_summary.csv",
        title="Ablation Experiments",
    )
    _write_variant_report(
        records["official"],
        analysis_dir / "official_repro_report.md",
        analysis_dir / "official_repro_summary.csv",
        title="Official Reproduction Variants",
    )
    if records["other"]:
        _write_other_report(records["other"], analysis_dir / "other_runs.csv")

    print(f"Saved analysis variants to {analysis_dir}/")


# ---------------------------------------------------------------------------
# Per-dataset paper table generation
# ---------------------------------------------------------------------------

def _build_caption(
    set_id: str,
    low_label: str,
    high_label: str,
    all_single: bool,
    full_notation: bool = False,
    victim_acc: Optional[float] = None,
) -> str:
    """Return a LaTeX \\caption{...} string for a per-dataset table.

    Paper convention: the first table introduces all notation; subsequent
    tables reference it implicitly. Pass ``full_notation=True`` for the
    primary table (typically SET-A1) and ``False`` for the rest.

    If ``victim_acc`` is provided, the caption notes the victim model's test
    accuracy on the same evaluation set used for the Acc column — this is
    the natural upper bound the attacker is approximating.
    """
    # --- budget tier ---
    if low_label and high_label:
        budget_desc = f"Pool-based: {low_label} surrogate queries; data-free: {high_label} synthetic."
    elif low_label:
        budget_desc = f"Pool-based: {low_label} surrogate queries."
    else:
        budget_desc = f"Data-free: {high_label} synthetic queries."

    victim_note = (
        f"Victim test accuracy: {victim_acc*100:.2f}\\%." if victim_acc is not None else ""
    )

    if not full_notation:
        if all_single:
            notes = r"Best Acc/Fid per Soft/Hard group in \textbf{bold}; all single-seed."
        else:
            notes = r"Best Acc/Fid per Soft/Hard group in \textbf{bold}; $^\dag$ single-seed."
        body = " ".join(p for p in [budget_desc, victim_note, notes] if p)
        return f"\\caption{{Model extraction results on {set_id}. {body}}}"

    # --- full notation (primary table) ---
    col_defs = (
        r"\textbf{Acc}: accuracy vs ground-truth labels. "
        r"\textbf{Fid}: agreement rate with victim outputs. "
        r"\textbf{Label}: \emph{Soft} (probability vector) or \emph{Hard} (top-1 only)."
    )
    pool_abbrevs = (
        r"\emph{Pool-based}---"
        r"AT: ActiveThief (kC: $k$-Center, DFAL: DeepFool AL); "
        r"BB-Dissector: BlackBox Dissector."
    )
    free_abbrevs = (
        r"\emph{Data-free}---"
        r"DFME: Data-Free Model Extraction; "
        r"DFMS: Data-Free Model Stealing (hard-label); "
        r"DS: Dual Student; "
        r"MAZE: Zeroth-Order Gradient Estimation; "
        r"DiSGuide: Disagreement-Guided."
    )
    if all_single:
        notes = (
            r"Best Acc/Fid per Soft/Hard group (within each tier) in "
            r"\textbf{bold}; all results single-seed."
        )
    else:
        notes = (
            r"Best Acc/Fid per Soft/Hard group (within each tier) in "
            r"\textbf{bold}; $^\dag$ marks single-seed results."
        )
    body = " ".join(
        p for p in [budget_desc, victim_note, col_defs, pool_abbrevs, free_abbrevs, notes] if p
    )
    return f"\\caption{{Model extraction results on {set_id}. {body}}}"


def _generate_per_dataset_tables(summary_df: pd.DataFrame, output_dir: str) -> None:
    """Generate LaTeX + Markdown tables per (canonical dataset, optimizer, aug).

    Three flavours per (dataset, optimizer):
      - baseline (aug=False):       paper_table_<set>_<opt>.tex
      - augmented (aug=True):       paper_table_<set>_<opt>_aug.tex
      - side-by-side comparison:    paper_table_<set>_<opt>_compare.tex
                                    (only when both flavours have data for ≥1 attack)

    Each table has rows = attacks and columns = Method | Label | Acc | Fid,
    with row groups separating pool-based vs data-free budget tiers.
    Best Acc and Fid per tier are bolded. Single-seed rows are marked †.

    Combined outputs:
      - paper_tables.tex      : baseline tables in COMBINED_PAPER_ORDER (existing)
      - paper_tables_aug.tex  : augmented tables in COMBINED_PAPER_ORDER (new)
      - paper_tables_compare.tex : side-by-side comparison tables (new)
      - paper_tables.md       : Markdown preview of all flavours
    """
    out = Path(output_dir)

    # Filter to canonical sets; drop parenthetical variant rows
    df = summary_df[summary_df["Set"].isin(CANONICAL_SETS)].copy()
    df = df[~df["Attack"].str.contains(r"\(", na=False)]
    # Augmentation column may be absent in old CSVs — default to False
    if "Augmentation" not in df.columns:
        df["Augmentation"] = ""
    df["Augmentation"] = df["Augmentation"].fillna("").astype(str)

    # Build normalised records with display names
    records = []
    for _, row in df.iterrows():
        key = _parse_attack_key(row["Attack"])
        if key is None:
            continue
        display, qtype = key
        records.append(
            {
                "Set": row["Set"],
                "Optimizer": row.get("Optimizer", "sgd"),
                "Augmentation": str(row.get("Augmentation", "")),
                "Display": display,
                "QueryType": qtype,
                "Budget": row["Budget"],
                "Accuracy": row["Accuracy"],
                "Agreement": row["Agreement"],
                "Victim_Accuracy": row.get("Victim_Accuracy"),
                "IsLow": row["Budget"] <= LOW_BUDGET_MAX,
            }
        )

    if not records:
        print("[WARN] No records matched for per-dataset paper tables.")
        return

    rdf = pd.DataFrame(records)
    unique_optimizers = sorted(rdf["Optimizer"].unique())
    # Per-Set victim accuracy: take any non-null value (engine.py verifies the
    # same victim checkpoint for every run within a Set, so all entries should
    # agree). Warn if a Set ends up with conflicting values.
    victim_acc_by_set: Dict[str, float] = {}
    if "Victim_Accuracy" in rdf.columns:
        for set_id_, grp in rdf.groupby("Set"):
            vals = grp["Victim_Accuracy"].dropna().unique().tolist()
            if not vals:
                continue
            if len(vals) > 1:
                print(
                    f"[paper] {set_id_} has multiple victim accs {vals}; "
                    f"using first ({vals[0]})"
                )
            victim_acc_by_set[set_id_] = float(vals[0])
    # Per-table sort: within each tier, attacks group Soft-first then Hard, and
    # within each (tier, label) group rows are ordered by descending Acc shown
    # in that specific table (so each table reads top-down best-to-worst).
    _LABEL_GROUP_RANK = {"Soft": 0, "Hard": 1}

    def _row_sort_key(r: dict) -> Tuple[int, float, str]:
        label_rank = _LABEL_GROUP_RANK.get(r["QueryType"], 99)
        acc = _extract_mean(r.get("Accuracy", ""))
        acc_score = acc if acc is not None else float("-inf")
        return (label_rank, -acc_score, r["Display"])

    md_lines: List[str] = ["# Paper Tables — Per-Dataset\n"]

    # Decide which (set, optimizer, aug="") pair gets the full-notation
    # caption. The convention is "the first baseline table in the combined
    # paper layout"; if none of COMBINED_PAPER_ORDER survives the data filter,
    # fall back to None.
    def _has_enough_rows(set_id_: str, optimizer_: str, aug_: str = "") -> bool:
        s = rdf[
            (rdf["Set"] == set_id_)
            & (rdf["Optimizer"] == optimizer_)
            & (rdf["Augmentation"] == aug_)
        ]
        return len(s) >= MIN_TABLE_ROWS

    primary_pair: Optional[Tuple[str, str]] = next(
        (p for p in COMBINED_PAPER_ORDER if _has_enough_rows(*p)),
        None,
    )

    # Augmentation variants found in data (preserve a stable order).
    _AUG_ORDER = ["", "strong", "soft"]
    aug_present = [a for a in _AUG_ORDER if (rdf["Augmentation"] == a).any()]

    # Only produce Markdown preview per (set × optimizer × augmentation) —
    # no intermediate .tex files. All final .tex output is generated below via
    # `_build_unified_per_set_table` split by tier (pool-based / data-free).
    for set_id in sorted(CANONICAL_SETS):
        sdf_all = rdf[rdf["Set"] == set_id]
        if sdf_all.empty:
            continue
        section_added = False

        # Loop over optimizers, then augmentation flavour (markdown preview only)
        for optimizer in sorted(sdf_all["Optimizer"].unique()):
            opt_df = sdf_all[sdf_all["Optimizer"] == optimizer]
            if opt_df.empty:
                continue

            for augmentation in aug_present:
                sdf = opt_df[opt_df["Augmentation"] == augmentation]
                if sdf.empty:
                    continue
                if len(sdf) < MIN_TABLE_ROWS:
                    print(
                        f"[paper] skipped {set_id}/{optimizer}/aug={augmentation}: "
                        f"only {len(sdf)} row(s) (<{MIN_TABLE_ROWS})"
                    )
                    continue
                if not section_added:
                    md_lines.append(f"\n## {set_id}\n")
                    section_added = True

                low_rows = sorted(sdf[sdf["IsLow"]].to_dict("records"), key=_row_sort_key)
                high_rows = sorted(sdf[~sdf["IsLow"]].to_dict("records"), key=_row_sort_key)

                # Best values per (tier, label) for bolding. Soft and Hard are
                # treated as independent leaderboards: bolding the best Soft and
                # the best Hard separately is more informative for a benchmark
                # than picking a single tier-wide winner that mixes label types.
                def _best_by_label(tier_df: pd.DataFrame) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
                    out_: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
                    for label in ("Soft", "Hard"):
                        sub = tier_df[tier_df["QueryType"] == label]
                        out_[label] = (
                            _best_mean(sub, "Accuracy"),
                            _best_mean(sub, "Agreement"),
                        )
                    return out_

                best_low = _best_by_label(sdf[sdf["IsLow"]])
                best_high = _best_by_label(sdf[~sdf["IsLow"]])

                # Budget display labels for section headers
                low_budgets = sorted(sdf[sdf["IsLow"]]["Budget"].unique())
                high_budgets = sorted(sdf[~sdf["IsLow"]]["Budget"].unique())
                low_label_parts = list(dict.fromkeys(f"{b // 1000}K" for b in low_budgets))
                high_label_parts = list(dict.fromkeys(f"{round(b / 1_000_000)}M" for b in high_budgets))
                low_label = "/".join(low_label_parts) if low_budgets else ""
                high_label = "/".join(high_label_parts) if high_budgets else ""

                # † in caption when the whole cell is single-seed
                all_single = sdf["Accuracy"].apply(_is_single_seed).all()

                # Primary (baseline-only) caption gets full notation
                is_primary = (
                    augmentation == "" and (set_id, optimizer) == primary_pair
                )
                caption_body = _build_caption(
                    set_id, low_label, high_label, all_single,
                    full_notation=is_primary,
                    victim_acc=victim_acc_by_set.get(set_id),
                )
                opt_display = {"sgd": "SGD", "adamw": "AdamW"}.get(optimizer, optimizer)
                aug_caption_map = {
                    "strong": " Augmentation: strong (SwiftThief-style RRC+HFlip+CJ+Grayscale).",
                    "soft":   " Augmentation: soft (RandomCrop padding=2 only).",
                }
                aug_tag = aug_caption_map.get(augmentation, "")
                opt_tag = f" Optimizer: {opt_display}.{aug_tag}"
                if caption_body.endswith("}"):
                    caption_body = caption_body[:-1] + opt_tag + "}"
                else:
                    caption_body += opt_tag

                # ------------------------------------------------------------------
                # Markdown preview (no .tex written here — see final tier tables below)
                # ------------------------------------------------------------------
                aug_md_map = {
                    "strong": " — with strong augmentation",
                    "soft":   " — with soft augmentation",
                }
                aug_md = aug_md_map.get(augmentation, "")
                md_lines.append(f"\n### {set_id} — optimizer: {optimizer}{aug_md}\n")

                def _md_tier(rows: List[dict], tier_label: str,
                             best_by_label: Dict[str, Tuple[Optional[float], Optional[float]]]) -> None:
                    if not rows:
                        return
                    md_lines.append(f"#### {tier_label}\n")
                    md_lines.append("| Method | Label | Acc | Fid |")
                    md_lines.append("|---|---|---|---|")
                    for r in rows:
                        single = _is_single_seed(r["Accuracy"])
                        name = r["Display"] + (" †" if (single and not all_single) else "")
                        b_acc, b_agr = best_by_label.get(r["QueryType"], (None, None))
                        acc_str = f"**{r['Accuracy']}**" if _is_best(r["Accuracy"], b_acc) else r["Accuracy"]
                        agr_str = f"**{r['Agreement']}**" if _is_best(r["Agreement"], b_agr) else r["Agreement"]
                        md_lines.append(f"| {name} | {r['QueryType']} | {acc_str} | {agr_str} |")
                    md_lines.append("")

                _md_tier(
                    low_rows,
                    f"Pool-based ({low_label} queries)" if low_label else "Pool-based",
                    best_low,
                )
                _md_tier(
                    high_rows,
                    f"Data-free ({high_label} queries)" if high_label else "Data-free",
                    best_high,
                )

    # ----------------------------------------------------------------------
    # Unified per-Set tables — single LaTeX `table*` per Set with all
    # (optimizer, augmentation) variants laid out as side-by-side column
    # pairs. Replaces the old "N separate tables per Set" layout for
    # readability in the paper.
    # ----------------------------------------------------------------------
    def _entry_label(opt_: str, aug_: str) -> str:
        opt_disp = {"sgd": "SGD", "adamw": "AdamW"}.get(opt_, str(opt_).upper())
        if aug_ == "strong":
            return f"{opt_disp}+Aug"
        if aug_ == "soft":
            return f"{opt_disp}+Aug(soft)"
        return opt_disp

    # ----------------------------------------------------------------------
    # FINAL TABLES ONLY. Per Set produce two tables (pool-based, data-free)
    # and one combined master. No auxiliary per-optimizer / per-aug outputs.
    # ----------------------------------------------------------------------
    tier_blocks: Dict[Tuple[str, str], str] = {}  # (set_id, tier) -> tex
    primary_set = next(
        (s for s in MAIN_MASTER_SETS if not rdf[rdf["Set"] == s].empty), None
    )
    for set_id, entries in PER_SET_PAPER_ORDER.items():
        for tier in ("pool", "datafree"):
            block = _build_unified_per_set_table(
                rdf, set_id, entries,
                victim_acc=victim_acc_by_set.get(set_id),
                row_sort_key=_row_sort_key,
                is_primary=(set_id == primary_set and tier == "pool"),
                tier=tier,
            )
            if block is None:
                continue
            tier_blocks[(set_id, tier)] = block

    # Combined master file: for each set, emit pool-based then data-free.
    master_blocks: List[str] = [
        "% Paper tables. One pool-based and one data-free table per Set.\n"
        "% Required: \\usepackage{booktabs}. Wide tables use \\begin{table*}.\n"
        f"% Sets in order: {', '.join(MAIN_MASTER_SETS)}\n\n"
    ]
    for set_id in MAIN_MASTER_SETS:
        for tier in ("pool", "datafree"):
            block = tier_blocks.get((set_id, tier))
            if block is None:
                continue
            master_blocks.append(block + "\n")
    if len(master_blocks) > 1:
        (out / "paper_tables.tex").write_text("\n".join(master_blocks))
        print(f"[paper] Final master → paper_tables.tex "
              f"({len(tier_blocks)} tables, sets={MAIN_MASTER_SETS})")

    # Compact 4-column tables (SGD vs AdamW × Acc/Fid), one per Set.
    # Pool-based rows show aug variants only; data-free rows unchanged.
    # SET-B1-legacy intentionally excluded.
    compact_blocks: List[str] = [
        "% Compact 4-column tables per Set: SGD vs AdamW × Acc/Fid.\n"
        "% Pool-based rows use a single aug variant per Set (declared in each\n"
        "% caption): SET-A1 -> Aug_soft, SET-B/C -> Aug. Data-free rows use no\n"
        "% aug. Method column shows the plain attack name.\n"
        "% Required: \\usepackage{booktabs}.\n"
        f"% Sets: {', '.join(COMPACT_SETS)}\n\n"
    ]
    n_compact = 0
    for set_id in COMPACT_SETS:
        block = _build_compact_per_set_table(
            rdf, set_id,
            victim_acc=victim_acc_by_set.get(set_id),
            row_sort_key=_row_sort_key,
        )
        if block is None:
            continue
        compact_blocks.append(block + "\n")
        n_compact += 1
    if n_compact > 0:
        (out / "paper_tables_compact.tex").write_text("\n".join(compact_blocks))
        print(f"[paper] Compact tables → paper_tables_compact.tex "
              f"({n_compact} tables, sets={COMPACT_SETS})")

    paper_md = out / "paper_tables.md"
    paper_md.write_text("\n".join(md_lines))
    print(f"[paper] Markdown preview → {paper_md.name}")


def _build_unified_per_set_table(
    rdf: pd.DataFrame,
    set_id: str,
    entries: List[Tuple[str, str]],
    victim_acc: Optional[float],
    row_sort_key,
    *,
    is_primary: bool = False,
    tier: str = "both",
) -> Optional[str]:
    """Unified LaTeX table for one Set: multiple (optimizer, augmentation)
    variants as side-by-side `Acc / Fid` column pairs in a single tabular.

    Layout (4 variants example):

        \\begin{tabular}{@{}ll rr rr rr rr@{}}
        \\toprule
                       & & SGD     & ... & AdamW   & ... & AdamW+Aug & ... \\\\
                       & & Acc Fid & ... & Acc Fid & ... &     ...   & ... \\\\
        \\midrule
        Pool-based (...)
        SwiftThief & Soft & ...    & ...           ...
        ...
        \\bottomrule
        \\end{tabular}

    Cells use percent values with subscript std (e.g. 92.1$_{\\pm 0.5}$).
    Best per (tier, label, variant_column, metric) is bolded.
    """
    set_df = rdf[rdf["Set"] == set_id]
    if set_df.empty:
        return None

    # Tier filter: pool-based (IsLow=True) or data-free (IsLow=False) or both.
    # Data-free rows never have augmentation, so restrict entries to baseline
    # (aug="") only when generating a data-free-only table.
    if tier == "pool":
        set_df = set_df[set_df["IsLow"]]
    elif tier == "datafree":
        set_df = set_df[~set_df["IsLow"]]
        entries = [(opt, aug) for (opt, aug) in entries if aug == ""]
    if set_df.empty or not entries:
        return None

    def _opt_disp(opt: str) -> str:
        return {"sgd": "SGD", "adamw": "AdamW"}.get(opt, str(opt).upper())

    def _aug_disp(aug: str) -> str:
        if aug == "strong":
            return "+Aug"
        if aug == "soft":
            return r"+Aug$_\mathrm{soft}$"
        return "Base"

    def _vlabel(opt: str, aug: str) -> str:
        return _opt_disp(opt) + ("" if aug == "" else " " + _aug_disp(aug))

    # Keep only entries with at least one row
    valid_entries: List[Tuple[str, str]] = []
    for opt, aug in entries:
        sub = set_df[(set_df["Optimizer"] == opt) & (set_df["Augmentation"] == aug)]
        if not sub.empty:
            valid_entries.append((opt, aug))
    if not valid_entries:
        return None
    variant_labels = [_vlabel(o, a) for o, a in valid_entries]

    # Group entries by optimizer (preserve order of first appearance)
    opt_order: List[str] = []
    opt_groups: Dict[str, List[str]] = {}
    for opt, aug in valid_entries:
        if opt not in opt_groups:
            opt_order.append(opt)
            opt_groups[opt] = []
        opt_groups[opt].append(aug)

    # row_map[(Display, QueryType, IsLow)][(opt, aug)] = (acc_str, agr_str)
    row_map: Dict[Tuple[str, str, bool], Dict[Tuple[str, str], Tuple[str, str]]] = {}
    for _, r in set_df.iterrows():
        key = (r["Display"], r["QueryType"], bool(r["IsLow"]))
        row_map.setdefault(key, {})[(r["Optimizer"], r["Augmentation"])] = (
            r["Accuracy"], r["Agreement"]
        )

    rows: List[dict] = []
    for (display, qtype, is_low), cells in row_map.items():
        row_cells: List[Tuple[str, str]] = []
        for var in valid_entries:
            ac, fd = cells.get(var, ("", ""))
            row_cells.append((ac, fd))
        # Sort key uses the first variant's accuracy if present, else any
        sort_acc = ""
        for ac, _ in row_cells:
            if ac:
                sort_acc = ac
                break
        rows.append({
            "Display": display, "QueryType": qtype, "IsLow": is_low,
            "cells": row_cells, "Accuracy": sort_acc,
        })

    low_rows = sorted([r for r in rows if r["IsLow"]], key=row_sort_key)
    high_rows = sorted([r for r in rows if not r["IsLow"]], key=row_sort_key)

    def _bests_for_tier(tier_rows: List[dict]) -> Dict[Tuple[str, int, str], Optional[float]]:
        out: Dict[Tuple[str, int, str], Optional[float]] = {}
        for label in ("Soft", "Hard"):
            for vi, _ in enumerate(valid_entries):
                accs: List[float] = []
                fids: List[float] = []
                for r in tier_rows:
                    if r["QueryType"] != label:
                        continue
                    ac, fd = r["cells"][vi]
                    ma = _extract_mean(ac) if ac else None
                    mf = _extract_mean(fd) if fd else None
                    if ma is not None:
                        accs.append(ma)
                    if mf is not None:
                        fids.append(mf)
                out[(label, vi, "acc")] = max(accs) if accs else None
                out[(label, vi, "fid")] = max(fids) if fids else None
        return out

    bests_low = _bests_for_tier(low_rows)
    bests_high = _bests_for_tier(high_rows)

    low_budgets = sorted(set_df[set_df["IsLow"]]["Budget"].unique())
    high_budgets = sorted(set_df[~set_df["IsLow"]]["Budget"].unique())
    low_label_parts = list(dict.fromkeys(f"{b // 1000}K" for b in low_budgets))
    high_label_parts = list(dict.fromkeys(f"{round(b / 1_000_000)}M" for b in high_budgets))
    low_label_str = "/".join(low_label_parts) if low_budgets else ""
    high_label_str = "/".join(high_label_parts) if high_budgets else ""

    set_slug = set_id.lower().replace("-", "")
    victim_note = (
        f" Victim test accuracy: {victim_acc*100:.2f}\\%." if victim_acc is not None else ""
    )

    # Human-readable column layout for caption
    if len(opt_order) == 1:
        only_opt = _opt_disp(opt_order[0])
        cols_desc = f"{only_opt} optimizer with " + " / ".join(
            _aug_disp(a) for a in opt_groups[opt_order[0]]
        )
    else:
        cols_desc = " | ".join(
            f"{_opt_disp(o)} ({' / '.join(_aug_disp(a) for a in opt_groups[o])})"
            for o in opt_order
        )

    # Tier tag for caption / label
    if tier == "pool":
        tier_tag = f"pool-based extraction ({low_label_str} queries)"
        label_slug_extra = "-pool"
    elif tier == "datafree":
        tier_tag = f"data-free extraction ({high_label_str} queries)"
        label_slug_extra = "-datafree"
    else:
        tier_tag = "model extraction"
        label_slug_extra = "-unified"

    if is_primary:
        caption_lines = [
            f"{tier_tag.capitalize()} results on {set_id}." + victim_note,
            r"\textbf{Acc}: top-1 accuracy of the substitute on the victim's "
            r"test set $\mathcal{D}^{\mathrm{test}}$ (\%). "
            r"\textbf{Fid}: agreement rate between substitute and victim "
            r"predictions on $\mathcal{D}^{\mathrm{test}}$ (\%). "
            r"\emph{Soft}: probability-vector queries; \emph{Hard}: top-1-only queries.",
            f"Columns: {cols_desc}. All cells report mean$\\pm$1 std over $n{{=}}3$ seeds.",
            r"Within each variant column, \textbf{bold} marks the column-best "
            r"per Soft/Hard sub-group.",
        ]
    else:
        caption_lines = [
            f"{tier_tag.capitalize()} results on {set_id} "
            f"(Acc/Fid in \\%; mean$\\pm$1 std, $n{{=}}3$)." + victim_note,
            f"Columns: {cols_desc}.",
            r"Notation and bolding rule follow the SET-A1 pool-based table.",
        ]
    caption = " ".join(caption_lines)

    ncols = 2 + 2 * len(valid_entries)
    # Add column-group separators (vertical whitespace at optimizer boundaries)
    col_spec_parts = ["@{}ll"]
    col_cursor = 3
    opt_col_ranges: List[Tuple[str, int, int]] = []  # (opt, col_start, col_end)
    for opt in opt_order:
        n_sub = len(opt_groups[opt])
        opt_col_ranges.append((opt, col_cursor, col_cursor + 2 * n_sub - 1))
        col_spec_parts.append(" " + ("rr " * n_sub).strip())
        col_cursor += 2 * n_sub
    col_spec = " ".join(col_spec_parts) + "@{}"

    # Wide tables (≥3 variant columns) overflow standard \textwidth, so we
    # scale them. Narrow tables stay at their natural booktabs size — wrapping
    # them in \resizebox would force-stretch the table and look awkward.
    wide_table = len(valid_entries) >= 3
    table_env = "table*" if wide_table else "table"

    tex: List[str] = []
    tex.append(rf"\begin{{{table_env}}}[t]")
    tex.append(r"\centering")
    tex.append(f"\\caption{{{caption}}}")
    tex.append(f"\\label{{tab:{set_slug}{label_slug_extra}}}")
    if wide_table:
        # Compact spacing + auto-scale to text width (NeurIPS/ICML standard
        # idiom for wide multi-variant tables).
        tex.append(r"\setlength{\tabcolsep}{4pt}")
        tex.append(r"\resizebox{\textwidth}{!}{%")
    tex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    tex.append(r"\toprule")

    # Header layout:
    #   • If >1 optimizer: 3-row header (Optimizer | Base/+Aug | Acc/Fid)
    #   • If 1  optimizer: 2-row header (Base/+Aug      | Acc/Fid)
    if len(opt_order) > 1:
        # Row 1: optimizer spans (Base, +Aug, ...) → spans 2 * n_sub cols
        row1 = ["", ""]
        for opt in opt_order:
            span = 2 * len(opt_groups[opt])
            row1.append(f"\\multicolumn{{{span}}}{{c}}{{{_opt_disp(opt)}}}")
        tex.append(" & ".join(row1) + r" \\")
        cmids_top = []
        for _opt, cs, ce in opt_col_ranges:
            cmids_top.append(f"\\cmidrule(lr){{{cs}-{ce}}}")
        tex.append(" ".join(cmids_top))

    # Variant (Base / +Aug / +Aug_soft) header — middle row.
    # Skip this row entirely when every optimizer has just a single variant
    # (typical for data-free tables), since a lone "Base" adds no information.
    any_multi_variant = any(len(opt_groups[o]) > 1 for o in opt_order)
    if any_multi_variant:
        row_var = ["", ""]
        col_cursor = 3
        cmids_var: List[str] = []
        for opt in opt_order:
            for aug in opt_groups[opt]:
                row_var.append(f"\\multicolumn{{2}}{{c}}{{{_aug_disp(aug)}}}")
                cmids_var.append(f"\\cmidrule(lr){{{col_cursor}-{col_cursor + 1}}}")
                col_cursor += 2
        tex.append(" & ".join(row_var) + r" \\")
        tex.append(" ".join(cmids_var))

    # Sub-header: Method | Label | Acc Fid Acc Fid ...
    sub = ["Method", "Label"]
    for _opt in opt_order:
        for _aug in opt_groups[_opt]:
            sub.extend(["Acc", "Fid"])
    tex.append(" & ".join(sub) + r" \\")

    def _tier_lines(tier_rows: List[dict],
                    bests: Dict[Tuple[str, int, str], Optional[float]],
                    tier_label: str,
                    *,
                    include_header: bool = True) -> List[str]:
        if not tier_rows:
            return []
        lines: List[str] = []
        if include_header:
            lines += [r"\midrule",
                      f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{tier_label}}}}} \\\\"]
        lines.append(r"\midrule")
        for r in tier_rows:
            parts = [r["Display"], r["QueryType"]]
            for vi in range(len(valid_entries)):
                ac, fd = r["cells"][vi]
                b_acc = bests.get((r["QueryType"], vi, "acc"))
                b_fid = bests.get((r["QueryType"], vi, "fid"))
                parts.append(_fmt_cell(ac or "", _is_best(ac or "", b_acc)))
                parts.append(_fmt_cell(fd or "", _is_best(fd or "", b_fid)))
            lines.append(" & ".join(parts) + r" \\")
        return lines

    # When rendering a single-tier table, omit the tier heading and place all
    # rows directly under the column header — no need for a "Pool-based (…)"
    # midrule separator that adds visual noise.
    if tier == "pool":
        tex += _tier_lines(low_rows, bests_low, "", include_header=False)
    elif tier == "datafree":
        tex += _tier_lines(high_rows, bests_high, "", include_header=False)
    else:
        tex += _tier_lines(
            low_rows, bests_low,
            f"Pool-based ({low_label_str} queries)" if low_label_str else "Pool-based",
        )
        tex += _tier_lines(
            high_rows, bests_high,
            f"Data-free ({high_label_str} queries)" if high_label_str else "Data-free",
        )

    tex.append(r"\bottomrule")
    if wide_table:
        tex.append(r"\end{tabular}%")
        tex.append(r"}")  # close \resizebox
    else:
        tex.append(r"\end{tabular}")
    tex.append(rf"\end{{{table_env}}}")
    return "\n".join(tex)


# Sets included in the compact 4-column table file (excludes SET-B1-legacy).
COMPACT_SETS: List[str] = ["SET-A1", "SET-B1-main", "SET-C1"]

# Per-Set choice of the single augmentation variant used for pool-based rows
# in the compact table. SET-A1 uses the soft aug (Aug_soft); the other sets
# do not have a soft aug variant and use the standard strong Aug.
COMPACT_POOL_AUG: Dict[str, str] = {
    "SET-A1": "soft",
    "SET-B1-main": "strong",
    "SET-C1": "strong",
}


def _build_compact_per_set_table(
    rdf: pd.DataFrame,
    set_id: str,
    *,
    victim_acc: Optional[float],
    row_sort_key,
) -> Optional[str]:
    """Compact per-Set table with exactly 4 data columns:
    SGD (Acc, Fid) and AdamW (Acc, Fid).

    Row policy:
      - Pool-based attacks (IsLow=True): one row per attack. The aug variant
        used (per-Set: SET-A1 -> soft, SET-B/C -> strong) is declared in the
        caption only; the Method column shows the plain attack name.
      - Data-free attacks (IsLow=False): one row per attack (no aug variant).

    Best per (tier, QueryType, optimizer, metric) is bolded.
    """
    set_df = rdf[rdf["Set"] == set_id]
    if set_df.empty:
        return None

    optimizers = ["sgd", "adamw"]
    pool_aug = COMPACT_POOL_AUG.get(set_id, "strong")

    def _aug_caption(aug: str) -> str:
        if aug == "strong":
            return r"Aug"
        if aug == "soft":
            return r"Aug$_\mathrm{soft}$"
        return r"no augmentation"

    # Pool rows: for each (Display, QueryType, opt), pick the Set-specific
    # aug variant when it exists, otherwise fall back to base (aug=""). The
    # fallback covers attacks that bundle augmentation internally and thus
    # have no explicit *_aug config (e.g., SwiftThief's contrastive pipeline
    # already applies its own aug, so its base run is the fair comparison).
    # Data-free rows: keep only the base (aug="").
    pool_by_key: Dict[Tuple[str, str, str], Dict[str, Tuple[str, str]]] = {}
    #   (Display, QueryType, opt) -> {aug: (acc, agr)}
    datafree_map: Dict[Tuple[str, str], Dict[str, Tuple[str, str]]] = {}
    #   (Display, QueryType) -> {opt: (acc, agr)}
    for _, r in set_df.iterrows():
        opt = r["Optimizer"]
        if opt not in optimizers:
            continue
        aug = r["Augmentation"]
        is_low = bool(r["IsLow"])
        if is_low:
            if aug not in (pool_aug, ""):
                continue
            pool_by_key.setdefault(
                (r["Display"], r["QueryType"], opt), {}
            )[aug] = (r["Accuracy"], r["Agreement"])
        else:
            if aug != "":
                continue
            datafree_map.setdefault(
                (r["Display"], r["QueryType"]), {}
            )[opt] = (r["Accuracy"], r["Agreement"])

    # Assemble pool rows: pick pool_aug if available else base.
    pool_rows_map: Dict[Tuple[str, str], Dict[str, Tuple[str, str]]] = {}
    for (display, qtype, opt), by_aug in pool_by_key.items():
        picked = by_aug.get(pool_aug) or by_aug.get("")
        if picked is None:
            continue
        pool_rows_map.setdefault((display, qtype), {})[opt] = picked

    rows: List[dict] = []
    for (display, qtype), cells in pool_rows_map.items():
        row_cells = [cells.get(opt, ("", "")) for opt in optimizers]
        sort_acc = next((ac for ac, _ in row_cells if ac), "")
        rows.append({
            "Display": display, "QueryType": qtype, "IsLow": True,
            "cells": row_cells, "Accuracy": sort_acc,
        })
    for (display, qtype), cells in datafree_map.items():
        row_cells = [cells.get(opt, ("", "")) for opt in optimizers]
        sort_acc = next((ac for ac, _ in row_cells if ac), "")
        rows.append({
            "Display": display, "QueryType": qtype, "IsLow": False,
            "cells": row_cells, "Accuracy": sort_acc,
        })

    if not rows:
        return None

    low_rows = sorted([r for r in rows if r["IsLow"]], key=row_sort_key)
    high_rows = sorted([r for r in rows if not r["IsLow"]], key=row_sort_key)

    def _bests_for(tier_rows: List[dict]) -> Dict[Tuple[str, int, str], Optional[float]]:
        best: Dict[Tuple[str, int, str], Optional[float]] = {}
        for label in ("Soft", "Hard"):
            for oi, _ in enumerate(optimizers):
                accs: List[float] = []
                fids: List[float] = []
                for r in tier_rows:
                    if r["QueryType"] != label:
                        continue
                    ac, fd = r["cells"][oi]
                    ma = _extract_mean(ac) if ac else None
                    mf = _extract_mean(fd) if fd else None
                    if ma is not None:
                        accs.append(ma)
                    if mf is not None:
                        fids.append(mf)
                best[(label, oi, "acc")] = max(accs) if accs else None
                best[(label, oi, "fid")] = max(fids) if fids else None
        return best

    bests_low = _bests_for(low_rows)
    bests_high = _bests_for(high_rows)

    low_budgets = sorted(set_df[set_df["IsLow"]]["Budget"].unique())
    high_budgets = sorted(set_df[~set_df["IsLow"]]["Budget"].unique())
    low_label = "/".join(dict.fromkeys(f"{b // 1000}K" for b in low_budgets))
    high_label = "/".join(dict.fromkeys(
        f"{round(b / 1_000_000)}M" for b in high_budgets))

    set_slug = set_id.lower().replace("-", "")
    victim_note = (
        f" Victim test accuracy: {victim_acc*100:.2f}\\%."
        if victim_acc is not None else ""
    )
    caption = (
        f"Model-extraction results on {set_id} (Acc/Fid in \\%; "
        f"mean$\\pm$1 std, $n{{=}}3$).{victim_note} "
        f"Pool-based rows are trained with {_aug_caption(pool_aug)} "
        r"augmentation on the substitute; data-free rows use no augmentation. "
        r"\emph{Soft}/\emph{Hard}: probability-vector vs top-1-only queries. "
        r"Within each tier/QueryType/optimizer, \textbf{bold} marks the "
        r"column-best."
    )

    ncols = 2 + 2 * len(optimizers)  # Method + Label + 2*(Acc,Fid)
    col_spec = "@{}ll rr rr@{}"

    tex: List[str] = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(f"\\caption{{{caption}}}")
    tex.append(f"\\label{{tab:{set_slug}-compact}}")
    tex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    tex.append(r"\toprule")
    # Header row 1: optimizer spans
    row1 = ["", "",
            r"\multicolumn{2}{c}{SGD}",
            r"\multicolumn{2}{c}{AdamW}"]
    tex.append(" & ".join(row1) + r" \\")
    tex.append(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}")
    # Header row 2: sub-columns
    tex.append(" & ".join(["Method", "Label", "Acc", "Fid", "Acc", "Fid"]) + r" \\")

    def _tier_lines(tier_rows: List[dict],
                    bests: Dict[Tuple[str, int, str], Optional[float]],
                    tier_label: str) -> List[str]:
        if not tier_rows:
            return []
        lines: List[str] = [
            r"\midrule",
            f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{tier_label}}}}} \\\\",
            r"\midrule",
        ]
        for r in tier_rows:
            parts = [r["Display"], r["QueryType"]]
            for oi in range(len(optimizers)):
                ac, fd = r["cells"][oi]
                b_acc = bests.get((r["QueryType"], oi, "acc"))
                b_fid = bests.get((r["QueryType"], oi, "fid"))
                parts.append(_fmt_cell(ac or "", _is_best(ac or "", b_acc)))
                parts.append(_fmt_cell(fd or "", _is_best(fd or "", b_fid)))
            lines.append(" & ".join(parts) + r" \\")
        return lines

    tex += _tier_lines(
        low_rows, bests_low,
        f"Pool-based ({low_label} queries)" if low_label else "Pool-based",
    )
    tex += _tier_lines(
        high_rows, bests_high,
        f"Data-free ({high_label} queries)" if high_label else "Data-free",
    )
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    return "\n".join(tex)


def _build_unified_per_set_md(
    rdf: pd.DataFrame,
    set_id: str,
    entries: List[Tuple[str, str]],
    row_sort_key,
) -> List[str]:
    """Markdown counterpart of _build_unified_per_set_table."""
    set_df = rdf[rdf["Set"] == set_id]
    if set_df.empty:
        return []

    def _vlabel(opt: str, aug: str) -> str:
        opt_disp = {"sgd": "SGD", "adamw": "AdamW"}.get(opt, str(opt).upper())
        if aug == "strong":
            return opt_disp + "+Aug"
        if aug == "soft":
            return opt_disp + "+Aug(soft)"
        return opt_disp

    valid_entries: List[Tuple[str, str]] = []
    for opt, aug in entries:
        sub = set_df[(set_df["Optimizer"] == opt) & (set_df["Augmentation"] == aug)]
        if not sub.empty:
            valid_entries.append((opt, aug))
    if not valid_entries:
        return []
    variant_labels = [_vlabel(o, a) for o, a in valid_entries]

    row_map: Dict[Tuple[str, str, bool], Dict[Tuple[str, str], Tuple[str, str]]] = {}
    for _, r in set_df.iterrows():
        key = (r["Display"], r["QueryType"], bool(r["IsLow"]))
        row_map.setdefault(key, {})[(r["Optimizer"], r["Augmentation"])] = (
            r["Accuracy"], r["Agreement"]
        )

    rows: List[dict] = []
    for (display, qtype, is_low), cells in row_map.items():
        row_cells: List[Tuple[str, str]] = []
        for var in valid_entries:
            ac, fd = cells.get(var, ("", ""))
            row_cells.append((ac, fd))
        sort_acc = ""
        for ac, _ in row_cells:
            if ac:
                sort_acc = ac
                break
        rows.append({
            "Display": display, "QueryType": qtype, "IsLow": is_low,
            "cells": row_cells, "Accuracy": sort_acc,
        })

    low_rows = sorted([r for r in rows if r["IsLow"]], key=row_sort_key)
    high_rows = sorted([r for r in rows if not r["IsLow"]], key=row_sort_key)

    def _bests_for_tier(tier_rows: List[dict]) -> Dict[Tuple[str, int, str], Optional[float]]:
        out: Dict[Tuple[str, int, str], Optional[float]] = {}
        for label in ("Soft", "Hard"):
            for vi, _ in enumerate(valid_entries):
                accs, fids = [], []
                for r in tier_rows:
                    if r["QueryType"] != label:
                        continue
                    ac, fd = r["cells"][vi]
                    ma = _extract_mean(ac) if ac else None
                    mf = _extract_mean(fd) if fd else None
                    if ma is not None: accs.append(ma)
                    if mf is not None: fids.append(mf)
                out[(label, vi, "acc")] = max(accs) if accs else None
                out[(label, vi, "fid")] = max(fids) if fids else None
        return out

    bests_low = _bests_for_tier(low_rows)
    bests_high = _bests_for_tier(high_rows)

    out_lines: List[str] = []
    # Two-row markdown header: variant groups then Acc/Fid
    top_cells = ["Method", "Label"]
    for vl in variant_labels:
        top_cells.extend([f"{vl} Acc (%)", f"{vl} Fid (%)"])
    out_lines.append("| " + " | ".join(top_cells) + " |")
    out_lines.append("|" + "|".join(["---"] * len(top_cells)) + "|")

    def _emit_tier(tier_rows, bests, label):
        if not tier_rows:
            return
        out_lines.append(f"\n**{label}**\n")
        out_lines.append("| " + " | ".join(top_cells) + " |")
        out_lines.append("|" + "|".join(["---"] * len(top_cells)) + "|")
        for r in tier_rows:
            cells_md = [r["Display"], r["QueryType"]]
            for vi in range(len(valid_entries)):
                ac, fd = r["cells"][vi]
                b_acc = bests.get((r["QueryType"], vi, "acc"))
                b_fid = bests.get((r["QueryType"], vi, "fid"))
                cells_md.append(_fmt_cell_md(ac or "", _is_best(ac or "", b_acc)))
                cells_md.append(_fmt_cell_md(fd or "", _is_best(fd or "", b_fid)))
            out_lines.append("| " + " | ".join(cells_md) + " |")

    # Replace the single header we prepared above with tier-keyed versions
    out_lines = []
    _emit_tier(low_rows, bests_low, "Pool-based")
    _emit_tier(high_rows, bests_high, "Data-free")
    return out_lines


def _build_compare_table(
    rdf: pd.DataFrame,
    set_id: str,
    optimizer: str,
    victim_acc: Optional[float],
    row_sort_key,
    all_single_default: bool,
) -> Optional[str]:
    """Side-by-side LaTeX table comparing baseline vs augmented results.

    Columns: Method | Label | Acc | Acc+Aug | Fid | Fid+Aug
    Each attack appears on exactly one row. Cells without a counterpart show
    '--'. Δ is computed implicitly (reader can compare adjacent cells).
    Bolding: best mean within each (label, column) leaderboard.
    Returns None if no attack has both flavours populated.
    """
    sub_no = rdf[(rdf["Set"] == set_id) & (rdf["Optimizer"] == optimizer) & (rdf["Augmentation"] == "")]
    sub_yes = rdf[(rdf["Set"] == set_id) & (rdf["Optimizer"] == optimizer) & (rdf["Augmentation"] != "")]
    if sub_yes.empty:
        # No augmentation data → side-by-side comparison would be all '--'.
        # Skip; the master will fall back to the baseline-only table.
        return None

    # Build keyed maps for fast lookup
    def _key(r):
        return (r["Display"], r["QueryType"], r["IsLow"])

    no_map = {_key(r): r for r in sub_no.to_dict("records")}
    yes_map = {_key(r): r for r in sub_yes.to_dict("records")}

    all_keys = sorted(set(no_map) | set(yes_map))
    if not all_keys:
        return None

    rows_combined: List[dict] = []
    for k in all_keys:
        display, qtype, is_low = k
        r_no = no_map.get(k)
        r_yes = yes_map.get(k)
        rows_combined.append({
            "Display": display,
            "QueryType": qtype,
            "IsLow": is_low,
            "Accuracy": (r_no or {}).get("Accuracy", ""),
            "Agreement": (r_no or {}).get("Agreement", ""),
            "Accuracy_Aug": (r_yes or {}).get("Accuracy", ""),
            "Agreement_Aug": (r_yes or {}).get("Agreement", ""),
        })

    # Compute bests within (tier, label)
    def _best(tier_rows, col):
        return _best_mean(pd.DataFrame(tier_rows), col)

    def _bests_for(tier_rows: List[dict]) -> Dict[str, Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]]:
        out_: Dict[str, Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]] = {}
        for label in ("Soft", "Hard"):
            sub = [r for r in tier_rows if r["QueryType"] == label]
            sub_df = pd.DataFrame(sub) if sub else pd.DataFrame(columns=["Accuracy", "Accuracy_Aug", "Agreement", "Agreement_Aug"])
            out_[label] = (
                _best(sub_df, "Accuracy") if not sub_df.empty else None,
                _best(sub_df, "Accuracy_Aug") if not sub_df.empty else None,
                _best(sub_df, "Agreement") if not sub_df.empty else None,
                _best(sub_df, "Agreement_Aug") if not sub_df.empty else None,
            )
        return out_

    low_rows = sorted([r for r in rows_combined if r["IsLow"]], key=row_sort_key)
    high_rows = sorted([r for r in rows_combined if not r["IsLow"]], key=row_sort_key)
    best_low = _bests_for(low_rows)
    best_high = _bests_for(high_rows)

    set_slug = set_id.lower().replace("-", "")
    opt_display = {"sgd": "SGD", "adamw": "AdamW"}.get(optimizer, optimizer)
    victim_note = f" Victim test accuracy: {victim_acc*100:.2f}\\%." if victim_acc is not None else ""
    caption = (
        rf"\caption{{Baseline vs.\ Augmentation comparison on {set_id} (Optimizer: {opt_display})."
        rf" \texttt{{+Aug}} columns use SwiftThief-style augmentation pipeline (\%).{victim_note}"
        rf" Best Acc/Fid per Soft/Hard group in each column in \textbf{{bold}}.}}"
    )

    tex: List[str] = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(caption)
    tex.append(f"\\label{{tab:{set_slug}-{optimizer}-compare}}")
    tex.append(r"\begin{tabular}{ll rr rr}")
    tex.append(r"\toprule")
    tex.append(r" & & \multicolumn{2}{c}{Acc} & \multicolumn{2}{c}{Fid} \\")
    tex.append(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}")
    tex.append(r"Method & Label & Base & +Aug & Base & +Aug \\")

    def _tier(rows: List[dict], tier_label: str, bests) -> List[str]:
        if not rows:
            return []
        lines = [r"\midrule",
                 f"\\multicolumn{{6}}{{l}}{{\\textit{{{tier_label}}}}} \\\\",
                 r"\midrule"]
        for r in rows:
            b_acc, b_acc_aug, b_agr, b_agr_aug = bests.get(r["QueryType"], (None, None, None, None))
            lines.append(
                f"{r['Display']} & {r['QueryType']} & "
                f"{_fmt_cell(r['Accuracy'], _is_best(r['Accuracy'], b_acc))} & "
                f"{_fmt_cell(r['Accuracy_Aug'], _is_best(r['Accuracy_Aug'], b_acc_aug))} & "
                f"{_fmt_cell(r['Agreement'], _is_best(r['Agreement'], b_agr))} & "
                f"{_fmt_cell(r['Agreement_Aug'], _is_best(r['Agreement_Aug'], b_agr_aug))} \\\\"
            )
        return lines

    tex += _tier(low_rows, "Pool-based", best_low)
    tex += _tier(high_rows, "Data-free", best_high)
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    return "\n".join(tex)


def _build_compare_md(
    rdf: pd.DataFrame,
    set_id: str,
    optimizer: str,
    row_sort_key,
) -> List[str]:
    """Markdown side-by-side comparison (baseline vs aug) for a (set, optimizer)."""
    sub_no = rdf[(rdf["Set"] == set_id) & (rdf["Optimizer"] == optimizer) & (rdf["Augmentation"] == "")]
    sub_yes = rdf[(rdf["Set"] == set_id) & (rdf["Optimizer"] == optimizer) & (rdf["Augmentation"] != "")]
    if sub_no.empty and sub_yes.empty:
        return []

    def _key(r):
        return (r["Display"], r["QueryType"], r["IsLow"])

    no_map = {_key(r): r for r in sub_no.to_dict("records")}
    yes_map = {_key(r): r for r in sub_yes.to_dict("records")}

    rows_combined: List[dict] = []
    for k in sorted(set(no_map) | set(yes_map)):
        display, qtype, is_low = k
        r_no = no_map.get(k)
        r_yes = yes_map.get(k)
        rows_combined.append({
            "Display": display,
            "QueryType": qtype,
            "IsLow": is_low,
            "Accuracy": (r_no or {}).get("Accuracy", ""),
            "Agreement": (r_no or {}).get("Agreement", ""),
            "Accuracy_Aug": (r_yes or {}).get("Accuracy", ""),
            "Agreement_Aug": (r_yes or {}).get("Agreement", ""),
        })

    out: List[str] = []
    for tier_name, is_low in (("Pool-based", True), ("Data-free", False)):
        rows = sorted([r for r in rows_combined if r["IsLow"] == is_low], key=row_sort_key)
        if not rows:
            continue
        out.append(f"#### {tier_name}\n")
        out.append("| Method | Label | Acc (base) | Acc (+Aug) | Fid (base) | Fid (+Aug) |")
        out.append("|---|---|---|---|---|---|")
        for r in rows:
            out.append(
                f"| {r['Display']} | {r['QueryType']} | "
                f"{r['Accuracy'] or '--'} | {r['Accuracy_Aug'] or '--'} | "
                f"{r['Agreement'] or '--'} | {r['Agreement_Aug'] or '--'} |"
            )
        out.append("")
    return out


if __name__ == "__main__":
    analyze_results()
