from pathlib import Path

import pandas as pd

from analyze_results import analyze_results


def test_analyze_results_emits_reports_with_current_pandas(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "runs"
    run_dir = root_dir / "SET-A1_random_soft_10m_seed0" / "20260402_120000"
    for seed_name, acc, agreement in [
        ("seed_0", 0.75, 0.80),
        ("seed_1", 0.65, 0.70),
    ]:
        seed_dir = run_dir / seed_name
        seed_dir.mkdir(parents=True)
        (seed_dir / "summary.json").write_text(
            f"""
            {{
              "checkpoints": {{
                "10000000": {{
                  "track_b": {{
                    "acc_gt": {acc},
                    "agreement": {agreement}
                  }}
                }}
              }}
            }}
            """.strip(),
            encoding="utf-8",
        )

    monkeypatch.setattr(pd.DataFrame, "to_markdown", lambda self: "table")

    output_dir = tmp_path / "analysis_results"
    analyze_results(root_dir=str(root_dir), output_dir=str(output_dir))

    latex_report = (output_dir / "report.tex").read_text(encoding="utf-8")
    summary_csv = (output_dir / "final_summary.csv").read_text(encoding="utf-8")

    assert "$\\pm$" in latex_report
    assert "0.7000 ±" in summary_csv
    assert (output_dir / "final_acc_fidelity_matrix_10000000.csv").exists()
