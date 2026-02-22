import json
import os
from pathlib import Path

import pytest


def _load_metric_parity_spec() -> dict:
    spec_path = os.environ.get("MEBENCH_METRIC_PARITY_SPEC", "").strip()
    if not spec_path:
        pytest.skip("Set MEBENCH_METRIC_PARITY_SPEC to run metric parity assertions")

    path = Path(spec_path)
    if not path.exists():
        pytest.fail(f"Metric parity spec file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        pytest.fail("Metric parity spec must be a JSON object")
    return payload


def test_metric_parity_spec_schema() -> None:
    payload = _load_metric_parity_spec()
    required = {
        "attack",
        "metric",
        "official_value",
        "mebench_value",
        "tolerance_abs",
    }
    missing = required.difference(payload.keys())
    assert not missing, f"Missing required keys: {sorted(missing)}"


def test_metric_parity_within_tolerance() -> None:
    payload = _load_metric_parity_spec()
    official_value = float(payload["official_value"])
    mebench_value = float(payload["mebench_value"])
    tolerance_abs = float(payload["tolerance_abs"])

    assert tolerance_abs >= 0.0
    assert abs(mebench_value - official_value) <= tolerance_abs, (
        f"{payload['attack']}::{payload['metric']} parity failed: "
        f"official={official_value}, mebench={mebench_value}, tol={tolerance_abs}"
    )
