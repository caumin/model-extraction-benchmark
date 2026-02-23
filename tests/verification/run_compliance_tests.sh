#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
TEST_FILE="tests/verification/test_paper_compliance.py"

if [[ ! -f "$TEST_FILE" ]]; then
  echo "Compliance test file not found: $TEST_FILE"
  exit 1
fi

echo "Running compliance spot checks"
"$PYTHON_BIN" -m pytest "$TEST_FILE::TestDFMECompliance::test_gradient_estimation_no_dimension_scaling" -v
"$PYTHON_BIN" -m pytest "$TEST_FILE::TestSwiftThiefCompliance::test_entropy_weighted_cosine_similarity" -v
"$PYTHON_BIN" -m pytest "$TEST_FILE::TestArchitectureEnforcement::test_progan_enforcement_blackbox_ripper" -v
"$PYTHON_BIN" -m pytest "$TEST_FILE::TestGAMECompliance::test_fresh_victim_queries_for_acs" -v

echo "Running full compliance suite"
"$PYTHON_BIN" -m pytest "$TEST_FILE" -v --tb=short

echo "Compliance tests complete"
