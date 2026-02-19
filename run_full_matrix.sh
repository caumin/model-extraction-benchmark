#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

bash "$ROOT_DIR/archive/20260220_legacy_launchers/run_full_matrix.sh" "$@"
