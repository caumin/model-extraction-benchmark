#!/usr/bin/env bash
# Clone the official author repositories used for parity audits under repro/.
# These are intentionally not vendored to keep the benchmark repo small and to
# avoid bundling third-party licenses. Re-run any time after pulling main.
#
# Existing target directories are skipped. Use --force to re-clone.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)/repro"
FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

clone() {
  local url="$1"
  local dest="$2"
  if [[ -d "$dest" && "$FORCE" -eq 0 ]]; then
    echo "[skip] $dest exists"
    return 0
  fi
  if [[ -d "$dest" && "$FORCE" -eq 1 ]]; then
    echo "[force] removing $dest"
    rm -rf "$dest"
  fi
  echo "[clone] $url -> $dest"
  git clone --depth 1 "$url" "$dest"
}

clone https://github.com/iisc-seal/activethief.git           "$ROOT/activethief_official"
clone https://github.com/yunyuntsai/DNN-Model-Stealing       "$ROOT/cloudleak_official"
clone https://github.com/James-Beetham/dual_students.git     "$ROOT/dual_students_official"
clone https://github.com/Debabrota-Basu/marich.git           "$ROOT/marich_official"
clone https://github.com/yxwang-10/Blackbox-Dissector.git    "$ROOT/Blackbox-Dissector_official"
clone https://github.com/ku-air/SwiftThief.git               "$ROOT/SwiftThief_official"

echo "Done. Clones live under $ROOT/*_official/ (gitignored)."
