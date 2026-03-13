# Master Reproduction Specification

## Status

This repository now uses a `Track-B-only` benchmark contract.

## Scope

- Reproduction work should align attacks with the benchmark runtime actually used in `mebench/core/engine.py`.
- The benchmark compares attacks under unified victim, substitute, budget, seed, and reporting settings.
- Historical two-track language is deprecated and should not be used for new work.

## Reproduction Priorities

1. Preserve attack-native loop semantics where they matter.
2. Keep budget accounting exact: `1 query = 1 image`.
3. Keep victim inference deterministic and black-box compliant.
4. Emit canonical `track_b` artifacts only.

## Required Outputs

- `run_config.yaml`
- `summary.json`
- `metrics.csv`
- any attack-specific checkpoints or logs needed for parity debugging

## Verification

- Run attack-specific smoke tests first.
- Verify fresh runs only emit `track_b` in artifacts.
- Verify aggregation scripts consume `track_b` directly.
