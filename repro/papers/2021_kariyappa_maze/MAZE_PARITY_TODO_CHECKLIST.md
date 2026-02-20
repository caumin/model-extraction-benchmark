# MAZE Parity Todo + Checklist

## Todo (Completed)

- [x] Compare `mebench/attackers/maze.py` against `official_repo_clones/maze/src/attacks/maze.py` loop order.
- [x] Compare zeroth-order gradient formula against `official_repo_clones/maze/src/attacks/attack_utils.py`.
- [x] Align clone-phase replay append timing with official implementation.
- [x] Re-run MAZE regression tests after parity edits.

## Algorithmic Parity Checklist

- [x] **Generator update order**: `G` step before clone step, matching official.
- [x] **ZO gradient objective sign**: maximize disagreement with `-KL(T || S)` during generator step.
- [x] **ZO estimator scale**: `(d / m) * (loss_pert - loss_base) / epsilon`, then batch averaging.
- [x] **Clone update order**: first clone step reuses base generator batch labels, remaining steps query fresh synthetic samples.
- [x] **Replay source timing**: store replay sample once per outer iteration (official-style latest `(x, T(x))`).
- [x] **Replay training rule**: iterate replay loader and stop on short batch, matching official behavior.
- [x] **Optimizer family**: SGD + cosine scheduler path maintained for paper-profile runs.
- [x] **Budget gateway**: all target accesses are routed through `ctx.query` to preserve benchmark image-count accounting.

## Benchmark-Contract Compatibility Checklist

- [x] Oracle query path remains `BenchmarkContext.query` (no direct victim calls).
- [x] State budget and checkpoint hooks remain intact.
- [x] MAZE tests pass after parity update.
