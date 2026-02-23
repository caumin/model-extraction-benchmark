# GAME Refactor Ultrawork Plan

## Objective

Refactor `mebench/attackers/game.py` to achieve paper/official-repo-faithful behavior while preserving benchmark contracts.

- Paper reference: `papers/GAME.pdf`
- Official code reference: `official_repo_clones/game_attack/GAME/methods.py`
- Repro target workflow: `repro/papers/2022_xie_game/`

## Hard Constraints

1. Benchmark contract remains valid (`1 query = 1 image`, metered oracle path only).
2. No silent behavior drift: every parity decision must be documented.
3. Keep tests green for GAME-specific regression files.

## Refactor Phases

### Phase 1 - Structural Cleanup

- [ ] Remove dead/unused GAME internal methods and stale state fields.
- [ ] Make proxy dataset fallback deterministic and explicit (no silent `pass` branch).
- [ ] Simplify attack-state fields to only those consumed by loop logic.

### Phase 2 - Algorithm Parity Alignment

- [ ] Match extraction loop order to official path:
  - query -> append full buffer -> student retrain -> AGU updates -> sampler update
- [ ] Keep AGU default terms and weights in official-compatible form.
- [ ] Keep ACGAN TDL fake-class semantics and label smoothing logic stable.
- [ ] Keep latent width default behavior (`noise_dim = 10 * proxy_num_classes` when implicit).

### Phase 3 - Contract-Safe Metering

- [ ] Ensure all victim accesses in GAME go through `ctx.query` during `run`.
- [ ] Keep `nominal_querybudget` vs `querybudget` behavior explicit and reproducible.
- [ ] Verify no unmetered victim forward path exists in runtime.

### Phase 4 - Repro Preparation (2022_xie_game)

- [ ] Re-check pair-1 and pair-2 config consistency with refactored options.
- [ ] Produce a minimal runbook for smoke and full settings.
- [ ] Update parity checklist/report artifacts in `repro/papers/2022_xie_game/`.

## Verification Checklist

- [ ] `python -m pytest tests/test_game_basic.py -q`
- [ ] `python -m pytest tests/test_game_budget_metering.py -q`
- [ ] `python -m pytest tests/test_game_proxy_class_alignment.py -q`
- [ ] `python -m pytest tests/test_attack_porting_defaults.py -q`
- [ ] `python -m compileall mebench/attackers/game.py`

## Re-audit Loop

If any verification step fails:

1. create a fresh todo/checklist slice,
2. fix root cause,
3. rerun verification,
4. repeat until all checks pass.
