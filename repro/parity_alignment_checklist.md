# Parity Alignment Checklist (MAZE / GAME / DFMS)

## MAZE (official_repo_clones/maze)

- [x] Generator zeroth-order perturbation moved to pre-tanh path.
- [x] Zeroth-order gradient estimate scaled by batch size.
- [x] Cosine scheduler iteration estimate aligned with official query formula using `(iter_clone - 1)` term.
- [x] Replay changed from random stored-batch sampling to full-buffer minibatch replay behavior.
- [x] Targeted parity tests pass.

## GAME (official_repo_clones/game_attack)

- [x] Proxy class-space decoupled from victim class-space (`proxy_num_classes` support).
- [x] ACGAN generator/discriminator class dimensions now follow proxy class count.
- [x] TDL proxy-label bounds validated against proxy class count (not victim class count).
- [x] Query buffer accumulation wired into online extraction path.
- [x] `gmd_steps` now applied in the extraction loop.
- [x] ACS class distribution operates on proxy class-space.
- [x] Targeted parity tests pass.
- [x] Fixed student-loss scaling bug in GMD (`loss_res` is no longer multiplied by `beta1`).
- [x] AGU loss scaling aligned closer to official (sum-based `L_res`/`L_bou`, `L_dif` reduction parity).
- [x] AGU discriminator update made optional and default-off for official-like extraction path.
- [ ] Smoke/full reproduction still underperforming (near-random accuracy plateau); runtime parity investigation ongoing.
- [ ] AGU strict parity review pending (official extraction loop behavior vs current discriminator updates).
- [ ] Experimental parity gaps remain (`32x32` resize, half-* substitute architectures, `[-1,1]` preprocess vs benchmark `[0,1]`).
- [ ] Detailed live checklist maintained at `repro/papers/2022_xie_game/GAME_PARITY_CHECKLIST.md`.

## DFMS (official_repo_clones/dfms_hl)

- [x] AutoAugment tensor dtype compatibility fixed (`float32 -> uint8 -> float32` around Equalize path).
- [x] Added regression test for AutoAugment Equalize dtype path.
- [x] Existing basic DFMS tests pass.

## Verification Status

- [x] Local static checks on edited attackers (`py_compile`) pass.
- [x] Local targeted tests for MAZE/GAME/DFMS pass.
- [ ] Remote full reproduction runs complete and table-level tolerance checks pass.

## Remote Execution Commands (server)

```bash
git pull origin <branch>
conda activate mebench
pip install -e ".[dev]"

python repro/run_experiment.py run --paper-id 2021_kariyappa_maze --profile full --device cuda:0 --stages attack,collect,compare --no-live-output
python repro/run_experiment.py run --paper-id 2022_xie_game --profile full --device cuda:0 --stages attack,collect,compare --no-live-output
python repro/run_experiment.py run --paper-id 2022_sanyal_dfms --profile full --device cuda:0 --stages attack,collect,compare --no-live-output
```
