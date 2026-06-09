# Threat Model

This benchmark evaluates *black-box model extraction*: an attacker tries to
train a substitute model that approximates a deployed victim model using only
query access to the victim's prediction interface.

## Adversary

| Capability | Setting |
|---|---|
| Access to the victim | Black-box query access only |
| Output type | Either soft probability vector (`output_mode=soft_prob`) or hard top-1 label (`output_mode=hard_top1`); see per-attack `output_mode` field |
| Number of queries | Bounded by `budget.max_budget`. **One image = one query** (`mebench/core/context.py`, `mebench/oracles/oracle.py:66`) |
| Query payload | A tensor in the victim's expected input scale; preprocessing is path-specific (see "Preprocessing contract" in `README.md`) |
| Concurrent queries | Allowed within the same context; budget is decremented atomically per image |
| Knowledge of victim | None except dataset family and num_classes (declared in config) |
| Auxiliary data | Public surrogate datasets (e.g. ImageNet) for pool-based attacks; data-free attacks generate their own queries |

## Victim

| Property | Setting |
|---|---|
| Forward mode | `eval()` and `torch.no_grad()` (oracle-side) |
| Temperature | `1.0` by default; configurable per victim entry |
| Determinism | Inputs are forwarded as-is; no benchmark-side normalization wrapper. Path-specific preprocessing is the attack's responsibility |
| Side channels | Not exposed (no logits, no gradients, no internal activations) |
| Watermarking / defenses | Out of scope for v1 |

The output-mode constraint is enforced before any attack runs
(`mebench/core/validate.py:117-148`). Soft-only attacks cannot silently fall
back to hard labels and vice versa.

## Budget semantics

- The unit is *images*, not forward passes. A batch of 64 images counts as 64
  queries.
- Gradient-estimation attacks (MAZE, DFME, etc.): every query image used for
  finite-difference probing decrements the budget. Substitute training steps
  do not.
- Data-free attacks: synthetic samples sent to the victim count toward the
  budget; the generator's own update steps do not.
- Validation/seed bootstrapping samples count toward the budget.
- Checkpoints are recorded at user-specified budget thresholds
  (`budget.checkpoints` in the YAML config).

## Out of scope (v1)

- White-box attacks (parameter or gradient access).
- Membership inference and inversion attacks (different threat models).
- Watermarking, adversarial training, or other defensive mechanisms applied
  to the victim.
- Side-channel attacks (timing, hardware, API metadata).
- Attacks that require modifying the victim or its training procedure.

## Reporting policy

Within-set comparison is the primary fairness target (see "Interpreting
SET-A/B/C" in `README.md`). Cross-set comparisons are contextual, not a
single absolute leaderboard, because budgets, victim domains, and surrogate
pool caps differ by set.
