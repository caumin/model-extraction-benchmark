# REPRODUCTION SPECIFICATION
**Target:** Random Baseline (Uniform Pool Sampling)
**Status:** PENDING REFACTORING

## 1. Global Constants & Configuration Map
*Define the exact variable values mandated by the paper. The Coding Agent must enforce these.*

| Paper Symbol | Description | Required Value/Constraint | Current Code Var | Action |
| :--- | :--- | :--- | :--- | :--- |
| $k$ | Queries per step | Engine-provided | `mebench/attackers/random_baseline.py:62-92` | **NO CHANGE** |
| $P$ | Pool sampling | Uniform without replacement until exhausted | `mebench/attackers/random_baseline.py:65-78` | **NO CHANGE** |

## 2. Threat Model Constraints (Hard Blockers)
*List operations that are FORBIDDEN by the paper's threat model.*
* **Forbidden:** Any adaptive selection based on victim outputs.
* **Allowed:** Uniform random sampling from pool only.
* **Enforcement:** `_handle_oracle_output` must remain a no-op.

## 3. Refactoring Tasks (Logic Corrections)
*Break down logic errors into discrete, implementable tasks.*

### [TASK-001] Preserve Uniform Sampling Semantics
* **Target Function:** `mebench/attackers/random_baseline.py:_select_query_batch()`
* **Paper Reference:** Baseline definition
* **Defect:** None observed; ensure no class-conditional bias.
* **Required Logic:**
    * Sample uniformly from `unqueried_indices` until exhausted, then with replacement.

## 4. Verification Assertions
*Boolean checks the agent should run after refactoring.*
1. `assert sampling_is_uniform`
2. `assert no_state_updates_in_handle_oracle_output`
