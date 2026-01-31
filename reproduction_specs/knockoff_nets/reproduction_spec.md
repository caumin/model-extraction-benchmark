# REPRODUCTION SPECIFICATION
**Target:** Knockoff Nets: Stealing Functionality of Black-Box Models
**Status:** PENDING REFACTORING

## 1. Global Constants & Configuration Map
*Define the exact variable values mandated by the paper. The Coding Agent must enforce these.*

| Paper Symbol | Description | Required Value/Constraint | Current Code Var | Action |
| :--- | :--- | :--- | :--- | :--- |
| $R^{cert}$ | Certainty reward | $p_{k1} - p_{k2}$ | `mebench/attackers/knockoff_nets.py:142-143` | **VERIFY** |
| $R^{div}$ | Diversity reward | $\sum_k \max(0, y_{t,k} - \bar{y}_{t-\Delta,k})$ | `mebench/attackers/knockoff_nets.py:145-149` | **VERIFY** |
| $R^{L}$ | Loss reward | $\mathcal{L}(y_t, \hat{y}_t)$ | `mebench/attackers/knockoff_nets.py:152-159` | **VERIFY** |
| $\pi$ | Policy (softmax) | $\pi_t(z) = e^{H_t(z)} / \sum e^{H_t(z)}$ | `mebench/attackers/knockoff_nets.py:180-191` | **VERIFY** |
| $H$ | Bandit update | $H_{t+1}(z)=H_t(z)+\alpha(r-\bar{r})(1-\pi(z))$ | `mebench/attackers/knockoff_nets.py:189-193` | **VERIFY** |
| $\alpha$ | Bandit rate | $1/N(z)$ | `mebench/attackers/knockoff_nets.py:185-189` | **NO CHANGE** |
| $\Delta$ | Reward window | `100` | `mebench/attackers/knockoff_nets.py:32` | **NO CHANGE** |
| $w_C$ | Certainty weight | `1.0` | `mebench/attackers/knockoff_nets.py:33` | **NO CHANGE** |
| $w_D$ | Diversity weight | `1.0` | `mebench/attackers/knockoff_nets.py:34` | **NO CHANGE** |
| $w_L$ | Loss weight | `1.0` | `mebench/attackers/knockoff_nets.py:35` | **NO CHANGE** |
| $B$ | Batch size | `128` | `mebench/attackers/knockoff_nets.py:28` | **NO CHANGE** |

## 2. Threat Model Constraints (Hard Blockers)
*List operations that are FORBIDDEN by the paper's threat model.*
* **Forbidden:** Access to victim parameters or gradients.
* **Allowed:** Query-only victim outputs; policy uses reward signals derived from victim outputs.
* **Enforcement:** Rewards computed from outputs only.

## 3. Refactoring Tasks (Logic Corrections)
*Break down logic errors into discrete, implementable tasks.*

### [TASK-001] Reward Rescaling to [0, 1]
* **Target Function:** `mebench/attackers/knockoff_nets.py:_handle_oracle_output()`
* **Paper Reference:** Section 4.1.2 (rewards rescaled to [0,1])
* **Defect:** Current code sums raw rewards without rescaling.
* **Required Logic:**
    * Normalize each reward component to `[0, 1]` before weighted sum.

### [TASK-002] Handle Unlabeled Action Space
* **Target Function:** `mebench/attackers/knockoff_nets.py:_load_pool()`
* **Paper Reference:** Section 4.1.2 (actions need not be ground-truth labels)
* **Defect:** Current code uses dataset labels to define action classes.
* **Required Logic:**
    * Provide fallback action groups when pool labels are unavailable.

## 4. Verification Assertions
*Boolean checks the agent should run after refactoring.*
1. `assert rewards_are_rescaled_to_unit_interval`
2. `assert alpha == 1.0 / count`
3. `assert bandit_update_matches_eq2_eq3`
