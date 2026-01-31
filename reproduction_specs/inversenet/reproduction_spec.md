# REPRODUCTION SPECIFICATION
**Target:** INVERSENET: Augmenting Model Extraction Attacks with Training Data Inversion
**Status:** PENDING REFACTORING

## 1. Global Constants & Configuration Map
*Define the exact variable values mandated by the paper. The Coding Agent must enforce these.*

| Paper Symbol | Description | Required Value/Constraint | Current Code Var | Action |
| :--- | :--- | :--- | :--- | :--- |
| $K_1:K_2:K_3$ | Phase ratios | `0.45 : 0.45 : 0.1` | `mebench/attackers/inversenet.py:38` | **NO CHANGE** |
| $\xi$ | HCSS overshoot | `0.02` | `mebench/attackers/inversenet.py:46` | **NO CHANGE** |
| $m$ | Truncation top-k | `1` | `mebench/attackers/inversenet.py:44` | **NO CHANGE** |
| $C(G_V)$ | Inversion loss | $\mathbb{E}[L(G_V(trunc(F_V(x))), x)]$ (MSE) | `mebench/attackers/inversenet.py:223-229` | **VERIFY** |
| $D$ | HCSS score | $(1+\xi)\Delta x$ (DeepFool) | `mebench/attackers/inversenet.py:452-457` | **VERIFY** |
| $\eta_{inv}$ | Inversion LR | `1e-3` | `mebench/attackers/inversenet.py:34` | **NO CHANGE** |
| $\eta_S$ | Substitute LR | `0.01` | `mebench/attackers/inversenet.py:35` | **NO CHANGE** |
| $B$ | Batch size | `128` | `mebench/attackers/inversenet.py:28` | **NO CHANGE** |

## 2. Threat Model Constraints (Hard Blockers)
*List operations that are FORBIDDEN by the paper's threat model.*
* **Forbidden:** Access to victim parameters/gradients.
* **Allowed:** Soft labels from victim only.
* **Enforcement:** Require `oracle_output.kind == soft_prob`.

## 3. Refactoring Tasks (Logic Corrections)
*Break down logic errors into discrete, implementable tasks.*

### [TASK-001] Enforce Truncation to Top-1
* **Target Function:** `mebench/attackers/inversenet.py:_truncate_logits()`
* **Paper Reference:** Section 3.3 (top-1 truncation)
* **Defect:** None observed; ensure default remains `k=1`.
* **Required Logic:**
    * Keep `truncation_k = 1` unless explicitly overridden.

### [TASK-002] Confirm HCSS Sort Direction
* **Target Function:** `mebench/attackers/inversenet.py:_hcss_select()`
* **Paper Reference:** Eq. 2 (largest distance = high confidence)
* **Defect:** None observed; verify descending sort.
* **Required Logic:**
    * Select samples with largest $D = (1+\xi)\Delta x$.

## 4. Verification Assertions
*Boolean checks the agent should run after refactoring.*
1. `assert truncation_k == 1`
2. `assert phase_ratios == [0.45, 0.45, 0.1]`
3. `assert oracle_output.kind == "soft_prob"`
4. `assert hcss_selects_largest_distance`
