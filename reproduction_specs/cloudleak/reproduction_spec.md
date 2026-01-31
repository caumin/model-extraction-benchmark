# REPRODUCTION SPECIFICATION
**Target:** CloudLeak: Large-Scale Deep Learning Models Stealing Through Adversarial Examples
**Status:** PENDING REFACTORING

## 1. Global Constants & Configuration Map
*Define the exact variable values mandated by the paper. The Coding Agent must enforce these.*

| Paper Symbol | Description | Required Value/Constraint | Current Code Var | Action |
| :--- | :--- | :--- | :--- | :--- |
| $M$ | Margin | $M = \alpha - \text{avg intra-class dist}$ (Eq. 11) | `mebench/attackers/cloudleak.py:521-541` | **VERIFY** |
| $L$ | FeatureFool loss | $\|x'-x\|_2^2 + \lambda \cdot \max(D(\phi(x'),\phi(x_t)) - D(\phi(x'),\phi(x_s)) + M, 0)$ | `mebench/attackers/cloudleak.py:182-191` | **VERIFY** |
| $x'$ | Box constraint | $x' \in [0,1]^n$ | `mebench/attackers/cloudleak.py:174-199` | **VERIFY** |
| $\eta$ | Perturbation bound | $\|x'-x\|_2 < \eta$ | `mebench/attackers/cloudleak.py:174-199` | **VERIFY** |
| $m$ | L-BFGS iters | `20` | `mebench/attackers/cloudleak.py:249-252` | **NO CHANGE** |
| $\lambda$ | Adversarial weight | Small positive | `mebench/attackers/cloudleak.py:255-256` | **VERIFY** |
| $\epsilon$ | Max per-pixel delta | `8/255` | `mebench/attackers/cloudleak.py:257` | **NO CHANGE** |
| $Q_0$ | Initial pool size | `1000` | `mebench/attackers/cloudleak.py:265-266` | **NO CHANGE** |
| $B$ | Batch size | `64` | `mebench/attackers/cloudleak.py:269` | **NO CHANGE** |

## 2. Threat Model Constraints (Hard Blockers)
*List operations that are FORBIDDEN by the paper's threat model.*
* **Forbidden:** Access to victim parameters or gradients.
* **Allowed:** Substitute model for adversarial generation; victim queried only on generated samples.
* **Enforcement:** FeatureFool uses substitute features only.

## 3. Refactoring Tasks (Logic Corrections)
*Break down logic errors into discrete, implementable tasks.*

### [TASK-001] Use Per-Class Margin $M$ in FeatureFool
* **Target Functions:**
  - `mebench/attackers/cloudleak.py:_get_margin_m()`
  - `mebench/attackers/cloudleak.py:FeatureFool.generate_batch()`
* **Paper Reference:** Eq. 11
* **Defect:** Margin is computed in `_get_margin_m` but not used in `generate_batch` (uses fixed `self.margin_m`).
* **Required Logic:**
    * For each source class, compute `M` via `_get_margin_m(class_id)` and pass per-sample `margin_m` into `generate_batch(...)`.

### [TASK-002] Enforce Perturbation-Norm Constraint
* **Target Function:** `mebench/attackers/cloudleak.py:FeatureFool.generate_batch()`
* **Paper Reference:** Eq. 6, 8-10
* **Defect:** Code clamps per-pixel epsilon but does not enforce global $\|x'-x\|_2 < \eta$.
* **Required Logic:**
    * After optimization, project `delta` to satisfy L2 bound if exceeded.

## 4. Verification Assertions
*Boolean checks the agent should run after refactoring.*
1. `assert margin_m_is_computed_per_class`
2. `assert 0.0 <= x_adv.min() and x_adv.max() <= 1.0`
3. `assert perturbation_l2_norm <= eta`
