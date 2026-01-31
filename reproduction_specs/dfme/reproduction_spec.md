# REPRODUCTION SPECIFICATION
**Target:** Data-Free Model Extraction
**Status:** PENDING REFACTORING

## 1. Global Constants & Configuration Map
*Define the exact variable values mandated by the paper. The Coding Agent must enforce these.*

| Paper Symbol | Description | Required Value/Constraint | Current Code Var | Action |
| :--- | :--- | :--- | :--- | :--- |
| $n_G$ | Generator steps per cycle | `1` | `mebench/attackers/dfme.py:53` | **NO CHANGE** |
| $n_S$ | Student steps per cycle | `5` | `mebench/attackers/dfme.py:54` | **NO CHANGE** |
| $B$ | Batch size | `256` | `mebench/attackers/dfme.py:46` | **NO CHANGE** |
| $\epsilon$ | Forward-diff step size | `1e-3` | `mebench/attackers/dfme.py:56` | **NO CHANGE** |
| $m$ | Random directions | `1` | `mebench/attackers/dfme.py:55` | **NO CHANGE** |
| $\eta_S$ | Student LR (SGD) | `0.1` | `mebench/attackers/dfme.py:47` | **NO CHANGE** |
| $\eta_G$ | Generator LR (Adam) | `5e-4` | `mebench/attackers/dfme.py:50` | **NO CHANGE** |
| $z$ | Noise distribution | $z \sim \mathcal{N}(0,1)$ | `mebench/attackers/dfme.py:58-60` | **NO CHANGE** |
| $L_{\ell_1}$ | Disagreement loss | $\sum_i |s_i - v_i|$ | `mebench/attackers/dfme.py:221-223` | **VERIFY** |
| $\hat{v}$ | Logit recovery | $\log p_i - \frac{1}{K}\sum_j \log p_j$ | `mebench/attackers/dfme.py:152-155` | **NO CHANGE** |
| $\nabla_{FWD}$ | ZO grad | $\frac{1}{m}\sum_i \frac{L(x+\epsilon u_i)-L(x)}{\epsilon}u_i$ | `mebench/attackers/dfme.py:230-235` | **UPDATE** |

## 2. Threat Model Constraints (Hard Blockers)
*List operations that are FORBIDDEN by the paper's threat model.*
* **Forbidden:** Access to victim parameters or gradients.
* **Allowed:** Query-only access to victim outputs via `ctx.query(...)`.
* **Enforcement:** All gradient estimation uses forward differences with victim outputs only.

## 3. Refactoring Tasks (Logic Corrections)
*Break down logic errors into discrete, implementable tasks.*

### [TASK-001] Enforce DFME Generator Architecture
* **Target Function:** `mebench/attackers/dfme.py:_create_generator()`
* **Paper Reference:** Section 6.2 (3 conv + linear upsampling + BN + ReLU + Tanh)
* **Defect:** Allows `DCGANGenerator` fallback.
* **Required Logic:**
    * If `generator_type != "dfme"`, raise `ValueError`.

### [TASK-002] Remove Extra $d$ Scaling in Gradient Estimate
* **Target Function:** `mebench/attackers/dfme.py:run()`
* **Paper Reference:** Eq. 6 (no dimension scaling)
* **Defect:** Code multiplies gradient by `d` (input dimension).
* **Required Logic:**
    * Use $\frac{1}{m}\sum_i \frac{L(x+\epsilon u_i)-L(x)}{\epsilon}u_i$ without $d$.

### [TASK-003] Centralize Logit Recovery
* **Target Function:** `mebench/attackers/dfme.py:_handle_oracle_output()`
* **Paper Reference:** Section 6.3
* **Defect:** Inline logit recovery duplicates `_to_victim_logits`.
* **Required Logic:**
    * Replace inline computation with `_to_victim_logits()`.

## 4. Verification Assertions
*Boolean checks the agent should run after refactoring.*
1. `assert grad_approx_epsilon == 1e-3`
2. `assert grad_approx_m == 1`
3. `assert n_g_steps == 1 and n_s_steps == 5`
4. `assert batch_size == 256`
5. `assert generator_type == "dfme"`
