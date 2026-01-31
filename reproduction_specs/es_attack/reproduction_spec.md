# REPRODUCTION SPECIFICATION
**Target:** ES Attack: Model Stealing Against Deep Neural Networks Without Data Hurdles
**Status:** PENDING REFACTORING

## 1. Global Constants & Configuration Map
*Define the exact variable values mandated by the paper. The Coding Agent must enforce these.*

| Paper Symbol | Description | Required Value/Constraint | Current Code Var | Action |
| :--- | :--- | :--- | :--- | :--- |
| $M$ | Student epochs per E-step | `10` | `mebench/attackers/es_attack.py:40` | **NO CHANGE** |
| $m$ | OPT-SYN iterations | `30` | `mebench/attackers/es_attack.py:37` | **NO CHANGE** |
| $\eta_S$ | Student LR (SGD) | `0.01` | `mebench/attackers/es_attack.py:27` | **NO CHANGE** |
| $\eta_G$ | Generator LR (Adam) | `0.001` | `mebench/attackers/es_attack.py:28` | **NO CHANGE** |
| $\eta_{opt}$ | OPT-SYN LR (Adam) | `0.01` | `mebench/attackers/es_attack.py:38` | **NO CHANGE** |
| $\lambda$ | Mode seeking weight | `1.0` | `mebench/attackers/es_attack.py:42` | **NO CHANGE** |
| $L_{KD}$ | KD loss | CE/soft-CE on victim outputs | `mebench/attackers/es_attack.py:212-217` | **VERIFY** |
| $L_{img}$ | DNN-SYN image loss | $CE(f_s(G(z,l)), l)$ | `mebench/attackers/es_attack.py:282-292` | **VERIFY** |
| $L_{ms}$ | Mode seeking | $\frac{\|z_1-z_2\|}{\|G(z_1)-G(z_2)\|}$ | `mebench/attackers/es_attack.py:315-321` | **VERIFY** |
| $L_{opt}$ | OPT-SYN | $CE(f_s(x), y)$ with $y \sim Dir(\alpha)$ | `mebench/attackers/es_attack.py:372-376` | **UPDATE** |

## 2. Threat Model Constraints (Hard Blockers)
*List operations that are FORBIDDEN by the paper's threat model.*
* **Forbidden:** Access to victim parameters or gradients.
* **Allowed:** Victim outputs only, used for knowledge distillation.
* **Enforcement:** No victim backward pass.

## 3. Refactoring Tasks (Logic Corrections)
*Break down logic errors into discrete, implementable tasks.*

### [TASK-001] Use Cross-Entropy in OPT-SYN
* **Target Function:** `mebench/attackers/es_attack.py:_optimize_syn_batch()`
* **Paper Reference:** Eq. 6
* **Defect:** Uses KL divergence instead of cross-entropy to Dirichlet target.
* **Required Logic:**
    ```python
    loss = -(y_target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    ```

### [TASK-002] Enforce ACGAN Generator for DNN-SYN
* **Target Function:** `mebench/attackers/es_attack.py:_init_models()`
* **Paper Reference:** Section III-B-1 (ACGAN-style generator)
* **Defect:** Uses `DCGANGenerator` regardless of class-conditioning.
* **Required Logic:**
    * Replace with ACGAN generator that explicitly conditions on labels.

## 4. Verification Assertions
*Boolean checks the agent should run after refactoring.*
1. `assert student_epochs == 10`
2. `assert opt_steps == 30 and opt_lr == 0.01`
3. `assert generator_lr == 1e-3 and student_lr == 0.01`
4. `assert opt_syn_loss_uses_cross_entropy`
5. `assert generator_is_acgan_conditioned`
