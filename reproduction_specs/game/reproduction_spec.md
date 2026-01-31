# REPRODUCTION SPECIFICATION
**Target:** GAME: Generative-Based Adaptive Model Extraction Attack
**Status:** PENDING REFACTORING

## 1. Global Constants & Configuration Map
*Define the exact variable values mandated by the paper. The Coding Agent must enforce these.*

| Paper Symbol | Description | Required Value/Constraint | Current Code Var | Action |
| :--- | :--- | :--- | :--- | :--- |
| $L_{res}$ | Responsivity | $-\sum \max(0, f_S^i)$ | `mebench/attackers/game.py:324-325` | **VERIFY** |
| $L_{bou}$ | Boundary distance | $p_{top1} - p_{top2}$ | `mebench/attackers/game.py:325-327` | **VERIFY** |
| $L_{adv}$ | Adversarial correction | $-CE(N_S(x), \arg\max N_S(x))$ | `mebench/attackers/game.py:327-329` | **VERIFY** |
| $L_{dif}$ | Prediction difference | $-KL(N_S(x) || N_V(x))$ | `mebench/attackers/game.py:330-344` | **VERIFY** |
| $L_{total}$ | Generator total | $\sum_{i=1}^4 \beta_i L_i$ | `mebench/attackers/game.py:346-350` | **VERIFY** |
| $P_{unc}$ | ACS uncertainty | $1 - \max softmax$ | `mebench/attackers/game.py:471-477` | **VERIFY** |
| $P_{dev}$ | ACS deviation | $KL(N_S(G(z,i))||N_V^*)$ | `mebench/attackers/game.py:434-468` | **UPDATE** |
| $\beta_1..\beta_4$ | Loss weights | Must be explicit | `mebench/attackers/game.py:39-42` | **VERIFY** |
| $B$ | Batch size | `128` | `mebench/attackers/game.py:26` | **NO CHANGE** |

## 2. Threat Model Constraints (Hard Blockers)
*List operations that are FORBIDDEN by the paper's threat model.*
* **Forbidden:** Access to victim parameters/gradients.
* **Allowed:** Victim outputs only; if hard-label, use discriminator soft output for GMD.
* **Enforcement:** GMD uses discriminator soft labels when victim output is hard.

## 3. Refactoring Tasks (Logic Corrections)
*Break down logic errors into discrete, implementable tasks.*

### [TASK-001] Enforce ACGAN Architecture and Dropout
* **Target Function:** `mebench/attackers/game.py:_init_models()`
* **Paper Reference:** Section 3.2 (AC-GAN; discriminator dropout 0.25)
* **Defect:** Uses generic DCGAN modules without dropout.
* **Required Logic:**
    * Replace with AC-GAN generator/discriminator; add dropout `p=0.25` in discriminator.

### [TASK-002] Implement Exact ACS Deviation ($P_{dev}$)
* **Target Function:** `mebench/attackers/game.py:_compute_class_distribution()`
* **Paper Reference:** Eq. 9
* **Defect:** Uses cached averages instead of victim outputs for current samples.
* **Required Logic:**
    * Query victim for $G(z,i)$ samples to compute $KL(N_S||N_V^*)$ when using deviation.

## 4. Verification Assertions
*Boolean checks the agent should run after refactoring.*
1. `assert beta1..beta4 are explicitly configured`
2. `assert discriminator_dropout == 0.25`
3. `assert acs_strategy in {"uncertainty", "deviation"}`
4. `assert p_dev_uses_victim_outputs`
