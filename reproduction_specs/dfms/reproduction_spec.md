# REPRODUCTION SPECIFICATION
**Target:** Towards Data-Free Model Stealing in a Hard Label Setting
**Status:** PENDING REFACTORING

## 1. Global Constants & Configuration Map
*Define the exact variable values mandated by the paper. The Coding Agent must enforce these.*

| Paper Symbol | Description | Required Value/Constraint | Current Code Var | Action |
| :--- | :--- | :--- | :--- | :--- |
| $B$ | Batch size | `128` | `mebench/attackers/dfms.py:26` | **NO CHANGE** |
| $\eta_C$ | Clone LR (SGD) | `0.1` | `mebench/attackers/dfms.py:27` | **NO CHANGE** |
| $\eta_G$ | Generator LR (Adam) | `2e-4` | `mebench/attackers/dfms.py:28` | **NO CHANGE** |
| $\eta_D$ | Discriminator LR (Adam) | `2e-4` | `mebench/attackers/dfms.py:29` | **NO CHANGE** |
| $\lambda_{div}$ | Diversity weight | CIFAR-10: `500`; CIFAR-100: `100` | `mebench/attackers/dfms.py:38-40` | **VERIFY** |
| $\lambda_{cert}$ | Certainty weight | `1.0` | `mebench/attackers/dfms.py:40` | **NO CHANGE** |
| $n_C$ | Init query count | `50,000` (CIFAR-10) | `mebench/attackers/dfms.py:42` | **UPDATE** |
| $E_{pre}$ | GAN pretrain epochs | `200` | `mebench/attackers/dfms.py:45-46` | **NO CHANGE** |
| $L_{div}$ | Diversity loss | $\sum_j \alpha_j \log \alpha_j$ | `mebench/attackers/dfms.py:300-303` | **VERIFY** |
| $L_{cert}$ | Certainty loss | $-\frac{1}{N}\sum H(p_i)$ | `mebench/attackers/dfms.py:304-305` | **VERIFY** |
| $L_G$ | Generator loss | $L_{adv}+\lambda_{div}L_{div}+\lambda_{cert}L_{cert}$ | `mebench/attackers/dfms.py:282-309` | **VERIFY** |
| $L_C$ | Clone loss | CE on hard labels | `mebench/attackers/dfms.py:346-350` | **VERIFY** |

## 2. Threat Model Constraints (Hard Blockers)
*List operations that are FORBIDDEN by the paper's threat model.*
* **Forbidden:** Access to victim parameters/gradients; victim is hard-label black box.
* **Allowed:** Only top-1 labels from victim.
* **Enforcement:** Convert soft outputs to argmax labels.

## 3. Refactoring Tasks (Logic Corrections)
*Break down logic errors into discrete, implementable tasks.*

### [TASK-001] Align Initial Query Count $n_C$
* **Target Function:** `mebench/attackers/dfms.py:__init__()`
* **Paper Reference:** Section 3.5 (CIFAR-10 uses 50,000 init queries)
* **Defect:** Default `init_nc=1000`.
* **Required Logic:**
    * Set dataset-specific defaults: CIFAR-10/100 use `init_nc=50000` unless explicitly configured.

### [TASK-002] Verify Generator Architecture Depth
* **Target Function:** `mebench/attackers/dfms.py:_init_models()`
* **Paper Reference:** DCGAN with 5-layer transposed conv for 32x32
* **Defect:** `DCGANGenerator` uses default upsamples; verify depth.
* **Required Logic:**
    * For 32x32 outputs, force 5-layer upsampling if not already.

## 4. Verification Assertions
*Boolean checks the agent should run after refactoring.*
1. `assert init_nc == 50000 for CIFAR-10`
2. `assert diversity_weight in {500.0, 100.0}`
3. `assert loss_g == adv + lambda_div * L_div + lambda_cert * L_cert`
