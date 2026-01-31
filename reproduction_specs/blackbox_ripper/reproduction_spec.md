# REPRODUCTION SPECIFICATION
**Target:** Black-Box Ripper: Copying Black-Box Models Using Generative Evolutionary Algorithms
**Status:** PENDING REFACTORING

## 1. Global Constants & Configuration Map
*Define the exact variable values mandated by the paper. The Coding Agent must enforce these.*

| Paper Symbol | Description | Required Value/Constraint | Current Code Var | Action |
| :--- | :--- | :--- | :--- | :--- |
| $K$ | Population size | `30` | `mebench/attackers/blackbox_ripper.py:32-36` | **NO CHANGE** |
| $k$ | Elite size | `10` | `mebench/attackers/blackbox_ripper.py:35-37` | **NO CHANGE** |
| $u$ | Latent bound | `3` | `mebench/attackers/blackbox_ripper.py:33-35` | **NO CHANGE** |
| $t$ | Fitness threshold | `0.02` | `mebench/attackers/blackbox_ripper.py:39` | **NO CHANGE** |
| $I$ | Max evolve iters | `10` | `mebench/attackers/blackbox_ripper.py:40` | **NO CHANGE** |
| $z$ | Latent init | $U(-u, u)$ | `mebench/attackers/blackbox_ripper.py:115-119` | **NO CHANGE** |
| $\mathcal{F}$ | Fitness | $-\sum (\hat{y} - y)^2$ | `mebench/attackers/blackbox_ripper.py:287-292` | **NO CHANGE** |
| $\Delta z$ | Mutation | $P_c \leftarrow P_c + \mathcal{N}(0,1)$ | `mebench/attackers/blackbox_ripper.py:314-320` | **NO CHANGE** |
| $E$ | Substitute epochs | `200` | `mebench/attackers/blackbox_ripper.py:52-54` | **NO CHANGE** |
| $B$ | Substitute batch size | `64` | `mebench/attackers/blackbox_ripper.py:424-427` | **VERIFY** |

## 2. Threat Model Constraints (Hard Blockers)
*List operations that are FORBIDDEN by the paper's threat model.*
* **Forbidden:** Access to victim parameters or gradients.
* **Allowed:** Query-only outputs; fitness computed from victim probabilities.
* **Enforcement:** No victim backward pass.

## 3. Refactoring Tasks (Logic Corrections)
*Break down logic errors into discrete, implementable tasks.*

### [TASK-001] Align GAN Backbone with Paper
* **Target Function:** `mebench/attackers/blackbox_ripper.py:_init_models()`
* **Paper Reference:** Section 4.3 (ProGAN/SNGAN backbone)
* **Defect:** Current code uses DCGAN/SNDCGAN generators.
* **Required Logic:**
    * Add ProGAN or SNGAN modules and set as default for Blackbox Ripper.

### [TASK-002] Confirm Substitute Training Optimizer
* **Target Function:** `mebench/attackers/blackbox_ripper.py:_train_substitute()`
* **Paper Reference:** Section 4.3 (Adam optimizer; 200 epochs)
* **Defect:** Uses Adam with configurable LR; verify matches paper defaults.
* **Required Logic:**
    * Ensure Adam LR matches paper's setting and remains constant across epochs.

## 4. Verification Assertions
*Boolean checks the agent should run after refactoring.*
1. `assert population_size == 30 and elite_size == 10`
2. `assert latent_init_is_uniform(-u, u)`
3. `assert max_evolve_iters == 10`
4. `assert gan_backbone in {"progan", "sngan"}`
5. `assert fitness_is_negative_mse_to_onehot`
