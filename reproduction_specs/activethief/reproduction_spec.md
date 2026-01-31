# REPRODUCTION SPECIFICATION
**Target:** ACTIVETHIEF: Model Extraction Using Active Learning and Unannotated Public Data
**Status:** PENDING REFACTORING

## 1. Global Constants & Configuration Map
*Define the exact variable values mandated by the paper. The Coding Agent must enforce these.*

| Paper Symbol | Description | Required Value/Constraint | Current Code Var | Action |
| :--- | :--- | :--- | :--- | :--- |
| $H_n$ | Entropy (uncertainty) | $H_n = -\sum_j p_j \log p_j$ | `mebench/attackers/activethief.py:211-214` | **VERIFY** |
| $d(x)$ | K-center distance | $\min_{x_m \in L} \|\tilde{y}(x) - \tilde{y}(x_m)\|_2$ | `mebench/attackers/activethief.py:245-286` | **VERIFY** |
| $\alpha_n$ | DFAL perturbation norm | $\alpha_n = \|x_n - \hat{x}_n\|_2$ (DeepFool) | `mebench/attackers/activethief.py:373-437` | **VERIFY** |
| $\rho$ | DFAL pre-filter size | Default to full pool | `mebench/attackers/activethief.py:150-153` | **NO CHANGE** |
| $|S_0|$ | Initial seed size | `0.1 * budget` | `mebench/attackers/activethief.py:53-55` | **NO CHANGE** |
| $N$ | Rounds/iterations | `10` | `mebench/attackers/activethief.py:50-51` | **NO CHANGE** |
| $k$ | Batch size (image) | `150` | `mebench/attackers/activethief.py:60` | **NO CHANGE** |
| $E$ | Max epochs | `1000` | `mebench/attackers/activethief.py:61` | **NO CHANGE** |
| $P$ | Early stopping patience | `100` (images) | `mebench/attackers/activethief.py:62` | **NO CHANGE** |
| $V$ | Validation split | `0.2` of labeled set | `mebench/attackers/activethief.py:508-511` | **NO CHANGE** |
| $\lambda$ | L2 regularization | `0.001` | `mebench/attackers/activethief.py:64` | **NO CHANGE** |
| $p$ | Dropout | `0.1` (CIFAR-10 uses `0.2`) | `mebench/attackers/activethief.py:63` | **UPDATE** |
| $\eta$ | Optimizer | Adam, `lr=0.001` | `mebench/attackers/activethief.py:560-667` (SGD) | **UPDATE** |

## 2. Threat Model Constraints (Hard Blockers)
*List operations that are FORBIDDEN by the paper's threat model.*
* **Forbidden:** Access to victim parameters or gradients.
* **Allowed:** Query-only access to victim outputs and pool dataset inputs.
* **Enforcement:** Substitute retraining must be from scratch each round using labeled queries only.

## 3. Refactoring Tasks (Logic Corrections)
*Break down logic errors into discrete, implementable tasks.*

### [TASK-001] Replace SGD with Adam for Substitute Training
* **Target Function:** `mebench/attackers/activethief.py:train_substitute()`
* **Paper Reference:** Training regime (Adam optimizer)
* **Defect:** Current code uses SGD at `lr=0.001`.
* **Required Logic:**
    ```python
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=self.l2_reg,
    )
    ```

### [TASK-002] Implement Dropout in Substitute Architecture
* **Target Files:**
  - `mebench/attackers/activethief.py` (pass dropout config)
  - `mebench/models/substitute_factory.py` (apply dropout in model definition)
* **Paper Reference:** Training regime (dropout 0.1; CIFAR-10 0.2)
* **Defect:** `self.dropout` is defined but not used in model creation.
* **Required Logic:**
    * Add optional dropout parameter to `create_substitute(...)` and insert dropout layers in classifier/FC blocks.

## 4. Verification Assertions
*Boolean checks the agent should run after refactoring.*
1. `assert initial_seed_size == int(0.1 * budget)`
2. `assert num_rounds == 10`
3. `assert batch_size == 150`
4. `assert optimizer_is_adam and lr == 0.001`
5. `assert dropout_applied_with_p in {0.1, 0.2}`
