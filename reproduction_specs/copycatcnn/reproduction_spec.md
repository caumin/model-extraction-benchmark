# REPRODUCTION SPECIFICATION
**Target:** Copycat CNN: Are Random Non-Labeled Data Enough to Steal Knowledge from Black-box Models?
**Status:** PENDING REFACTORING

## 1. Global Constants & Configuration Map
*Define the exact variable values mandated by the paper. The Coding Agent must enforce these.*

| Paper Symbol | Description | Required Value/Constraint | Current Code Var | Action |
| :--- | :--- | :--- | :--- | :--- |
| $D_{NPDD}$ | Transfer set | Non-problem-domain natural images | `mebench/attackers/copycatcnn.py:96-115` | **UPDATE** |
| $E$ | Copycat epochs | `5` | `mebench/attackers/copycatcnn.py:30` | **NO CHANGE** |
| $B$ | Batch size | `128` | `mebench/attackers/copycatcnn.py:24` | **NO CHANGE** |
| $A$ | Augmentations | 22 offline augmentation types | `mebench/attackers/copycatcnn.py:161-204` | **NO CHANGE** |
| $\mathcal{L}$ | Distillation loss | Cross-Entropy on hard labels | `mebench/attackers/copycatcnn.py:274-277` | **NO CHANGE** |
| $\eta$ | Optimizer | SGD with step-down schedule | `mebench/attackers/copycatcnn.py:244-297` | **NO CHANGE** |

## 2. Threat Model Constraints (Hard Blockers)
*List operations that are FORBIDDEN by the paper's threat model.*
* **Forbidden:** Access to victim parameters or gradients.
* **Allowed:** Hard-label queries to victim on NPDD transfer set.
* **Enforcement:** Use argmax labels for training if soft outputs are returned.

## 3. Refactoring Tasks (Logic Corrections)
*Break down logic errors into discrete, implementable tasks.*

### [TASK-001] Enforce NPDD Transfer Set Configuration
* **Target Function:** `mebench/attackers/copycatcnn.py:_get_pool_dataset_config()`
* **Paper Reference:** Section 3.1 (NPDD definition)
* **Defect:** Defaults can point to problem-domain datasets.
* **Required Logic:**
    * Require explicit NPDD dataset in `attack.dataset` and error if missing.

### [TASK-002] Hard-Label Distillation Enforcement
* **Target Function:** `mebench/attackers/copycatcnn.py:_handle_oracle_output()`
* **Paper Reference:** Eq. 6-7 (one-hot hard labels)
* **Defect:** None observed; confirm argmax for soft outputs.
* **Required Logic:**
    * Force `labels = argmax(oracle_output.y)` when `soft_prob`.

## 4. Verification Assertions
*Boolean checks the agent should run after refactoring.*
1. `assert substitute_epochs == 5`
2. `assert len(augs) == 22`
3. `assert transfer_set_is_npdd`
4. `assert hard_label_distillation_only`
