# REPRODUCTION SPECIFICATION
**Target:** Black-Box Dissector: Toward Erasing-Based Hard-Label Model Stealing
**Status:** PENDING REFACTORING

## 1. Global Constants & Configuration Map
*Define the exact variable values mandated by the paper. The Coding Agent must enforce these.*

| Paper Symbol | Description | Required Value/Constraint | Current Code Var | Action |
| :--- | :--- | :--- | :--- | :--- |
| $\alpha_k^c$ | Grad-CAM weights | $\alpha_k^c = \frac{1}{Z} \sum_{i,j} \partial F^c/\partial A^k_{ij}$ | `mebench/attackers/blackbox_dissector.py:98-100` | **VERIFY** |
| $S_{Grad-CAM}^c$ | CAM heatmap | $S^c = ReLU(\sum_k \alpha_k^c A^k)$ | `mebench/attackers/blackbox_dissector.py:98-101` | **VERIFY** |
| $\psi$ | CAM erasing | Erase region sampled by CAM | `mebench/attackers/blackbox_dissector.py:204-249` | **VERIFY** |
| $\Pi(x)$ | Best erased variant | $\arg\min_i \hat{f}(\psi_i(x))_{y_0}$ | `mebench/attackers/blackbox_dissector.py:612-637` | **VERIFY** |
| $\text{MSP}$ | Selection on erased | $\max_j \hat{f}(\Pi(x))_j$ | `mebench/attackers/blackbox_dissector.py:631-642` | **VERIFY** |
| $y_p$ | Pseudo-label | $y_p = \frac{1}{N}\sum_i \hat{f}(\psi_i(x))$ | `mebench/attackers/blackbox_dissector.py:857-885` | **VERIFY** |
| $\mathcal{L}$ | Total loss | $L = L_{sup} + L_{consistency}$ | `mebench/attackers/blackbox_dissector.py:834-895` | **VERIFY** |
| $N$ | Erasing variants | `10` | `mebench/attackers/blackbox_dissector.py:286` | **NO CHANGE** |
| $s_l$ | Erasing area lower | `0.02` | `mebench/attackers/blackbox_dissector.py:288-289` | **NO CHANGE** |
| $s_h$ | Erasing area upper | `0.4` | `mebench/attackers/blackbox_dissector.py:289-290` | **NO CHANGE** |
| $r_1$ | Aspect ratio lower | `0.3` | `mebench/attackers/blackbox_dissector.py:291` | **NO CHANGE** |
| $r_2$ | Aspect ratio upper | `3.3` | `mebench/attackers/blackbox_dissector.py:292` | **NO CHANGE** |
| $B$ | Batch size | `128` | `mebench/attackers/blackbox_dissector.py:297` | **VERIFY** |
| $\eta$ | Base LR | $0.02 * (B/128)$ | `mebench/attackers/blackbox_dissector.py:298` | **UPDATE** |
| $W$ | Weight decay | $5e-4$ for small datasets; `0` otherwise | `mebench/attackers/blackbox_dissector.py:302` | **UPDATE** |
| $T$ | Training epochs | `200` | `mebench/attackers/blackbox_dissector.py:299` | **UPDATE** |
| $Q$ | Iterative budgets | `{0.1K..30K}` | `mebench/attackers/blackbox_dissector.py:305-463` | **NO CHANGE** |

## 2. Threat Model Constraints (Hard Blockers)
*List operations that are FORBIDDEN by the paper's threat model.*
* **Forbidden:** Use of victim probabilities; only hard labels are available.
* **Allowed:** Substitute soft outputs for selection/consistency.
* **Enforcement:** Require `oracle_output.kind == hard_top1`.

## 3. Refactoring Tasks (Logic Corrections)
*Break down logic errors into discrete, implementable tasks.*

### [TASK-001] Apply Paper LR Schedule and Epoch Count
* **Target Function:** `mebench/attackers/blackbox_dissector.py:train_substitute()`
* **Paper Reference:** Section 4.1 (LR 0.02 * B/128, step decay every 60 epochs, 200 epochs)
* **Defect:** Current defaults are `lr=0.1` and `max_epochs=1000`.
* **Required Logic:**
    * Set `lr = 0.02 * (batch_size / 128)`.
    * Apply decay `lr *= 0.1` at epochs 60 and 120.
    * Set `max_epochs = 200`.

### [TASK-002] Dataset-Specific Weight Decay
* **Target Function:** `mebench/attackers/blackbox_dissector.py:train_substitute()`
* **Paper Reference:** Section 4.1 (weight decay 5e-4 for small datasets; 0 otherwise)
* **Defect:** Weight decay fixed at `5e-4`.
* **Required Logic:**
    * Set `weight_decay=5e-4` for CIFAR-10/SVHN; otherwise `0`.

## 4. Verification Assertions
*Boolean checks the agent should run after refactoring.*
1. `assert n_variants == 10`
2. `assert max_epochs == 200`
3. `assert lr == 0.02 * (batch_size / 128)`
4. `assert selection_uses_msp`
5. `assert pseudo_labels_are_mean_over_variants`
