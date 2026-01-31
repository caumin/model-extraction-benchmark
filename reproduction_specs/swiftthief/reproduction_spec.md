# REPRODUCTION SPECIFICATION
**Target:** SwiftThief: Enhancing Query Efficiency of Model Stealing by Contrastive Learning
**Status:** PENDING REFACTORING

## 1. Global Constants & Configuration Map
*Define the exact variable values mandated by the paper. The Coding Agent must enforce these.*

| Paper Symbol | Description | Required Value/Constraint | Current Code Var | Action |
| :--- | :--- | :--- | :--- | :--- |
| $L_c^{self}$ | SimSiam loss | $-\mathbb{E}[z_i^\top z'_j]$ | `mebench/attackers/swiftthief.py:77-95` | **VERIFY** |
| $L_c^{soft}$ | Soft supervised loss | Entropy-weighted alignment | `mebench/attackers/swiftthief.py:98-129` | **VERIFY** |
| $\eta_{ij}$ | Weighting term | Eq. 3 (entropy + cosine) | `mebench/attackers/swiftthief.py:109-121` | **VERIFY** |
| $L_c^{reg}$ | FGSM regularizer | Adversarial CL on minority | `mebench/attackers/swiftthief.py:132-156` | **VERIFY** |
| $I$ | Outer iterations | `10` | `mebench/attackers/swiftthief.py:84` | **NO CHANGE** |
| $\epsilon$ | FGSM constraint | `0.01` | `mebench/attackers/swiftthief.py:87` | **NO CHANGE** |
| $\lambda_2$ | Regularizer weight | `0.01` | `mebench/attackers/swiftthief.py:86` | **NO CHANGE** |
| $a'$ | Projection dim | `2048` | `mebench/attackers/swiftthief.py:88` | **NO CHANGE** |
| $E$ | CL epochs | `40` | `mebench/attackers/swiftthief.py:98` | **NO CHANGE** |
| $B$ | CL batch size | `256` | `mebench/attackers/swiftthief.py:94` | **NO CHANGE** |
| $\sigma$ | KDE bandwidth | `1.0` | `mebench/attackers/swiftthief.py:91` | **NO CHANGE** |
| $s_j$ | Rare-class score | $\sum_{x_i \in Q_{y_n}} \kappa(f_r(x_j)-f_r(x_i),\sigma)$ | `mebench/attackers/swiftthief.py:610-612` | **VERIFY** |
| Switch cond | Entropy→rare | $B-|Q| \le N_R(\mu-\mu_R)$ | `mebench/attackers/swiftthief.py:507-512` | **VERIFY** |

## 2. Threat Model Constraints (Hard Blockers)
*List operations that are FORBIDDEN by the paper's threat model.*
* **Forbidden:** Access to victim parameters/gradients.
* **Allowed:** Soft labels from victim for queried set $Q$; self-supervised CL on unlabeled set $U$.
* **Enforcement:** CL losses use substitute features and victim outputs only.

## 3. Refactoring Tasks (Logic Corrections)
*Break down logic errors into discrete, implementable tasks.*

### [TASK-001] Verify $\eta_{ij}$ Weight Formula
* **Target Function:** `mebench/attackers/swiftthief.py:SoftSupSimSiamLossV17.forward()`
* **Paper Reference:** Eq. 3
* **Defect:** Ensure the coefficient matches entropy-normalized cosine similarity.
* **Required Logic:**
    * $\eta_{ij} = 1_{i \ne j}(1+H(y_i)/\log K)(1+H(y_j)/\log K)\cos(\angle(y_i,y_j))$.

### [TASK-002] Align Rare-Class Switching Condition
* **Target Function:** `mebench/attackers/swiftthief.py:_update_sampling_mode()`
* **Paper Reference:** Eq. 8
* **Defect:** Current threshold uses `len(rare_classes)` proxy; verify against $N_R$ definition.
* **Required Logic:**
    * Implement $B-|Q| \le N_R(\mu-\mu_R)$ with explicit $N_R$.

## 4. Verification Assertions
*Boolean checks the agent should run after refactoring.*
1. `assert I == 10`
2. `assert fgsm_epsilon == 0.01`
3. `assert lambda2 == 0.01`
4. `assert eta_ij_matches_equation_3`
5. `assert rare_class_switch_condition_matches_eq8`
