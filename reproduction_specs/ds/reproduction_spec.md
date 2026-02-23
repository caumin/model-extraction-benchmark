# REPRODUCTION SPECIFICATION
**Target:** Dual Students (ICLR 2023)
**Status:** READY FOR REPRO PIPELINE EXECUTION

## 1. Global Constants & Configuration Map

| Paper Symbol | Description | Required Value/Constraint | Current Code Var | Action |
| :--- | :--- | :--- | :--- | :--- |
| $n_G$ | Generator steps per cycle | `1` | `mebench/attackers/ds.py:g_iter` | **NO CHANGE** |
| $n_S$ | Student steps per cycle | `5` | `mebench/attackers/ds.py:d_iter` | **NO CHANGE** |
| $B$ | Batch size | `256` | `mebench/attackers/ds.py:batch_size` | **NO CHANGE** |
| `eta_G` | Generator LR (Adam) | `1e-4` | `mebench/attackers/ds.py:generator_lr` | **NO CHANGE** |
| `eta_S` | Student LR (SGD) | `0.3` (paper), `0.01` (contract matrix) | `mebench/attackers/ds.py:student_lr` | **CONFIG-DEPENDENT** |
| Students | Number of students | `2` | `mebench/attackers/ds.py:num_students` | **NO CHANGE** |
| Loss (soft mode) | Student objective | `l1` | `mebench/attackers/ds.py:loss_mode` | **NO CHANGE** |

## 2. Threat Model Constraints

* **Forbidden:** Victim gradient/parameter access.
* **Allowed:** Query-only victim interface (`ctx.query`).
* **Enforcement:** Generator step maximizes student disagreement without victim query; student step uses victim outputs.

## 3. Repro Pipeline Mapping

* Repro paper folder: `repro/papers/2023_beetham_dual_students/`
* Full config: `repro/papers/2023_beetham_dual_students/configs/experiment.yaml`
* Smoke config: `repro/papers/2023_beetham_dual_students/configs/experiment_smoke.yaml`
* Preflight id: `2023_beetham_dual_students`

## 4. Verification Assertions

1. `assert attack.name == "ds"`
2. `assert attack.num_students >= 2`
3. `assert attack.g_iter == 1 and attack.d_iter == 5`
4. `assert attack.generator_lr == 1e-4`
5. `assert budget.max_budget == attack.max_budget`
