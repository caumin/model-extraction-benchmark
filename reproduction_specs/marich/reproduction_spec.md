# REPRODUCTION SPECIFICATION
**Target:** Marich: A Query-efficient Distributionally Equivalent Model Extraction Attack using Public Data
**Status:** INITIAL PORT READY, PAPER TARGET EXTRACTION PENDING

## 1. Global Constants & Configuration Map

| Paper / repo concept | Description | Required Value/Constraint | Current Code Var | Action |
| :--- | :--- | :--- | :--- | :--- |
| staged selection | entropy -> loss-dependent -> gradient filtering | use staged selector for image branch | `mebench/attackers/marich.py:_select_round_indices` | **VERIFY** |
| init points | initial queried pool size | `500` (CIFAR10 ResNet + ResNet18/ImageNet in Table 7) | `init_points` | **SET FOR PAPER PROFILE** |
| rounds | iterative loop count | `10` (Table 7 row) | `rounds` | **SET FOR PAPER PROFILE** |
| per-round budget | base round budget | `750` (Table 7 row) | `round_budget` (`budget`) | **SET FOR PAPER PROFILE** |
| gamma1/gamma2 | stage down-selection factors | `0.8`, `0.8` | `gamma1`, `gamma2` | **NO CHANGE** |
| growth schedule | per-round budget/epoch growth (official code path) | `budget_growth=1.01`, `epochs_growth=1.02` | same names | **SET FOR CODE-FIRST PROFILE** |
| query preprocess | ImageNet32->CIFAR10 normalize before victim query | `marich_imagenet32_cifar10_query` | `query_preprocess_profile` | **SET FOR PAPER PROFILE** |
| substitute init | use pretrained ResNet18 backbone | `true` | `substitute.pretrained` | **SET FOR PAPER PROFILE** |
| best checkpoint rule | select/load best model by validation loss each round | fixed holdout ratio `0.2` from full surrogate pool | `validation_source=pool_split`, `validation_budget_ratio` + trainer `load_best=True` | **SET FOR PAPER PROFILE** |
| query budget interpretation | official MARICH query counts track active extraction samples; fixed validation split labeling cost is treated as pre-available | active-query-only counts | reproduction report + experiment analysis | **MUST DISCLOSE IN RESULTS** |
| output mode | victim response type used by implementation | `hard_top1` only | `run()` mode check + validate contract | **NO CHANGE** |

## 2. Threat Model Constraints (Hard Blockers)

- **Forbidden:** soft-probability oracle path in current MARICH port.
- **Allowed:** hard-label querying over unlabeled public pool data.
- **Reporting requirement:** when `validation_source=pool_split`, document that validation-label
  construction is excluded from reported active-query counts (paper-style accounting).
- **Enforcement:** `mebench/core/validate.py` + runtime check in `marich.py` require `hard_top1`.

## 3. Refactoring Tasks (Remaining Parity Work)

### [TASK-001] Paper Table Target Extraction
- **Target files:** `repro/papers/2023_karmakar_marich/extracted_spec.yaml`
- **Required logic:** transcribe exact paper table targets (dataset/model/budget/metric) for compare-stage verdicting.

### [TASK-002] Reproduction Profile Expansion
- **Target files:** `repro/papers/2023_karmakar_marich/configs/*`
- **Required logic:** add additional profiles for non-CIFAR image setting and text branch if/when benchmark supports MARICH text pipeline.

## 4. Verification Assertions

1. `assert attack.name == "marich"`
2. `assert victim.output_mode == "hard_top1" and attack.output_mode == "hard_top1"`
3. `assert attack.sampling == "all_elg"`
4. `assert attack.init_points == 500`
5. `assert attack.gamma1 == 0.8 and attack.gamma2 == 0.8`
6. `assert attack.query_preprocess_profile == "marich_imagenet32_cifar10_query"`
7. `assert substitute.pretrained is True`
