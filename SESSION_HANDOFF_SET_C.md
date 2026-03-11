# SET-C SewerML Handoff

## Current Priority

Unblock remote execution of `SET-C1` for the SewerML binary victim.

## Current Intended Contract

- `SET-C1` is the only SewerML matrix setup.
- It uses the defect-vs-normal binary victim only.
- Checkpoint:
  - `runs/victims/xie2019_binary-binary-version_1.pth`
- Victim config:
  - `victim_id: sewerml_xie2019_binary_defect_normal`
  - `num_classes: 1`
  - single-logit binary only
- Dataset config generated for `SET-C1` should include:
  - `sewerml_label_mode: binary`
  - `sewerml_ann_root: D:/Sewer-ML`
  - `sewerml_data_root: D:/Sewer-ML`

## Why SET-C Uses Binary Only

SewerML provides both:

- a multilabel model
- a defect-vs-normal binary model

The benchmark chooses the binary model because the multilabel model is not a good fit for standard model extraction attacks used in this project.

## What Was Already Updated

- `generate_configs.py`
  - `SET-C2` removed
  - `SET-C1` made binary-only
  - real checkpoint path wired in
  - `D:/Sewer-ML` roots hardcoded into generated configs
- docs updated:
  - `README.md`
  - `README-ko.md`
  - `AGENTS.md`
  - `Model_Extraction_Benchmark_v1.0.1_Contract_and_Implementation_Guide.md`
- tests updated:
  - `tests/test_generate_configs_paper_mode.py`
  - `tests/test_contract_validation.py`

## Critical Runtime Bug Found

Generated config is correct, but remote run still failed with:

`FileNotFoundError: SewerML annotation CSV not found. Checked: data\\SewerML\\annotations\\Test13.csv, data\\SewerML\\annotations\\SewerML_Test.csv`

This means the runtime path propagation fix was still missing on the remote code that actually executed.

## Required Runtime Fix

The generated config already contains SewerML roots, but runtime loader call sites must forward them into `get_test_dataloader(...)`.

Files that need the propagation fix:

- `mebench/data/loaders.py`
- `mebench/core/engine.py`
- `mebench/attackers/runner.py`
- `scripts/eval_victim.py`
- `mebench/attackers/temp_activethief.py`
- `tests/test_sewerml_loader.py`

### Expected Code Shape

`mebench/data/loaders.py`

- `get_test_dataloader(...)` must accept:
  - `sewerml_ann_root: Optional[str] = None`
  - `sewerml_data_root: Optional[str] = None`
- In SewerML branch, call:

```python
ann_root, data_root = _resolve_sewerml_roots(
    ann_root=sewerml_ann_root,
    img_root=sewerml_data_root,
)
```

`mebench/core/engine.py`

```python
sewerml_ann_root=config.get("dataset", {}).get("sewerml_ann_root"),
sewerml_data_root=config.get("dataset", {}).get("sewerml_data_root"),
```

`mebench/attackers/runner.py`

```python
sewerml_ann_root=self.state.metadata.get("dataset_config", {}).get("sewerml_ann_root"),
sewerml_data_root=self.state.metadata.get("dataset_config", {}).get("sewerml_data_root"),
```

`scripts/eval_victim.py`

```python
sewerml_ann_root=_cfg_get(dataset_cfg, "sewerml_ann_root", default=None),
sewerml_data_root=_cfg_get(dataset_cfg, "sewerml_data_root", default=None),
```

## Local Verification Already Performed

These passed locally after the propagation fix:

```bash
python -m pytest tests/test_sewerml_loader.py tests/test_generate_configs_paper_mode.py tests/test_contract_validation.py -q
```

Observed result:

- `38 passed`

Also verified generated `SET-C1` config emitted:

- `D:/Sewer-ML`
- `D:/Sewer-ML`
- `runs/victims/xie2019_binary-binary-version_1.pth`

## Important Note About the Environment

There was a tooling issue in this session for git commands:

- the shell wrapper injected Unix `export ...`
- but the shell was Windows `cmd`
- so git actions from the tool failed before `git` itself ran

This is not a repo bug.

## Next Session Checklist

1. Confirm the runtime propagation fix is actually present in the current local/remote checkout.
2. Regenerate configs:

```bash
python generate_configs.py
```

3. Confirm generated config contains:

```yaml
dataset:
  sewerml_ann_root: D:/Sewer-ML
  sewerml_data_root: D:/Sewer-ML
  sewerml_label_mode: binary
victim:
  checkpoint_ref: runs/victims/xie2019_binary-binary-version_1.pth
```

4. Retry remote run:

```bash
python -m mebench run --config configs/matrix/SET-C1_random_soft_20k_seed0.yaml --device cuda:0
```

5. If it still fails, inspect whether the remote checkout is missing the runtime propagation patch rather than the generated config patch.
