# Utils API Minimization Plan

## Goal

Minimize the public `mebench.utils` API to symbols that are required by internal runtime paths.

## Scope

- In scope: `mebench/utils/*.py`, `mebench/utils/__init__.py`
- Out of scope: attack algorithm behavior changes

## Policy

1. Keep symbols that are used by runtime code paths (`mebench/*`, `scripts/*`).
2. Keep symbols needed by tests only if they represent intended public API.
3. Mark uncertain symbols as deprecated before removal.
4. Remove only after one version window (`v1.3` -> `v1.4`).

## Current Classification

### Keep

- `clamp_unit`, `tanh_to_unit`, `unit_to_tanh`, `normalize_input_scale`
- dataloader helpers (`pool_loader_kwargs`, `resolve_*_num_workers`, `load_pool_to_memory`)
- `ChunkedProcessor.process_tensor_chunks`
- `memory_efficient_cat`
- `LearningRateValidator`, `validate_learning_rates`, `auto_fix_learning_rates`

### Deprecated (v1.3), Remove target (v1.4)

- `ChunkedProcessor.stream_dataset_items`
- `ChunkedProcessor.accumulate_chunks`

### Removed in this cleanup slice

- `create_chunked_dataloader`
- `chunked_inference`
- `_global_validator` (unused module global)

## Checklist

- [x] Remove confirmed dead symbols from `chunked.py`
- [x] Remove stale module-level global from `validation.py`
- [x] Update `mebench.utils` exports to match reduced API
- [x] Add deprecation warnings for P2 candidates with explicit removal version
- [ ] Add changelog note before release tag
- [ ] Remove deprecated methods in `v1.4` cleanup PR

## Verification Commands

```bash
python -m pytest tests/test_comprehensive_regression.py::TestChunkedProcessing -q
python -m compileall mebench/utils/chunked.py mebench/utils/validation.py mebench/utils/__init__.py
```
