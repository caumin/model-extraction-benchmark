# REPRODUCTION_REPORT

## Scope

- Paper: ACTIVETHIEF (AAAI 2020)
- Profile: CIFAR-10 victim + ActiveThief uncertainty strategy
- Status: pipeline-first (low-resource local)

## Feasibility

- Strict paper reproduction (ImageNet thief split 100K/20K) is currently blocked locally.
- Blocker: missing local `data/imagenet/train` and `data/imagenet/val` ImageFolder assets.
- Current objective: maintain reproducible set/pipeline and validate end-to-end execution on fallback profile.

## Environment

- OS: pending capture
- Python: pending capture
- PyTorch/CUDA: pending capture

## Commands

```bash
python scripts/train_victim.py --config repro/papers/2020_pal_activethief/configs/victim_train.yaml --epochs 2 --device cuda:0
python scripts/eval_victim.py --config repro/papers/2020_pal_activethief/configs/victim_eval.yaml --device cuda:0
python -m mebench run --config repro/papers/2020_pal_activethief/configs/experiment.yaml --device cuda:0
```

## Notes

- Main paper uses ImageNet-derived thief dataset split (100K/20K).
- Local run profile currently uses CIFAR10 as surrogate because local ImageNet subset is unavailable.
