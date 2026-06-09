# Victim Training and Provenance

The benchmark reads pretrained victim weights from `runs/victims/{victim_id}.pt`.
Inputs are assumed to be in `[0, 1]` (raw `ToTensor()` scaling, no extra
mean/std normalization at the oracle); see `mebench/oracles/oracle.py` and
`scripts/train_victim.py`.

Each tracked checkpoint has a sibling `*.pt.json` that records the recipe
used to train it (`arch`, `dataset`, `batch_size`, `lr`, `optimizer`,
`scheduler`, `epochs`, `seed`, `best_acc`, `best_epoch`).

## Tracked victims

| Checkpoint | Dataset | Arch | Best acc | Source / how to reproduce |
|---|---|---|---|---|
| `cifar10_resnet18_seed0.pt` | CIFAR-10 | resnet18 | see `*.pt.json` | `python scripts/train_victim.py --dataset CIFAR10 --arch resnet18 --epochs <see json> --seed 0` |
| `cifar10_activethief_cnn_seed0.pt` | CIFAR-10 | activethief_cnn | 0.9217 (epoch 283) | `python scripts/train_victim.py --dataset CIFAR10 --arch activethief_cnn --optimizer adam --lr 1e-3 --weight-decay 1e-3 --batch-size 150 --epochs 300 --seed 0` |
| `mnist_lenet_mnist_seed0.pt` | MNIST | lenet_mnist | 0.9872 | `python scripts/train_victim.py --dataset MNIST --arch lenet_mnist --optimizer adam --lr 1e-3 --batch-size 128 --epochs 20 --seed 0` |
| `cifar10-resnet34_8x.pt` | CIFAR-10 | resnet34 (×8) | external | Externally sourced (DFME repo, ResNet34-8x official weights). Not produced by this repo's training scripts |
| `swiftthief_cifar10_victim.pt` | CIFAR-10 | swiftthief victim arch | external | Sourced from the SwiftThief reproduction artifact (see `repro/papers/2024_lee_swiftthief/`) |
| `xie2019_binary-binary-version_1.pth` | GTSRB-binary | xie2019 | external | Sourced from Xie et al. 2019 binary classifier release |

**Sibling backups** (`*.bak`, `*.bak1`, …) are produced automatically by
`scripts/train_victim.py:_backup_if_exists` whenever a checkpoint is
overwritten. They are gitignored and safe to delete.

## Reproducing from scratch

```bash
# CIFAR-10, ResNet18 victim
python scripts/train_victim.py --dataset CIFAR10 --arch resnet18 --seed 0 \
    --epochs 200 --optimizer sgd --lr 0.1 --momentum 0.9 \
    --weight-decay 1e-4 --scheduler cosine --batch-size 256

# MNIST, LeNet victim
python scripts/train_victim.py --dataset MNIST --arch lenet_mnist --seed 0 \
    --epochs 20 --optimizer adam --lr 1e-3 --batch-size 128
```

Outputs:
- `runs/victims/{dataset}_{arch}_seed{N}.pt` (state_dict)
- `runs/victims/{dataset}_{arch}_seed{N}.pt.json` (recipe + best metrics)
- Any pre-existing checkpoint is rotated to `*.bak`, `*.bak1`, … before write.

## Normalization invariant

`scripts/train_victim.py:_build_transforms` keeps tensors in `[0, 1]` (raw
`ToTensor()` for CIFAR / MNIST; `Resize(32)` then `ToTensor()` for LeNet on
MNIST). Do **not** add mean/std normalization to the victim training pipeline:
the oracle assumes raw `[0, 1]` inputs at inference time, and any mismatch
produces silently degraded victim accuracy.

## External checkpoints

For the externally sourced files (`cifar10-resnet34_8x.pt`,
`swiftthief_cifar10_victim.pt`, `xie2019_binary-binary-version_1.pth`),
upstream license terms apply. If redistributing, retain attribution to the
original authors and ensure the derived license is compatible with the
benchmark's MIT license.

## Dataset licensing

- **MNIST**: public domain (Y. LeCun et al.).
- **CIFAR-10 / CIFAR-100**: research use (Krizhevsky et al.). No
  redistribution issue for trained model weights.
- **ImageNet (surrogate)**: academic license; the benchmark does not
  redistribute imagery, only points at a user-provided `IMAGENET_ROOT`.
- **GTSRB / SewerML**: per upstream terms; supply via env vars
  (`SEWERML_ROOT`).
