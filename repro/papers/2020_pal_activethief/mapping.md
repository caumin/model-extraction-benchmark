# Paper-to-Code Mapping

| Paper item | Paper value | Code target | Mapping |
|---|---:|---|---|
| Query budget | 20K (main curve) | `budget.max_budget` | `20000` (full); mini-run uses `2000` in `experiment.yaml` assumption profile |
| Iterations | 10 | `attack.rounds` | `10` |
| Initial seed ratio | 10% | `attack.initial_seed_ratio` | `0.1` |
| Validation ratio | 20% | `attack.validation_budget_ratio` | `0.2` |
| Batch size (image) | 150 | `attack.batch_size`, `substitute.batch_size` | `150` |
| Optimizer | Adam default | `substitute.optimizer` | `name=adam`, `betas=[0.9,0.999]` |
| Weight decay | 0.001 | `substitute.optimizer.weight_decay` | `0.001` |
| Max epochs | 1000 | `substitute.max_epochs` | `1000` |
| Early stopping patience | 100 | `substitute.patience` | `100` |
| Dropout (CIFAR-10) | 0.2 | `substitute.dropout_prob` | `0.2` |
| Oracle interface | top-1 label | `victim.output_mode`, `attack.output_mode` | `hard_top1` |
| Thief dataset | downsampled ILSVRC subset | `dataset.surrogate_name` | **ASSUMPTION**: set to `CIFAR10` in runnable config because local ImageNet subset unavailable |

## train_victim YAML support

- Added optional `--config <yaml>` support in `scripts/train_victim.py`.
- CLI overrides YAML values when both are present.
- YAML schema used:

```yaml
dataset: CIFAR10 | MNIST
arch: activethief_cnn | resnet18 | resnet20 | classifier | lenet_mnist
seed: 0
victim_id: cifar10_activethief_cnn
out: runs/victims/cifar10_activethief_cnn_seed0.pt
save_metadata: true
train:
  epochs: 300
  batch_size: 150
  scheduler: multistep | cosine | none
  dropout_prob: 0.2
  num_workers: 0
  optimizer:
    name: adam | sgd
    lr: 0.001
    momentum: 0.9
    weight_decay: 0.001
```
