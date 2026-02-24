# Paper-to-Code Mapping

| Paper item | Paper value | Code target | Mapping |
|---|---|---|---|
| Attack | GAME | `attack.name` | `game` |
| Victim dataset (pair-1) | MNIST | `dataset.name` + `victim_* configs` | `MNIST` |
| Proxy dataset (pair-1) | Fashion-MNIST | `attack.proxy_dataset.surrogate_name` | `FashionMNIST` |
| Victim architecture (pair-1) | LeNet | `victim.arch` | `lenet_mnist` (closest runnable mapping) |
| Victim training epochs (pair-1) | 15 | `configs/victim_train.yaml:train.epochs` | `15` |
| Victim optimizer | Adam, lr=0.001 | `configs/victim_train.yaml:train.optimizer.*` | `adam`, `0.001` |
| Attack query budget (pair-1) | 8k | `attack.max_budget`, `attack.querybudget`, `budget.max_budget` | `8000` |
| Attacker batch size | 1024 | `attack.batch_size` | `1024` |
| Attacker optimizer | Adam | `attack.optimizer` | `adam` |
| Attacker train epochs | 40 | `attack.attack_train_epoch` | `40` |
| Metrics | fidelity, accuracy, relative | `repro collected metrics + spec targets` | mapped to `acc_gt` target and paper table values in `extracted_spec.yaml` |
| Pair-2 dataset family | BelgiumTSC/GTSRB | `extracted_spec.yaml` assumptions | documented as unsupported in current full pipeline |
| Input resolution detail | resized to 32x32 (paper) | current training/data pipeline | uses native MNIST 28x28 for runnable parity; recorded as assumption |
