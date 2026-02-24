# Paper-to-Code Mapping

| Paper item | Paper value | Code target | Mapping |
|---|---|---|---|
| Attack | MARICH | `attack.name` | `marich` |
| Oracle access | hard-label API setting supported | `victim.output_mode`, `attack.output_mode` | `hard_top1` |
| Query source | public pool data | `dataset.data_mode`, `dataset.surrogate_name` | `surrogate`, `IMAGENET` |
| Query preprocessing | ImageNet32->CIFAR10 normalize on query tensors | `attack.query_preprocess_profile` | `marich_imagenet32_cifar10_query` |
| Validation holdout | fixed 20% split from full surrogate pool (official-style) | `attack.validation_source`, `attack.validation_budget_ratio` | `pool_split`, `0.2` |
| Query-count semantics | official reported query counts exclude validation-label construction from fixed pool split | interpretation note in report/spec | active extraction count only |
| Selection strategy | entropy/loss/gradient staged sampling | `attack.sampling` | `all_elg` |
| Initial query points | 500 (CIFAR10 ResNet + ResNet18/ImageNet row in Table 7) | `attack.init_points` | `500` |
| Round budget base | 750 (CIFAR10 ResNet + ResNet18/ImageNet row in Table 7) | `attack.budget` | `750` |
| Gamma parameters | gamma1=0.8, gamma2=0.8 | `attack.gamma1`, `attack.gamma2` | `0.8`, `0.8` |
| Rounds | 10 | `attack.rounds` | `10` |
| Epochs per round | 8 | `attack.epochs` | `8` |
| Optimizer LR | 0.02 | `attack.lr`, `substitute.optimizer.lr` | `0.02` |
| Round growth | official code path growth (`budget*=1.01`, `epochs*=1.02`) | `attack.budget_growth`, `attack.epochs_growth` | `1.01`, `1.02` |
| Substitute model | image classifier substitute | `substitute.arch` | `resnet18` |
| Substitute init | ImageNet-pretrained backbone initialization | `substitute.pretrained` | `true` |
| Victim width | CIFAR10 ResNet34_8x official checkpoint | `victim.width_mult` | `8` |
