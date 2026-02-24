# Paper-to-Code Mapping

| Paper item | Paper value | Code target | Mapping |
|---|---|---|---|
| Attack | Black-Box Dissector | `attack.name` | `blackbox_dissector` |
| Oracle setting | hard-label | `victim.output_mode`, `attack.output_mode` | `hard_top1` |
| Query pool | public/unlabeled image pool | `dataset.data_mode`, `dataset.surrogate_name` | `surrogate`, `IMAGENET` |
| CAM erasing variants | N=10 | `attack.n_variants` | `10` |
| Erase split | 25% erased queries | `attack.erase_rate` | `0.25` |
| Erase geometry | sl/sh/r1/r2 | `attack.sl`, `attack.sh`, `attack.r1`, `attack.r2` | `0.02`, `0.4`, `0.3`, `3.3` |
| Iterative budgets | progressive milestones | `attack.iterative_budgets` | `[1000, 5000, 10000, 20000, 30000]` |
| Substitute training | SGD + long training schedule | `substitute.optimizer`, `substitute.max_epochs` | `sgd`, `200` |
