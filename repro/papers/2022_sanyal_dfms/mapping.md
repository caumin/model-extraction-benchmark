# Paper-to-Code Mapping

| Paper item | Paper value | Code target | Mapping |
|---|---|---|---|
| Attack | DFMS-HL | `attack.name` | `dfms` |
| Oracle output | hard label only | `victim.output_mode` / `attack.output_mode` | `hard_top1` |
| Query budget (paper mode in repo) | CIFAR10: 8M | `budget.max_budget` | `8000000` in `experiment.yaml` |
| Query budget (smoke) | low-resource validation | `budget.max_budget` | `2000` in `experiment_smoke.yaml` |
| Official stage pipeline | enabled | `attack.use_official_stages` | `true` |
| Proxy dataset | CIFAR100 subset | `attack.proxy_dataset.*` | class-subset and max-samples mapped in YAML |
| Stage epoch controls | large in full, tiny in smoke | `attack.*_epochs` | full paperlike defaults + smoke reductions |
