# Paper-to-Code Mapping

| Paper item | Paper value | Code target | Mapping |
|---|---|---|---|
| Attack | DFMS-HL | `attack.name` | `dfms` |
| Victim architecture | ResNet-18 (selected paper setting) | `victim.arch` | `resnet18` |
| Victim checkpoint source | provided trained victim | `victim.checkpoint_ref` | `runs/victims/cifar10_resnet18_seed0.pt` |
| Oracle output | hard label only | `victim.output_mode` / `attack.output_mode` | `hard_top1` |
| Victim input scaling | checkpoint-aligned inference scale | `victim.input_scale_mode` / `input_scale_mode` | `unit` in full+smoke+victim train/eval configs |
| Internal DFMS model scale | official scripts normalize model inputs | `attack.internal_input_scale_mode` | `tanh` in full+smoke configs |
| Query budget (paper mode in repo) | CIFAR10: 8M | `budget.max_budget` | `8000000` in `experiment.yaml` |
| Query budget (smoke) | low-resource validation | `budget.max_budget` | `2000` in `experiment_smoke.yaml` |
| Official stage pipeline | enabled | `attack.use_official_stages` | `true` |
| Proxy dataset | CIFAR100 subset | `attack.proxy_dataset.*` | class-subset and max-samples mapped in YAML |
| Stage epoch controls | large in full, tiny in smoke | `attack.*_epochs` | full paperlike defaults + smoke reductions |
