# Paper-to-Code Mapping

| Paper item | Paper value | Code target | Mapping |
|---|---|---|---|
| Attack | MAZE | `attack.name` | `maze` |
| Oracle output | soft probabilities | `victim.output_mode` / `attack.output_mode` | `soft_prob` |
| Query budget (paper mode in repo) | CIFAR10: 30M | `budget.max_budget` | `30000000` in `experiment.yaml` |
| Query budget (smoke) | low-resource validation | `budget.max_budget` | `2000` in `experiment_smoke.yaml` |
| Generator/clone steps | `N_G=1`, `N_C=5` | `attack.n_g_steps`, `attack.n_c_steps` | `1`, `5` (full) |
| Gradient approx | forward differences | `attack.grad_approx_m`, `attack.grad_approx_epsilon` | `10`, `0.001` (full) |
| Clone arch | WideResNet-22 style | `substitute.arch`, `substitute.width_mult` | `wideresnet22`, `2` |
