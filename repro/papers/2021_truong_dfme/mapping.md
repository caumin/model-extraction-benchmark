# Paper-to-Code Mapping

| Paper item | Paper value | Code target | Mapping |
|---|---|---|---|
| Attack | DFME | `attack.name` | `dfme` |
| Oracle output | soft probabilities | `victim.output_mode` / `attack.output_mode` | `soft_prob` |
| Query budget (paper) | CIFAR10: 20M | `budget.max_budget` | `20000000` in `experiment.yaml` |
| Query budget (smoke) | low-resource validation | `budget.max_budget` | `2000` in `experiment_smoke.yaml` |
| G/S step ratio | `n_G:n_S = 1:5` | `attack.n_g_steps`, `attack.n_s_steps` | `1`, `5` (full) |
| Gradient approx | `m=1`, `epsilon=1e-3` | `attack.grad_approx_m`, `attack.grad_approx_epsilon` | `1`, `0.001` |
| Student arch | ResNet-18-8x | `substitute.arch`, `substitute.width_mult` | `resnet18`, `8` |
