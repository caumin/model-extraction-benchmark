# Paper-to-Code Mapping

| Paper item | Paper value | Code target | Mapping |
|---|---|---|---|
| Attack | MAZE | `attack.name` | `maze` |
| Oracle output | soft probabilities | `victim.output_mode` / `attack.output_mode` | `soft_prob` |
| Query budget (paper mode in repo) | CIFAR10: 30M | `budget.max_budget` | `30000000` in `experiment.yaml` |
| Query budget (smoke) | low-resource validation | `budget.max_budget` | `2000` in `experiment_smoke.yaml` |
| Victim/clone input scale | `[-1,1]` path from official MAZE code | `victim.input_scale_mode`, `attack.clone_input_scale_mode` | `tanh`, `tanh` |
| Generator/clone steps | `N_G=1`, `N_C=5` | `attack.n_g_steps`, `attack.n_c_steps` | `1`, `5` (full) |
| Clone query reuse | first clone update reuses generator-base labels | MAZE run-loop clone phase | first clone step uses cached `(x_base, y_t_base)`; fresh oracle queries for remaining `N_C-1` steps |
| Gradient approx | forward differences | `attack.grad_approx_m`, `attack.grad_approx_epsilon` | `10`, `0.001` (full) |
| Generator optimizer | SGD with momentum/weight decay | `attack.generator_*` | `lr=1e-4`, `momentum=0.9`, `weight_decay=5e-4` |
| Clone arch | WideResNet-22 style | `substitute.arch`, `substitute.width_mult` | `wideresnet22`, `2` |
