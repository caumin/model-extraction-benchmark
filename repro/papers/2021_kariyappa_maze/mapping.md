# Paper-to-Code Mapping

| Paper item | Paper value | Code target | Mapping |
|---|---|---|---|
| Attack | MAZE | `attack.name` | `maze` |
| Victim architecture (repro profile) | same victim profile as DFME reproduction | `victim.arch` / `victim.checkpoint_ref` | `resnet34`, `runs/victims/cifar10-resnet34_8x.pt` |
| Width multiplier note | official `resnet34_8x` name is legacy; implementation is standard ResNet-34 width | `victim.width_mult` | fixed to `1` (`8x` kept only as checkpoint naming/provenance) |
| Oracle output | soft probabilities | `victim.output_mode` / `attack.output_mode` | `soft_prob` |
| Query budget (paper mode in repo) | CIFAR10: 30M | `budget.max_budget` | `30000000` in `experiment.yaml` |
| Query budget (smoke) | low-resource validation | `budget.max_budget` | `2000` in `experiment_smoke.yaml` |
| Victim/clone input scale | victim wrapper normalizes incoming tanh-query tensors; clone uses tanh internal path | fixed runtime policy | tanh query + tanh clone path |
| Generator/clone steps | `N_G=1`, `N_C=5` | `attack.n_g_steps`, `attack.n_c_steps` | `1`, `5` (full) |
| Clone query reuse | first clone update reuses generator-base labels | MAZE run-loop clone phase | first clone step uses cached `(x_base, y_t_base)`; fresh oracle queries for remaining `N_C-1` steps |
| Gradient approx | forward differences | `attack.grad_approx_m`, `attack.grad_approx_epsilon` | `10`, `0.001` (full) |
| Generator optimizer | official repo optimizer profile | `attack.generator_*` | `lr=1e-4`, `momentum=0.9`, `weight_decay=5e-4` |
| Clone arch | WideResNet-22 style | `substitute.arch`, `substitute.width_mult` | `wideresnet22`, `2` |
