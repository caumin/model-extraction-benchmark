# Paper-to-Code Mapping

| Paper item | Paper value | Code target | Mapping |
|---|---|---|---|
| Attack | DFME | `attack.name` | `dfme` |
| Oracle output | soft probabilities | `victim.output_mode` / `attack.output_mode` | `soft_prob` |
| Victim/query scaling policy | data-free synthetic queries stay tanh-scale at attacker boundary; runtime oracle path consumes tensors as-is | fixed runtime policy | tanh query preserved end-to-end |
| Query budget (paper) | CIFAR10: 20M | `budget.max_budget` | `20000000` in `experiment.yaml` |
| Query budget (smoke) | low-resource validation | `budget.max_budget` | `2000` in `experiment_smoke.yaml` |
| G/S step ratio | `n_G:n_S = 1:5` | `attack.n_g_steps`, `attack.n_s_steps` | `1`, `5` (full) |
| Gradient approx | `m=1`, `epsilon=1e-3` | `attack.grad_approx_m`, `attack.grad_approx_epsilon` | `1`, `0.001` |
| Victim checkpoint source | official pretrained victim | `victim.checkpoint_ref` | `runs/victims/cifar10-resnet34_8x.pt` |
| Width multiplier note | official `resnet34_8x` name is legacy; implementation is standard ResNet-34 width | `victim.width_mult` | fixed to `1` (`8x` kept only as checkpoint naming/provenance) |
| Generator LR note | official repo default differs from paper text | `attack.generator_lr` | `1e-4` (paper body often references `5e-4`) |
| Student arch | ResNet-18 (official repo implementation) | `substitute.arch`, `substitute.width_mult` | `resnet18`, `1` |
