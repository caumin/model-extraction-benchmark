# Paper-to-Code Mapping

| Paper item | Paper value | Code target | Mapping |
|---|---|---|---|
| Attack | DisGUIDE | `attack.name` | `disguide` |
| Main objective | `L_G = L_D + lambda L_div` | `mebench/attackers/disguide.py` | disagreement + diversity generator loss |
| Query budget (CIFAR-10) | 20M | `budget.max_budget` | `20000000` in `configs/experiment.yaml` |
| Query budget (CIFAR-100) | 10M | `budget.max_budget` | available as alternate profile later |
| Victim architecture | ResNet-34-8x | `victim.arch`, `victim.width_mult` | `resnet34`, `8` |
| Victim input space | tanh (`[-1,1]`) | `victim.input_scale_mode` | `tanh` |
| Ensemble clones | >= 2 | `attack.ensemble_size` | `2` |
| Replay | Classic | `attack.replay`, `attack.rep_iter` | `Classic`, `3` |
| Student LR | 0.03 | `attack.student_lr` | `0.03` |
| Generator LR | `1e-4` | `attack.generator_lr` | `0.0001` |
| Budget tail handling | full-iteration budget chunks | `attack.strict_iteration_budget` | `true` |
