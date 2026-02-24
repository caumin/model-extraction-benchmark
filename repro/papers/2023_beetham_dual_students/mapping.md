# Paper-to-Code Mapping

| Paper item | Paper value | Code target | Mapping |
|---|---|---|---|
| Query budget (CIFAR-10) | 20M | `configs/experiment.yaml:attack.max_budget` | Set to `20000000` |
| Student updates per cycle | 5 | `configs/experiment.yaml:attack.d_iter` | Set to `5` |
| Generator updates per cycle | 1 | `configs/experiment.yaml:attack.g_iter` | Set to `1` |
| Students | 2 | `configs/experiment.yaml:attack.num_students` | Set to `2` |
| Student LR | 0.3 | `configs/experiment.yaml:attack.student_lr` | Set to `0.3` |
| Generator LR | 1e-4 | `configs/experiment.yaml:attack.generator_lr` | Set to `0.0001` |
