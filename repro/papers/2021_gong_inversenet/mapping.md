# Paper-to-Code Mapping

| Paper item | Paper value | Code target | Mapping |
|---|---|---|---|
| Attack | InverseNet | `attack.name` | `inversenet` |
| Oracle output | hard label | `victim.output_mode` / `attack.output_mode` | `hard_top1` |
| Query budget (paper mode in repo) | 30K | `budget.max_budget` | `30000` in `experiment.yaml` |
| Query budget (smoke) | low-resource validation | `budget.max_budget` | `2000` in `experiment_smoke.yaml` |
| Budget phase ratio | `0.45:0.45:0.1` | `attack.phase_ratios` | `[0.45, 0.45, 0.1]` |
| Coreset + HCSS | enabled | `attack.coreset_seed`, `attack.hcss_*` | mapped directly in YAML |
| Victim/Thief domain | MNIST / EMNIST letters | `dataset.*` | surrogate mode with `EMNIST letters` |
