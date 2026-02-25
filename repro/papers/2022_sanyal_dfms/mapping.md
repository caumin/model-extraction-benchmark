# Paper-to-Code Mapping

| Paper item | Paper value | Code target | Mapping |
|---|---|---|---|
| Attack | DFMS-HL | `attack.name` | `dfms` |
| Victim architecture (repro profile) | same victim profile as DFME reproduction | `victim.arch` | `resnet34` |
| Victim checkpoint source | shared DFME victim checkpoint | `victim.checkpoint_ref` | `runs/victims/cifar10-resnet34_8x.pt` |
| Width multiplier note | paper/checkpoint name contains `8x` | `victim.width_mult` | fixed to `1` (matches runnable code path; `8x` kept as naming/provenance) |
| Oracle output | hard label only | `victim.output_mode` / `attack.output_mode` | `hard_top1` |
| Victim/query scaling policy | data-free synthetic victim queries stay `[-1,1]` through oracle path; runtime oracle path consumes tensors as-is | fixed runtime policy | tanh query preserved end-to-end |
| Internal DFMS model scale | clone/discriminator/model internal path uses tanh scale | fixed runtime policy | tanh |
| Query budget (paper/offical) | CIFAR10: 8M | `budget.max_budget` | `8000000` in `experiment.yaml` |
| Budget semantics note | paper Sec.3.5 equations imply `Total=2*nC+NQ`; with `nC=50k`, `NQ=8M` gives `8.1M` total | benchmark policy | this repo fixes **total** oracle queries to `8,000,000` for fair cross-attack comparison; effectively `NQ` becomes `7.9M` when `nC=50k` is preserved |
| Budget planner mode | fairness-first exact budget execution | `attack.budget_planner.mode` | `paper_fair` (stage targets are query-count based) |
| Stage init query targets | paper Sec.3.5 `nC=50,000` for init clone phases | `attack.budget_planner.nC_target` | `50000` for Stage2 and Stage4 targets |
| Query budget (smoke) | low-resource validation | `budget.max_budget` | `2000` in `experiment_smoke.yaml` |
| Official stage pipeline | enabled | `attack.use_official_stages` | `true` |
| Proxy dataset | CIFAR100 subset | `attack.proxy_dataset.*` | class-subset and max-samples mapped in YAML |
| Stage epoch controls | official-stage defaults | `attack.*_epochs` | `dcgan=200`, `student_init=200`, `degan=100`, `student_degan=200`, `alternate=800` |
