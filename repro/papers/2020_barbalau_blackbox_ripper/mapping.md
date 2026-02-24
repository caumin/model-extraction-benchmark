# Paper-to-Code Mapping

| Paper item | Paper value | Code target | Mapping |
|---|---|---|---|
| Attack | Black-Box Ripper | `attack.name` | `blackbox_ripper` |
| Oracle setting | probability outputs | `victim.output_mode`, `attack.output_mode` | `soft_prob` |
| Population size | K=30 | `attack.population_size` | `30` |
| Elite count | k=10 | `attack.elite_size` | `10` |
| Latent initialization range | U(-3, 3) | `attack.latent_bound` | `3.0` |
| Mutation noise | N(0, 1) | `attack.mutation_scale` | `1.0` |
| Fitness threshold | t=0.02 | `attack.fitness_threshold` | `0.02` |
| Max evolutionary iterations | 10 | `attack.max_evolve_iters` | `10` |
| Generator dependency | pretrained backbone checkpoint | `attack.generator_name`, `attack.generator_checkpoint` | `cifar_progan`, `checkpoints/blackbox_ripper/official/cifar_100_6_classes_gan(.pth)` |
| Substitute training loop | epoch-based student updates | `attack.substitute_epochs`, `attack.train_batch_size` | `200`, `64` |
