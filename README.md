# Model Extraction Benchmark

PyTorch benchmark for comparing model extraction attacks under a unified experiment envelope.

[한국어](README-ko.md) • [Contract Guide](Model_Extraction_Benchmark_v1.0.1_Contract_and_Implementation_Guide.md)

## Overview

- Goal: compare attacks against each other, not preserve a legacy two-track evaluation protocol.
- Official protocol: `Track B` only.
- Runtime contract: `engine -> attack.run(ctx) -> checkpoint/final eval -> artifact logging`.
- Budget contract: `1 query = 1 image`.
- Oracle contract: victim runs in `eval()` and `torch.no_grad()` with `temperature=1.0` by default.

## What "Track B only" means

- Every attack is evaluated through its native benchmark runtime.
- The benchmark still unifies victim setup, substitute defaults, budgets, seeds, and reporting.
- The benchmark no longer performs a separate from-scratch retraining protocol at each checkpoint.
- Artifacts and reports are canonicalized around `track_b`.

## Quick Start

```bash
pip install -e ".[dev]"

# smoke
bash scripts/launch/run_smoke.sh cuda:0

# generate configs
IMAGENET_ROOT=D:/imagenet python generate_configs.py

# run matrix
IMAGENET_ROOT=D:/imagenet bash scripts/launch/run_matrix.sh

# aggregate
python aggregate_matrix.py
```

## Core Rules

1. `1 query = 1 image`.
2. Victim and attack `output_mode` must match.
3. Default oracle uses `T=1.0`.
4. Pool-based attacks use the shared benchmark substitute defaults defined by configs.
5. Data-free attacks keep their native loop and are evaluated through the same artifact/reporting contract.

## Preprocessing Contract

- Victim query path: attacker-produced query tensors are forwarded to the victim without an extra benchmark-side normalization wrapper.
- Victim evaluation path: benchmark metrics are computed on the task's public test set using the dataset's official evaluation normalization.
- Surrogate / pool path: attacker-side samples from public surrogate datasets use the surrogate dataset's official preprocessing or normalization for that attack path.
- Data-free path: attacker-side tensors stay in their native data-free scale (typically `[-1,1]`) unless an attack explicitly owns a different internal convention.
- Benchmark rule: preprocessing policy is path-specific. The benchmark does not force one shared normalization transform across victim query, victim evaluation, surrogate training, and data-free generation.

## Benchmark Policy vs. Paper Parity

- Matrix results are benchmark-policy comparisons: attacks run under a shared runtime contract, shared reporting, shared seeds, and setup-level substitute defaults.
- Benchmark-policy results are designed for controlled cross-attack comparison, not as a claim that every attack is reproduced under its original paper training stack.
- Paper-parity reproduction remains a separate objective under `repro/` and should be reported separately when claiming closeness to original paper numbers.

## Interpreting SET-A/B/C

- `SET-A1`, `SET-B1`, and `SET-C1` are benchmark cells with different victim/task regimes.
- Within-set comparisons are the primary fairness target: attacks in the same set share the same victim family, budget policy, output mode constraints, and reporting contract.
- Cross-set results should not be interpreted as a single absolute leaderboard because budgets, victim domains, and surrogate pool caps differ by set.
- Use SET-level rankings and trends for headline benchmark conclusions; treat cross-set comparisons as contextual rather than directly commensurate.

## SET-C1 Substitute Policy

- `SET-C1` uses a fixed substitute-training schedule for all attacks: `batch=128`, `optimizer=sgd(lr=0.1,momentum=0.9,wd=5e-4)`, `scheduler=multistep([0.5,0.75],gamma=0.1)`, `max_epochs=90`.
- Model selection is by best validation loss.
- Early stopping is disabled in practice by setting `patience=max_epochs`, so runs always complete the full 90 epochs before restoring the best validation-loss checkpoint.

## Implemented Attacks

The benchmark currently includes `17` attacks across four families, all evaluated under the same Track-B runtime and reporting contract.

| Family | Count | Coverage |
| :--- | ---: | :--- |
| Baseline | 1 | Reference strategy for pooled querying |
| Active Learning | 6 | Query-efficient sample selection and adaptive acquisition |
| Pool-based / Offline | 2 | Large-scale surrogate labeling and offline training |
| Data-Free / Generative | 8 | Query synthesis without a public surrogate pool |

### Baseline

| Attack | Paper | Key Strategy |
| :--- | :--- | :--- |
| Random | - | Uniform random sampling from the surrogate pool. |

### Active Learning

| Attack | Paper | Key Strategy |
| :--- | :--- | :--- |
| ActiveThief | [Pal et al. (2020)](https://ojs.aaai.org/index.php/AAAI/article/view/5432) | Uncertainty, K-Center, and DFAL-based pool selection. |
| Blackbox Dissector | [Wang et al. (2021)](https://arxiv.org/abs/2105.00623) | Erase-and-score active querying with Grad-CAM style masking. |
| CloudLeak | [Yu et al. (2020)](https://www.ndss-symposium.org/wp-content/uploads/2020/02/24178.pdf) | FeatureFool-style adversarial query generation from public images. |
| InverseNet | [Gong et al. (2021)](https://www.ijcai.org/proceedings/2021/336) | Inversion-inspired sample recovery and staged querying. |
| MARICH | [Karmakar et al. (2023)](https://arxiv.org/abs/2302.08466) | Multi-round hard-label querying with staged sample selection. |
| SwiftThief | [Lee et al. (2024)](https://www.ijcai.org/proceedings/2024/47) | SimSiam-style representation learning plus active selection. |

### Pool-based / Offline

| Attack | Paper | Key Strategy |
| :--- | :--- | :--- |
| KnockoffNets | [Orekondy et al. (2019)](https://arxiv.org/abs/1812.02766) | Bandit-style class sampling and offline substitute training. |
| CopycatCNN | [Correia-Silva et al. (2018)](https://arxiv.org/abs/1806.05476) | Large-scale offline labeling of natural images. |

### Data-Free / Generative

| Attack | Paper | Key Strategy |
| :--- | :--- | :--- |
| DFME | [Truong et al. (2021)](https://arxiv.org/abs/2011.14779) | Min-max data-free extraction with generator/student updates. |
| DFMS | [Sanyal et al. (2022)](https://arxiv.org/abs/2204.11022) | Diverse hard-label query synthesis with staged stealing. |
| DisGUIDE | [Tan et al. (2023)](https://ojs.aaai.org/index.php/AAAI/article/view/26150) | Disagreement-guided data-free extraction with replay. |
| MAZE | [Kariyappa et al. (2021)](https://arxiv.org/abs/2005.03161) | Zeroth-order gradient estimation for query synthesis. |
| ES-Attack | [Yuan et al. (2022)](https://arxiv.org/abs/2009.09560) | Evolutionary query synthesis against black-box victims. |
| GAME | [Xie et al. (2022)](https://link.springer.com/chapter/10.1007/978-3-031-17140-6_28) | Adaptive class-conditioned generation and extraction. |
| Dual Students | [Beetham et al. (2023)](https://arxiv.org/abs/2309.10058) | Two-student disagreement-guided data-free extraction. |
| Blackbox Ripper | [Barbalau et al. (2020)](https://arxiv.org/abs/2010.11158) | Latent-space evolution on pretrained GAN priors. |

## Artifacts

Each seed run writes the same canonical artifact bundle:

```text
runs/<run_name>/<timestamp>/seed_<seed>/
  run_config.yaml
  summary.json
  metrics.csv
  metrics_history.csv
```

- `run_config.yaml`: exact experiment config captured for reproducibility
- `summary.json`: checkpoint metrics keyed by budget under `track_b`
- `metrics.csv`: long-form result rows with `track=track_b`
- `metrics_history.csv`: checkpoint-by-checkpoint metric history

## Methodology

See `Model_Extraction_Benchmark_v1.0.1_Contract_and_Implementation_Guide.md` for the current benchmark contract.

## Contributing

When contributing a new attack or contract change, keep the benchmark surface consistent:

- Implementation: add new attacks under `mebench/attackers/`
- Runtime contract: keep implementations compatible with `AttackRunner.run(ctx)`
- Verification: run `python -m pytest tests/ -q`

## License

MIT. See `LICENSE`.
