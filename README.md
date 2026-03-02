# Model Extraction Benchmark

<div align="center">

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

**A rigorous, paper-grade benchmark for fair comparison of Model Extraction Attacks.**

[한국어](README-ko.md) • [Documentation](Model_Extraction_Benchmark_v1.0.1_Contract_and_Implementation_Guide.md)

</div>

---

## 🎯 Overview

**Model Extraction Benchmark** is a PyTorch-based framework designed to solve the "evaluation crisis" in model stealing research. By enforcing a **strict, validation-first contract**, it ensures that performance differences between attacks reflect actual algorithmic improvements, not inconsistent training recipes or budget accounting.

### Key Features
- **📚 13+ Implemented Attacks**: From classical Active Learning to state-of-the-art Data-Free Generative methods.
- **⚖️ Fair Comparison**: Eliminates "trick" variables (e.g., different substitute architectures, training schedules, or normalization).
- **🔒 Strict Contract**: Enforces **1 query = 1 image**. No hidden API calls.
- **🔄 Two-Track Evaluation**:
  - **Track A (Standard From-Scratch)**: Isolates *query quality* by retraining substitutes from scratch at every checkpoint.
  - **Track B (Attacker Original)**: Preserves the *native loop* (online/active) dynamics for fidelity.
- **🧪 Reproducibility**: Deterministic seeds for initialization, data sampling, and victim inference.

### Experiment Sets (Current)

| Set | Victim | Substitute | Surrogate | Input | Query Budget |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SET-A1** | LeNet on MNIST (trained in-project) | LeNet | ILSVRC2012 train (ImageNet) | `1x28x28` | `10k` (pool/hybrid), `10m` (synthetic) |
| **SET-B1** | ResNet34 on CIFAR10 (DFAD official checkpoint) | ResNet18 | ILSVRC2012 train (ImageNet) | `3x32x32` | `20k` (pool/hybrid), `20m` (synthetic) |

Notes:
- Seeds default to `0,1,2` in matrix generation.
- `SET-B1` victim checkpoint follows DFAD: `https://github.com/VainF/Data-Free-Adversarial-Distillation`.
- Matrix configs are generated from `generate_configs.py` into `configs/matrix/`.

### Matrix Hyperparameter Policy (Current)

- **SET-B1 (`substitute.arch=resnet18`)**:
  - For attacks with sufficiently comparable official/paper references, `generate_configs.py` aligns optimizer family and reference LR/batch.
  - Default aligned path uses **batch 512 + linear LR scaling** (throughput-oriented policy for SGD-family training).
  - Exception path (`keep_reference_pair=True`) keeps official/paper LR-batch pairs unchanged for stability-critical attacks.
  - Current keep-reference attacks: `dfme`, `ds`, `maze`, `swiftthief`.
- **SET-A1 (`substitute.arch=lenet_mnist`)**:
  - There is no one-to-one LeNet official profile consistently comparable to this benchmark setup.
  - Matrix generation therefore uses unified heuristic substitute defaults and keeps only attack-semantic knobs.
- Attacks without reliable one-to-one reference conditions in the target setup keep benchmark heuristic defaults (e.g., `activethief`, `marich`, `cloudleak`, `inversenet`, `random`, `dfms`).
- For old-vs-new comparison runs, keep legacy folders with a suffix like `__prealign`; `analyze_results.py` annotates these as `(PRE-ALIGN)` in aggregated reports.

---

## 🛡️ Supported Attacks (v1.0)

We support **17 state-of-the-art attacks**, faithfully re-implemented from their original papers:

| Category | Attack | Paper | Official Repo | Key Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | **Random** | - | - | Uniform random sampling from pool. |
| **Active Learning** | **ActiveThief** | [Pal et al. (2020)](https://ojs.aaai.org/index.php/AAAI/article/view/5432) | [GitHub](https://github.com/iisc-seal/activethief) | Uncertainty, K-Center, and DFAL sampling. |
| | **MARICH** | [Basu et al. (2023)](https://arxiv.org/abs/2302.08466) | [GitHub](https://github.com/Debabrota-Basu/marich) | Multi-stage active querying (entropy/loss/gradient). |
| | **Blackbox Dissector** | [Wang et al. (2021)](https://arxiv.org/abs/2105.00623) | [GitHub](https://github.com/yxwang-10/Blackbox-Dissector) | Grad-CAM based erasing & selection. |
| | **CloudLeak** | [Yu et al. (2020)](https://www.ndss-symposium.org/wp-content/uploads/2020/02/24178.pdf) | [GitHub](https://github.com/yunyuntsai/DNN-Model-Stealing) | Adversarial example generation (FeatureFool). |
| | **InverseNet** | [Gong et al. (2021)](https://www.ijcai.org/proceedings/2021/336) | - | Inversion-based sample recovery. |
| | **SwiftThief** | [Lee et al. (2024)](https://www.ijcai.org/proceedings/2024/47) | [GitHub](https://github.com/ku-air/SwiftThief) | Contrastive learning (SimSiam) + Active selection. |
| **Data-Free / Generative** | **DFME** | [Truong et al. (2021)](https://arxiv.org/abs/2011.14779) | [GitHub](https://github.com/cake-lab/datafree-model-extraction) | GAN-based query synthesis (Min-Max game). |
| | **Dual Students (DS)** | [Beetham et al. (2023)](https://arxiv.org/abs/2309.10058) | [GitHub](https://github.com/James-Beetham/dual_students) | Two-student disagreement-driven data-free extraction. |
| | **DFMS** | [Sanyal et al. (2022)](https://arxiv.org/abs/2204.11022) | [GitHub](https://github.com/val-iisc/Hard-Label-Model-Stealing) | Diverse query synthesis with entropy maximization. |
| | **DisGUIDE** | [Rosenthal et al. (2023)](https://ojs.aaai.org/index.php/AAAI/article/view/26150) | [GitHub](https://github.com/lin-tan/disguide) | Disagreement-guided data-free extraction with replay/diversity losses. |
| | **MAZE** | [Kariyappa et al. (2021)](https://arxiv.org/abs/2005.03161) | [GitHub](https://github.com/sanjaykariyappa/MAZE) | Zeroth-order gradient estimation. |
| | **ES-Attack** | [Yuan et al. (2022)](https://arxiv.org/abs/2009.09560) | - | Evolutionary strategy for query synthesis. |
| | **GAME** | [Xie et al. (2022)](https://link.springer.com/chapter/10.1007/978-3-031-17140-6_28) | [GitHub](https://github.com/xythink/game-attack) | Adaptive category selection & GAN training. |
| **Hybrid / Other** | **KnockoffNets** | [Orekondy et al. (2019)](https://arxiv.org/abs/1812.02766) | [GitHub](https://github.com/tribhuvanesh/knockoffnets) | Reinforcement learning (Bandit) for class selection. |
| | **CopycatCNN** | [Correia-Silva et al. (2018)](https://arxiv.org/abs/1806.05476) | [GitHub](https://github.com/jeiks/Stealing_DL_Models) | Large-scale offline augmentation of natural data. |
| | **Blackbox Ripper** | [Barbalau et al. (2020)](https://arxiv.org/abs/2010.11158) | [GitHub](https://github.com/antoniobarbalau/black-box-ripper) | Latent space evolution on pre-trained GANs. |

---

Implementation provenance (paper + official repo port links) is documented in
`docs/reference/ATTACK_REFERENCES.md`.

### Known Improvement Needed

- `MARICH` currently follows the paper/original staged protocol in `mebench/attackers/marich.py`.
- It is not yet fully harmonized with the shared pool-based benchmark protocol used for strict cross-attack fairness.
- Planned follow-up: integrate MARICH into a unified pool-based protocol layer (selection/training/evaluation schedule alignment).
- Planned follow-up: add a supplementary transferability evaluation protocol that measures victim robustness to white-box adversarial examples generated from checkpointed substitutes (post-hoc evaluation, budget-isolated from extraction queries).

## ⚡ Installation

```bash
git clone https://github.com/caumin/model-extraction-benchmark.git
cd model-extraction-benchmark

# Install in editable mode
pip install -e .

# Install dev dependencies (testing, linting)
pip install -e ".[dev]"
```

---

## ✅ Quick Start (Public Workflow)

```bash
# 1) Smoke run (recommended first)
bash scripts/launch/run_smoke.sh cuda:0

# 2) Generate matrix configs
# Default ImageNet surrogate root is D:/imagenet
# Change IMAGENET_ROOT to your local path before generation.
IMAGENET_ROOT=D:/imagenet python generate_configs.py

# WSL example
IMAGENET_ROOT=/mnt/c/imagenet python generate_configs.py

# 3) (optional) run full matrix with the same local path
IMAGENET_ROOT=D:/imagenet bash scripts/launch/run_matrix.sh

# 4) Aggregate completed runs
python aggregate_matrix.py
```

---

## 🚀 Usage

### 1. Run a Single Experiment
Execute a specific attack configuration. Artifacts (metrics, logs) are saved to `runs/`.

```bash
python -m mebench run --config configs/cifar10_resnet18_soft_dfme_10k_seed0.yaml --device cuda:0
```

### 2. Run the Full Benchmark Matrix
Reproduce the full v1.0 benchmark results (Matrix Protocol).

```bash
# 1. Generate all configuration files
IMAGENET_ROOT=D:/imagenet python generate_configs.py

# 2. Run experiments (sequentially or parallelize via shell)
IMAGENET_ROOT=D:/imagenet bash scripts/launch/run_matrix.sh

# 3. Aggregate results into CSV/LaTeX
python aggregate_matrix.py
```

Official launcher scripts are in `scripts/launch/`.
`IMAGENET_ROOT` should be changed to each user's local ImageNet path.

### 3. Smoke Run (Recommended First)

```bash
bash scripts/launch/run_smoke.sh cuda:0
```

### 4. Public Release Note

This public distribution is scoped to the `mebench` framework workflows above.
Private/internal research assets such as `papers/`, `repro/`, and `official_repo_clones/`
are intentionally excluded from the open-source release.

---

## 📂 Directory Structure

```
model-extraction-benchmark/
├── mebench/                 # Core benchmark package
├── configs/
│   └── smoke/               # Minimal tracked configs
├── scripts/
│   └── launch/              # Official launcher scripts
├── docs/
│   └── reference/
├── tests/                   # Contract and regression tests
├── generate_configs.py
└── aggregate_matrix.py
```

## Public Release Scope

- Public release targets only the `mebench` experiment framework and related runnable/testable assets.
- `papers/`, `repro/`, and `official_repo_clones/` are intentionally excluded from distribution.
- Local runtime artifacts (`data/`, `runs/`, logs, checkpoints, generated reports) are not part of source distribution.
- Non-public local planning/configuration files are excluded via `.gitignore`.

---

## 📜 Contract & Methodology

For a deep dive into the benchmark's philosophy, precise budget definitions, and "Track A vs Track B" protocol, please read the **[Contract & Implementation Guide](Model_Extraction_Benchmark_v1.0.1_Contract_and_Implementation_Guide.md)**.

For implementation details and provenance of specific attacks, see `docs/reference/ATTACK_REFERENCES.md`.

### Key Rules
1.  **Budget**: `1 query` = `1 image`. Batched queries count as `batch_size`.
2.  **Oracle**: Default `soft_prob` uses Temperature `T=1.0`. `hard_top1` returns labels.
3.  **Determinism**: Victims run in `eval()`/`no_grad()`. Seeds are fixed for Track A.
4.  **BlackBox MLaaS Input Contract**: The attacker does not know victim normalization. Query images are sent as-is and forwarded directly to victim inference (no runtime wrapper transform at query time). Pool-based attacks query/train in surrogate-standard normalized space (`dataset.surrogate_normalization`, default `standard`); data-free attacks query in `[-1,1]` and must not apply attacker-side tanh->unit conversion on the victim query path. Evaluation uses a shared, normalized test loader for fair comparison.

### 🔄 Protocol v1.2: Track Logic & Control Experiments
To ensure fair attribution of performance gains, we enforce **Protocol v1.2** branching rules:
- **Track A (Query Focus)**: Attacks that innovate on *selection* must use the **Unified Substitute Trainer** (Standard Loop).
- **Track B (System Focus)**: Attacks with tightly coupled generation/training loops (e.g., Data-Free GANs) retain their native loop.
- **Control Experiments**: All Track B/Hybrid attacks **must** report a "Loop Contribution" control experiment (Standard Loop vs. Native Loop) to isolate the benefit of custom training recipes from the query strategy itself.

*Protocol branch rules are summarized above and enforced in code/config validation.*

---

## 🤝 Contributing

We welcome new attacks! Please follow the attack interface under `mebench/attackers/` and pass contract tests.

1.  Implement `BaseAttack` in `mebench/attackers/`.
2.  Add a config in `configs/debug/`.
3.  Run `pytest tests/test_contract_validation.py`.

---

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details. (Note: Some attack implementations are based on their respective original open-source repositories).
