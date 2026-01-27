# Model Extraction Benchmark

<div align="center">

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

**A rigorous, paper-grade benchmark for fair comparison of Model Extraction Attacks.**

[한국어](README-ko.md) • [Paper](https://arxiv.org/abs/2305.14890) • [Documentation](Model_Extraction_Benchmark_v1.0.1_Contract_and_Implementation_Guide.md) • [Agents Guide](AGENTS.md)

</div>

---

## 🎯 Overview

**Model Extraction Benchmark** is a PyTorch-based framework designed to solve the "evaluation crisis" in model stealing research. By enforcing a **strict, validation-first contract**, it ensures that performance differences between attacks reflect actual algorithmic improvements, not inconsistent training recipes or budget accounting.

### Why this benchmark?
- **⚖️ Fair Comparison**: Eliminates "trick" variables (e.g., different substitute architectures, training schedules, or normalization).
- **🔒 Strict Contract**: Enforces **1 query = 1 image**. No hidden API calls.
- **🔄 Two-Track Evaluation**:
  - **Track A (Standard From-Scratch)**: Isolates *query quality* by retraining substitutes from scratch at every checkpoint.
  - **Track B (Attacker Original)**: Preserves the *native loop* (online/active) dynamics for fidelity.
- **🧪 Reproducibility**: Deterministic seeds for initialization, data sampling, and victim inference.

---

## 🛡️ Supported Attacks (v1.0)

We support **13 state-of-the-art attacks**, faithfully re-implemented from their original papers:

| Category | Attack | Paper | Key Strategy |
| :--- | :--- | :--- | :--- |
| **Baseline** | **Random** | - | Uniform random sampling from pool. |
| **Active Learning** | **ActiveThief** | [Pal et al. (2020)](https://arxiv.org/abs/2002.05646) | Uncertainty, K-Center, and DFAL sampling. |
| | **Blackbox Dissector** | [Wang et al. (2021)](https://arxiv.org/abs/2105.13733) | Grad-CAM based erasing & selection. |
| | **CloudLeak** | [Yu et al. (2020)](https://ndss-symposium.org/ndss-paper/cloudleak-large-scale-deep-learning-models-stealing-publicly-available-images/) | Adversarial example generation (FeatureFool). |
| | **InverseNet** | [He et al. (2021)](https://arxiv.org/abs/2104.04324) | Inversion-based sample recovery. |
| | **SwiftThief** | [Miao et al. (2024)](https://arxiv.org/abs/2405.15286) | Contrastive learning (SimSiam) + Active selection. |
| **Data-Free / Generative** | **DFME** | [Truong et al. (2021)](https://arxiv.org/abs/2010.12758) | GAN-based query synthesis (Min-Max game). |
| | **DFMS** | [Sanyal et al. (2022)](https://arxiv.org/abs/2205.12760) | Diverse query synthesis with entropy maximization. |
| | **MAZE** | [Kariyappa et al. (2021)](https://arxiv.org/abs/2005.03161) | Zeroth-order gradient estimation. |
| | **ES-Attack** | [Zhang et al. (2022)](https://arxiv.org/abs/2209.11797) | Evolutionary strategy for query synthesis. |
| | **GAME** | [Zhang et al. (2023)](https://arxiv.org/abs/2301.12759) | Adaptive category selection & GAN training. |
| **Hybrid / Other** | **KnockoffNets** | [Orekondy et al. (2019)](https://arxiv.org/abs/1812.02766) | Reinforcement learning (Bandit) for class selection. |
| | **CopycatCNN** | [Correia-Silva et al. (2018)](https://arxiv.org/abs/1806.05476) | Large-scale offline augmentation of natural data. |
| | **Blackbox Ripper** | [Barbalau et al. (2020)](https://arxiv.org/abs/2003.04884) | Latent space evolution on pre-trained GANs. |

---

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
python generate_configs.py

# 2. Run experiments (sequentially or parallelize via shell)
bash run_matrix.sh

# 3. Aggregate results into CSV/LaTeX
python aggregate_matrix.py
```

---

## 📂 Directory Structure

```
model-extraction-benchmark/
├── mebench/                 # Core package
│   ├── attackers/           # Attack implementations (Plugins)
│   ├── core/                # Engine, State, Validation (Stable)
│   ├── models/              # Substitute architectures, GANs
│   ├── oracles/             # Victim model wrappers
│   └── eval/                # Metrics & Evaluators
├── configs/                 # YAML Configuration files
│   ├── matrix/              # Generated full-benchmark configs
│   └── debug/               # Minimal configs for testing
├── runs/                    # Experiment outputs (metrics, logs)
├── data/                    # Datasets (CIFAR, MNIST, etc.)
├── papers/                  # Reference papers (PDFs)
└── tests/                   # Contract validation tests
```

---

## 📜 Contract & Methodology

For a deep dive into the benchmark's philosophy, precise budget definitions, and "Track A vs Track B" protocol, please read the **[Contract & Implementation Guide](Model_Extraction_Benchmark_v1.0.1_Contract_and_Implementation_Guide.md)**.

### Key Rules
1.  **Budget**: `1 query` = `1 image`. Batched queries count as `batch_size`.
2.  **Oracle**: Default `soft_prob` uses Temperature `T=1.0`. `hard_top1` returns labels.
3.  **Determinism**: Victims run in `eval()`/`no_grad()`. Seeds are fixed for Track A.

---

## 🤝 Contributing

We welcome new attacks! Please ensure your implementation follows the [Attack Interface](AGENTS.md#attack-interface) and passes the contract tests.

1.  Implement `BaseAttack` in `mebench/attackers/`.
2.  Add a config in `configs/debug/`.
3.  Run `pytest tests/test_contract_validation.py`.

---

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details. (Note: Some attack implementations are based on their respective original open-source repositories).
