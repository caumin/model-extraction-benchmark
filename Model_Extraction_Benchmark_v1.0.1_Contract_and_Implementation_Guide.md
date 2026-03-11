# Model Extraction Benchmark v1.0.1 — Contract + Implementation Guide (LLM-Friendly)

**Status:** Updated based on internal review of critical conflicts and implementation gaps.

This document combines:
1) the **benchmark contract** (requirements for comparability), and  
2) an **implementation guide** (code structure, interfaces, validation, logging).

It is intentionally **precise**, **modular**, and **validation-first**.

---

## 0. Goal

Build a paper-grade benchmark that compares model extraction attacks **fairly** across different loop styles:

- **Round-based** (collect queries in rounds, retrain from scratch per round)
- **Online** (train continuously as new queries arrive)
- **Generative / data-free** (train generator + student)

The benchmark must ensure reported results primarily reflect:
- **query quality**
- **query efficiency (budget vs performance)**

and not incidental differences in training dynamics.

**Primary outputs**: budget–performance curves for:
- **Test Accuracy (GT)**
- **Agreement (Fidelity)**
- **Soft-label distance metrics** (for `soft_prob` runs)

Aggregated over **3 seeds** by default (mean ± std).

---

# Part I — Benchmark Contract (Non-negotiable)

## 1. Core Definitions

### 1.1 Query Budget Definition (Critical)

> **1 query = 1 image** sent to the victim oracle **one time**.

- Batched inference is allowed for throughput, but budget MUST decrement by **number of images**.
- There is **no “API call” budget**. Budget is **image-count budget** only.

**Budget checkpoints (queries)**:
- **1k, 10k, 100k, 1000k**

---

### 1.2 Oracle Output Modes

The victim oracle supports:

| Mode | Name | Return |
|---|---|---|
| Soft | `soft_prob` | **Raw probability output** at **temperature T=1.0**: multiclass returns `[N, K]`, single-logit binary returns positive-class probabilities `[N, 1]` |
| Hard | `hard_top1` | **Top-1 class label** |

#### Temperature Contract
- The benchmark fixes **temperature T = 1.0** for `soft_prob`.
- If a paper/setting uses T≠1.0, it must be represented as a **separate, explicit oracle profile** (not the default).

#### Binary Output Contract
- Binary classification is represented as **single-logit binary only**.
- For binary tasks, set `victim.num_classes = 1` and use substitute architectures with a single output logit.
- In `soft_prob`, the oracle returns the positive-class probability with shape `[N, 1]`.
- In `hard_top1`, the oracle returns thresholded labels `0/1`.
- The benchmark does **not** define a separate 2-logit / 2-class softmax binary mode.
- For SewerML matrix experiments, the benchmark uses the defect-vs-normal binary checkpoint rather than the SewerML multilabel model, because the multilabel model is not a good fit for standard extraction-attack evaluation.

---

### 1.3 Metrics (What is Reported)

#### Test Accuracy (GT)
- Evaluate the substitute model on the dataset test split using **ground-truth labels**.

#### Agreement (Fidelity)
- For each test example `x`:
  - Victim prediction: `argmax(victim_prob(x))`
  - Substitute prediction: `argmax(substitute_output(x))`
- Agreement is the fraction of test examples where these predictions match.

#### Soft-Label Distance Metrics (Required for `soft_prob`)
For soft-label runs, report at least one distribution-level metric:

- **Mean KL Divergence**: `KL(victim_prob || substitute_prob)` averaged over test set
- **Mean L1 Distance (MAE)** on probabilities: `mean(|p_v - p_s|)` averaged over test set

> Rationale: Agreement/argmax metrics can hide differences in distribution fidelity. These metrics quantify “how well the probability function is stolen”.

---

## 2. Datasets & Input Policy

### 2.1 Supported Victim Datasets (Experiment-Selectable)

An experiment selects exactly **one** victim dataset profile:

| Victim dataset | Input size | Channels |
|---|---:|---:|
| MNIST | 28×28 | 1 |
| CIFAR-10 (v1.0 standard start) | 32×32 | 3 |
| Sewer-ML | 224×224 | 3 |

### 2.2 Normalization / Preprocessing Assumption
This benchmark assumes a **BlackBox MLaaS** victim interface:
- The attacker does **not** know the victim's internal preprocessing/normalization.
- The attacker sends an image query as-is.
- The MLaaS victim applies its own internal preprocessing/normalization, runs inference, and returns the response.

Scale policy at the attacker boundary:
- Pool-based attacks query/train in surrogate-standard normalized space (`dataset.surrogate_normalization`, default `standard`).
- Data-free attacks query synthesized images in `[-1,1]` scale.
- Data-free victim query path must not apply attacker-side tanh->unit conversion.
- Runtime victim query path uses direct model input (no wrapper-side query normalization).
- Evaluation uses a shared normalized test loader for fair comparison.
- The queried images are used **as-is** for substitute training data (no victim-normalization reconstruction at benchmark boundary).

---

## 3. Data Availability Modes

Each run selects exactly **one** attacker data condition:

| Mode | Name | Description |
|---|---|---|
| Seed | `seed` | Small in-domain seed set (default **100 images**) |
| Surrogate | `surrogate` | Out-of-domain/public data (dataset depends on victim profile) |
| Data-free | `data_free` | No external data; queries come from synthesis/generation |

### 3.1 Seed Default
- Default seed size: **100 images**, sampled from the victim domain.

### 3.2 Surrogate Dataset Policy (No “Forced Unification”)
We do **not** force surrogate datasets of different shapes to match victim input via ad-hoc conversions.
Instead, each experiment chooses surrogate data that **naturally matches** the victim profile.

Recommended mapping:
- Victim = **MNIST (28×28×1)** → surrogates: **EMNIST**, **Fashion-MNIST**
- Victim = **CIFAR-10 (32×32×3)** → surrogates: **SVHN (32×32×3)**, **GTSRB** (only under 32×32×3 conditions)
- Victim = **Sewer-ML (224×224×3)** → surrogate source specified by Sewer-ML profile later

---

## 4. Victim Model Policy

Victims are **pretrained** models loaded externally. The benchmark does **not** mandate a fixed victim at v1.0.

Each run MUST include victim metadata fields:
- `victim_id` (string)
- `input_size` (H, W)
- `channels` (1 or 3)
- `normalization` (mean/std or equivalent)
- `output_modes_supported` (subset of `{soft_prob, hard_top1}`)
- `checkpoint_ref` (path/hash/tag reference)

### 4.1 Victim Determinism Requirements (Mandatory)
- Victim inference MUST run in:
  - `model.eval()` mode
  - `torch.no_grad()` context
- Any stochastic layers (dropout, batchnorm updates) must be disabled via eval mode.

---

## 5. Substitute Model Policy

Substitute architecture is configurable per run (e.g., LeNet / ResNet / MobileNet / VGG).

Rules:
- Substitute training uses attacker-visible query tensors as collected (surrogate-standard normalized tensors for pool-based, `[-1,1]` for data-free), unless an attack explicitly defines an internal conversion step.
- Training augmentations are **disallowed by default**, unless an attack definition explicitly requires them (future extensions).

---

## 6. Attacks Included in v1.0 (MVP)

Attacks supported in v1.0:
1. **Random** (baseline)
2. **ActiveThief**
3. **DFME** (data-free; includes generator training)

### 6.1 Allowed (Attack × Data Mode) Combinations

| Attack | Allowed data modes |
|---|---|
| Random | `seed`, `surrogate` |
| ActiveThief | `seed`, `surrogate` |
| DFME | `data_free` only |

Output modes:
- The framework supports `soft_prob` and `hard_top1`.
- An attack may restrict output modes if required by the original method; restrictions must be validated and recorded in run metadata.

---

# Part II — Standardized Training & Evaluation Protocol

## 7. Snapshot at Budget B

At each budget checkpoint **B** (queries), define:

> **D_B** = labeled dataset from the first **B** oracle-queried images and their oracle outputs.

This snapshot is the basis for standardized evaluation.

---

## 8. Two-Track Reporting (Mandatory)

### Track A (Primary): **SFP — Standard From-Scratch Protocol**
For each checkpoint **B**:
1. Initialize the substitute model with a **fixed initialization seed**
2. Train **from scratch** on **D_B**
3. Evaluate **Test Accuracy (GT)**, **Agreement**, and (if `soft_prob`) **KL/L1 metrics**

This isolates “query quality” by eliminating online/round training dynamic differences.

#### Track A Training Steps Rule (Fixed)
Training steps scale with dataset size:

> **S(B) = ceil(0.2 × B)** steps

#### Track A Batch Size Rule (Mandatory)
Because step-based schedules depend on batch size, Track A MUST use a **fixed** batch size:

> **Track A batch size = 128 (default)**

If a dataset profile requires a different batch size (e.g., Sewer-ML memory constraints), that must be defined in the dataset profile and recorded in run metadata.

#### Track A Initialization Seed Rule
- Track A uses the **same initialization seed** for from-scratch training (consistent, comparable).
- Multi-seed experiments vary the overall run seed; within a run seed, Track A initialization is deterministic.

#### Track A Optimizer/Scheduler Reset (Mandatory)
At each checkpoint B in Track A, the following MUST be re-instantiated:
- model weights (fresh init)
- optimizer state (e.g., momentum buffers)
- LR scheduler state

---

### Track B (Secondary): **AOP — Attacker-Original Protocol**
- Run the attack using its **native loop** (round-based or online).
- Record performance at the same budget checkpoints.
- Track B is supplementary/appendix-style.

---

## 9. Generative Attacks and Track A Compatibility (Required Disclaimer)

**Issue:** For generative/data-free attacks (e.g., DFME), synthesized queries can be **state-dependent**, i.e., optimized for the *current* student (and/or generator) state.  
When evaluating Track A, we retrain a fresh student from scratch on `D_B`. Performance may drop compared to Track B not because queries are “bad”, but because they are **coupled** to the student state at generation time.

**Interpretation Guidance (Disclaimer):**
- If a generative attack yields lower Track A results, interpret this as:
  - the synthesized query set being **state-dependent** and having lower **cross-initialization generalizability**, not necessarily lower usefulness for the original end-to-end attack.
- Track B remains the faithful measure of the **end-to-end** generative attack loop.
- Track A should be read as a measure of **query set generalizability** (transfer across student initializations).

---

## 10. Reproducibility & Seeds

Default: **3 seeds**.

Implementation must control randomness across:
- Python RNG
- NumPy RNG
- Torch CPU/CUDA RNG
- dataset sampling (seed set and any attack sampling)

---

## 11. Artifacts, Logging, and Outputs

### 11.1 Per-Run Required Artifacts
Each run must produce:
- `run_config.yaml` (exact config used)
- `summary.json`
- `metrics.csv` (long format recommended)
- `final_substitute.ckpt` (final only)

### 11.2 Query Data Storage Policy (Updated for Practicality)
Default policy:
- Do **not** persist queried images as a permanent artifact.

However, **temporary caching is permitted** to enable Track A training at large budgets:

- Allowed: **temporary cache** (e.g., on-disk memmap/LMDB/temporary files) for `D_B` construction and Track A training
- Required: cache must be **deleted after the run completes** (or after producing all Track A metrics)
- Required: cache location must be under a run-scoped directory (easy cleanup)
- Required: run logs must record whether temporary caching was used

> Rationale: For B up to 1,000,000, keeping all images in RAM is unrealistic. Temporary caching reconciles Track A with large budgets.

---

# Part III — Implementation Guide (Coding Plan)

## 12. Repository Layout (Recommended)

```
mebench/
  run.py
  configs/
    *.yaml
  core/
    engine.py
    state.py
    types.py
    registry.py
    validate.py
    seed.py
    logging.py
    report.py
  profiles/
    dataset_profiles.py
    victim_profiles.py
  data/
    loaders.py
    sampling.py
    transforms.py
    cache.py              # NEW: temp caching utilities (memmap/LMDB/etc.)
  victims/
    loader.py
    wrapper.py
  oracles/
    oracle.py
  models/
    substitute_factory.py
    nets/
  attackers/
    base.py
    random_baseline.py
    activethief.py
    dfme.py
  training/
    trainer.py
    losses.py
    schedule.py
  eval/
    evaluator.py
    metrics.py
  utils/
    io.py
    hashing.py
    time.py
tests/
  test_contract_validation.py
  test_reproducibility.py
  test_budget_accounting.py
  test_victim_determinism.py
```

Design rule: **Engine is stable**. Attacks are plugins.

---

## 13. Key Data Structures (Types)

### 13.1 Query Representation

All attacks must “propose” a query batch in a unified format:
- `QueryBatch.x`: tensor of images to query
- `QueryBatch.meta`: per-image metadata (origin, pool index, generator step, scores, etc.)

Selection- and generation-based attacks must be interchangeable from the engine’s perspective.

### 13.2 Oracle Output Representation
Normalize oracle output to a single container:
- `kind`: `soft_prob` or `hard_top1`
- `y`: probs `[B,K]` or labels `[B]`

---

## 14. Interfaces (Minimal, Mandatory)

### 14.1 Oracle Interface
- `oracle.query(x_batch) -> OracleOutput`

Mandatory oracle behavior:
- Treat query tensors as external MLaaS inputs; apply victim preprocessing/normalization internally before inference
- Enforce output mode
- Run victim in `eval()` and `no_grad()` (see §4.1)
- Budget accounting is image-count based

### 14.2 Attack Interface (Plugin Contract)
Attacks are stateful but must expose a stable API:

- `propose(k, state) -> QueryBatch`  (returns **k images**)
- `observe(query_batch, oracle_output, state) -> None` (optional)

The engine must NOT know whether the attack uses pool selection, augmentation, or generation.

### 14.3 Engine Responsibilities
The engine:
1. validates config (see §15)
2. constructs profiles (victim/dataset)
3. iterates query collection until max budget
4. records dataset snapshots `D_B` at checkpoints (via buffering + optional temp caching)
5. runs Track A SFP evaluation at checkpoints
6. optionally runs Track B AOP evaluation
7. writes artifacts (summary, csv, ckpt)
8. cleans up temporary caches at the end of the run

---

## 15. Config Schema (YAML) + Validation Rules

### 15.1 Minimal YAML Schema (excerpt)

```yaml
run:
  name: "cifar10_resnet18_soft_random"
  seeds: [0, 1, 2]
  device: "cuda:0"

victim:
  victim_id: "cifar10_resnet18_v1"
  checkpoint_ref: "/path/to/ckpt.pt"
  input_size: [32, 32]
  channels: 3
  normalization:
    mean: [0.4914, 0.4822, 0.4465]
    std:  [0.2470, 0.2435, 0.2616]
  output_mode: "soft_prob"                 # or "hard_top1"
  temperature: 1.0                         # MUST be 1.0 in v1.0 default
  output_modes_supported: ["soft_prob", "hard_top1"]

dataset:
  name: "CIFAR10"
  data_mode: "surrogate"                   # seed | surrogate | data_free
  seed_size: 100
  surrogate_name: "SVHN"
  sewerml_label_mode: "argmax"             # SewerML-only optional: argmax(default) | binary

substitute:
  arch: "resnet18"
  init_seed: 1234                          # fixed init seed for Track A
  trackA:
    batch_size: 128                        # MUST be fixed unless profile override
    steps_coeff_c: 0.2                     # S(B)=ceil(c*B)
  optimizer: {name: "sgd", lr: 0.1, momentum: 0.9, weight_decay: 5e-4}
  loss: {soft: "kl", hard: "ce"}

attack:
  name: "random"                           # random | activethief | dfme
  output_mode: "soft_prob"

budget:
  max_budget: 1000000
  checkpoints: [1000, 10000, 100000, 1000000]

cache:
  enabled: true
  policy: "temporary"
  delete_on_finish: true
```

### 15.2 Validation Rules (Must Fail Fast)
- `budget.checkpoints` increasing and <= `max_budget`
- `victim.output_mode == attack.output_mode`
- `victim.temperature == 1.0` for default `soft_prob` oracle (unless explicitly using a non-default oracle profile)
- if `dataset.name == SewerML` and `dataset.sewerml_label_mode` is provided, it must normalize to `argmax` or `binary` (invalid values fail fast)
- if `dataset.name == SewerML` and `dataset.sewerml_label_mode == binary`, then `victim.num_classes` must be `1`
- `dataset.sewerml_label_mode` is ignored for non-SewerML datasets
- attack × data_mode constraints:
  - DFME requires `data_free`
- Track A constraints:
  - batch size fixed (=128 unless dataset profile override)
  - steps rule uses `c=0.2` by default
  - optimizer/scheduler reset at every checkpoint
- Victim determinism:
  - victim must run in eval/no_grad for all queries

---

## 16. Logging & Result Schema

### 16.1 Files Per Seed-Run
Recommended path:
- `runs/<run_name>/<timestamp>/seed_<s>/`

Required files:
- `run_config.yaml` (resolved config)
- `summary.json`
- `metrics.csv`
- `final_substitute.ckpt` (optional)

### 16.2 metrics.csv (Long Format)
Columns:
- `seed, checkpoint_B, track, acc_gt, agreement, kl_mean, l1_mean, attack, data_mode, output_mode, victim_id, substitute_arch`

(KL/L1 can be empty for hard-only runs.)

---

## 17. Testing Requirements (Minimal)

To prevent silent contract violations:

1) **Budget accounting**
- budget increments exactly by number of images queried

2) **Validation**
- invalid configs fail fast (DFME with surrogate, temp!=1.0 in default profile, etc.)

3) **Reproducibility**
- Track A from-scratch results deterministic under fixed seeds

4) **Victim determinism**
- queries run with `eval()` + `no_grad()`

5) **Cache cleanup**
- temporary caches removed after run completion

---

## 18. Coding Conventions (LLM-Friendly)
- Keep functions small and explicit.
- Avoid hidden global state; pass state/config explicitly.
- Centralize config validation.
- Prefer explicit type containers (`QueryBatch`, `OracleOutput`, etc.).
- Log every checkpoint event with enough metadata to reconstruct paper plots.

---

## 19. v1.1+ (Non-contract)
- Hard-label specialized attacks (Composite/Dissector/DFMS-HL)
- Transferability metrics (FGSM/PGD/CW)
- Defenses as oracle wrappers
- Standardized surrogate sources for Sewer-ML
