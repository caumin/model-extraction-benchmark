# MASTER REPRODUCTION SPECIFICATION: Model Extraction Benchmark

**Date:** January 31, 2026
**Version:** 1.0.0
**Status:** ACTIONABLE
**Scope:** 14 Model Extraction Attack Implementations

---

## 1. Executive Summary

This document serves as the authoritative source of truth for remediating the 14 model extraction attacks currently implemented in the `mebench` repository. A comprehensive analysis of the codebase against original paper specifications has revealed systemic deviations in three key areas:

1.  **Model Architecture Fidelity**: Several generative attacks (DFME, GAME, ES Attack) use generic GAN backbones instead of the specific architectures (ACGAN, specialized ResNets) mandated by their respective papers.
2.  **Algorithmic Logic Errors**: Critical mathematical operations (Loss functions, Gradient estimation, Margin calculations) deviate from the published equations.
3.  **Hyperparameter Drift**: Learning rates, batch sizes, and scheduling parameters often default to standard values rather than the specific configurations required for reproduction.

**Current Status**: All 14 attacks are marked **PENDING REFACTORING**.
**Goal**: Execute the remediation plan below to achieve **100% reproduction compliance**.

---

## 2. Priority Matrix

Tasks are classified by their impact on reproduction validity.

| Priority | Definition | Impact | Count |
| :--- | :--- | :--- | :--- |
| **P0** | **Logic/Architecture Critical** | Fundamental deviation from paper (e.g., wrong loss, wrong model, threat model violation). invalidates results. | 12 |
| **P1** | **Hyperparameter/Config** | Numerical deviation (LR, Batch Size, Epochs). Affects convergence/performance but not validity. | 15 |
| **P2** | **Code Health/Verify** | Structural refactoring, validation checks, or enforcing implicit constraints. | 8 |

### High-Impact P0 Tasks (Immediate Action Required)
*   **DFME**: Enforce DFME-specific Generator (Tanh, 3-layer) & Fix Gradient Scaling.
*   **ES Attack**: Fix OPT-SYN loss (KL $\to$ CrossEntropy) & Enforce ACGAN.
*   **GAME**: Enforce ACGAN w/ Dropout & Fix ACS Deviation logic.
*   **CloudLeak**: Fix FeatureFool margin calculation & Enforce perturbation constraint.
*   **SwiftThief**: Fix $\eta_{ij}$ weight formula & Switching condition.
*   **MAZE**: Enforce Generator Depth (3-layer) & Remove noise padding.

---

## 3. Implementation Roadmap

The remediation should follow this sequence to maximize stability:

1.  **Phase 1: Architecture Core (P0)**
    *   Fix shared Generator architectures (DFME, GAME, ES Attack, MAZE).
    *   Ensure strict Threat Model compliance (No gradient access).
2.  **Phase 2: Algorithmic Corrections (P0)**
    *   Correct Loss functions and Selection metrics (CloudLeak, SwiftThief, InverseNet).
3.  **Phase 3: Hyperparameter Alignment (P1)**
    *   Update LRs, Batch Sizes, Epoch counts across all attacks.
4.  **Phase 4: Verification (P2)**
    *   Run assertion suites and compliance checks.

---

## 4. Technical Specifications

### 4.1. ActiveThief
**Status:** PENDING REFACTORING
*   **[P1] Replace SGD with Adam**: Change optimizer in `train_substitute`.
    *   *Spec*: `Adam(lr=0.001, weight_decay=l2_reg)`
    *   *Loc*: `mebench/attackers/activethief.py:train_substitute`
*   **[P1] Implement Dropout**: Add dropout layer to substitute.
    *   *Spec*: `p=0.1` (Generic), `p=0.2` (CIFAR-10).
    *   *Loc*: `mebench/models/substitute_factory.py`

### 4.2. Blackbox Dissector
**Status:** PENDING REFACTORING
*   **[P1] Apply LR Schedule**: Implement specific decay schedule.
    *   *Spec*: `lr = 0.02 * (B/128)`, decay `0.1` at epochs 60, 120. `max_epochs=200`.
    *   *Loc*: `mebench/attackers/blackbox_dissector.py:train_substitute`
*   **[P1] Dataset-Specific Weight Decay**:
    *   *Spec*: `5e-4` (CIFAR-10/SVHN), `0` (Others).
    *   *Loc*: `mebench/attackers/blackbox_dissector.py:train_substitute`

### 4.3. Blackbox Ripper
**Status:** PENDING REFACTORING
*   **[P0] Align GAN Backbone**: Use ProGAN or SNGAN.
    *   *Spec*: Must use Spectral Normalization or Progressive Growing.
    *   *Loc*: `mebench/attackers/blackbox_ripper.py:_init_models`
*   **[P1] Confirm Optimizer**: Verify Adam settings.
    *   *Spec*: Adam, Constant LR (Verify value against paper).
    *   *Loc*: `mebench/attackers/blackbox_ripper.py:_train_substitute`

### 4.4. CloudLeak
**Status:** PENDING REFACTORING
*   **[P0] Use Per-Class Margin $M$**:
    *   *Spec*: Pass dynamic `margin_m` from `_get_margin_m(class_id)` to `generate_batch`.
    *   *Loc*: `mebench/attackers/cloudleak.py:FeatureFool.generate_batch`
*   **[P0] Enforce Perturbation Norm**:
    *   *Spec*: Project delta to satisfy $\|x'-x\|_2 < \eta$.
    *   *Loc*: `mebench/attackers/cloudleak.py:FeatureFool.generate_batch`

### 4.5. Copycat CNN
**Status:** PENDING REFACTORING
*   **[P1] Enforce NPDD Transfer Set**:
    *   *Spec*: Raise error if `attack.dataset` is not a Non-Problem-Domain dataset.
    *   *Loc*: `mebench/attackers/copycatcnn.py:_get_pool_dataset_config`
*   **[P2] Hard-Label Distillation**:
    *   *Spec*: Force `labels = argmax(outputs)` if soft probabilities provided.
    *   *Loc*: `mebench/attackers/copycatcnn.py:_handle_oracle_output`

### 4.6. DFME (Data-Free Model Extraction)
**Status:** PENDING REFACTORING
*   **[P0] Enforce DFME Generator**:
    *   *Spec*: 3x Conv + Linear Upsample + BN + ReLU + Tanh. Raise if `generator_type != "dfme"`.
    *   *Loc*: `mebench/attackers/dfme.py:_create_generator`
*   **[P0] Fix Gradient Scaling**:
    *   *Spec*: Remove dimension `d` multiplication in ZO gradient estimate.
    *   *Loc*: `mebench/attackers/dfme.py:run`
*   **[P2] Centralize Logit Recovery**:
    *   *Spec*: Use `_to_victim_logits()` helper instead of inline code.
    *   *Loc*: `mebench/attackers/dfme.py:_handle_oracle_output`

### 4.7. DFMS (Data-Free Model Stealing)
**Status:** PENDING REFACTORING
*   **[P1] Align Initial Query Count**:
    *   *Spec*: `init_nc=50000` for CIFAR-10 (default is 1000).
    *   *Loc*: `mebench/attackers/dfms.py:__init__`
*   **[P0] Verify Generator Depth**:
    *   *Spec*: Ensure 5-layer upsampling for 32x32 output.
    *   *Loc*: `mebench/attackers/dfms.py:_init_models`

### 4.8. ES Attack
**Status:** PENDING REFACTORING
*   **[P0] Use Cross-Entropy in OPT-SYN**:
    *   *Spec*: Replace KL Divergence with `-(y_target * log_softmax(logits)).sum()`.
    *   *Loc*: `mebench/attackers/es_attack.py:_optimize_syn_batch`
*   **[P0] Enforce ACGAN Generator**:
    *   *Spec*: Use Class-Conditional Generator (ACGAN).
    *   *Loc*: `mebench/attackers/es_attack.py:_init_models`

### 4.9. GAME
**Status:** PENDING REFACTORING
*   **[P0] Enforce ACGAN & Dropout**:
    *   *Spec*: ACGAN Generator + Discriminator with `dropout=0.25`.
    *   *Loc*: `mebench/attackers/game.py:_init_models`
*   **[P0] Implement Exact ACS Deviation**:
    *   *Spec*: Compute $KL(N_S(G(z)) || N_V^*)$ using fresh victim queries, not cached averages.
    *   *Loc*: `mebench/attackers/game.py:_compute_class_distribution`

### 4.10. InverseNet
**Status:** PENDING REFACTORING
*   **[P1] Enforce Top-1 Truncation**:
    *   *Spec*: Default `truncation_k = 1`.
    *   *Loc*: `mebench/attackers/inversenet.py:_truncate_logits`
*   **[P2] Confirm HCSS Direction**:
    *   *Spec*: Sort descending by distance $D$.
    *   *Loc*: `mebench/attackers/inversenet.py:_hcss_select`

### 4.11. Knockoff Nets
**Status:** PENDING REFACTORING
*   **[P0] Reward Rescaling**:
    *   *Spec*: Normalize reward components to `[0, 1]` before summation.
    *   *Loc*: `mebench/attackers/knockoff_nets.py:_handle_oracle_output`
*   **[P2] Handle Unlabeled Actions**:
    *   *Spec*: Support pool without ground-truth labels (fallback grouping).
    *   *Loc*: `mebench/attackers/knockoff_nets.py:_load_pool`

### 4.12. MAZE
**Status:** PENDING REFACTORING
*   **[P0] Remove Noise Padding**:
    *   *Spec*: Do not pad small batches with noise; carry over remainder.
    *   *Loc*: `mebench/attackers/maze.py:_select_query_batch`
*   **[P0] Enforce Generator Architecture**:
    *   *Spec*: 3-layer Generator for 32x32.
    *   *Loc*: `mebench/attackers/maze.py:_create_generator`

### 4.13. Random Baseline
**Status:** PENDING REFACTORING
*   **[P2] Verify Uniform Sampling**:
    *   *Spec*: Ensure strict uniform sampling without class bias.
    *   *Loc*: `mebench/attackers/random_baseline.py:_select_query_batch`

### 4.14. SwiftThief
**Status:** PENDING REFACTORING
*   **[P0] Fix $\eta_{ij}$ Weight Formula**:
    *   *Spec*: Implement exact Eq. 3: $(1+H_i)(1+H_j)\cos(\theta)$.
    *   *Loc*: `mebench/attackers/swiftthief.py:SoftSupSimSiamLossV17`
*   **[P0] Align Switching Condition**:
    *   *Spec*: Use $B-|Q| \le N_R(\mu-\mu_R)$.
    *   *Loc*: `mebench/attackers/swiftthief.py:_update_sampling_mode`

---

## 5. Verification Protocol

For each fixed attack, the following verification steps must be executed:

1.  **Static Analysis**:
    *   Verify configuration values match the "Verification Assertions" in individual specs.
    *   Check for existence of required architectural components (e.g., Dropout layers).
2.  **Unit Testing**:
    *   Run `pytest tests/` to ensure no regression in contract compliance.
    *   Create specific tests for corrected logic (e.g., test that `margin_m` varies per class in CloudLeak).
3.  **Integration Run**:
    *   Execute a dry-run (1000 queries) to ensure stability.
    *   `python -m mebench run --config configs/debug/<attack>_debug.yaml`

---

## 6. Compliance Checklist

**Final Acceptance Criteria:**

- [ ] All P0 Logic/Architecture tasks completed.
- [ ] All P1 Hyperparameter tasks completed.
- [ ] No `NotImplementedError` or `TODO` left in critical paths.
- [ ] All 14 attacks pass `tests/test_contract_validation.py`.
- [ ] Reproduction Specs updated to status **VERIFIED**.
