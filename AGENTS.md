# AGENTS.md — Model Extraction Benchmark

This file provides essential guidelines for agents working on the Model Extraction Benchmark repository.

## Project Overview

This is a **specification-first** PyTorch project implementing a paper-grade benchmark for comparing model extraction attacks fairly. The project follows strict validation-first principles defined in `Model_Extraction_Benchmark_v1.0.1_Contract_and_Implementation_Guide.md`.

**Status**: Design/Specification phase. No implementation exists yet.

---

## Commands

### Installation & Setup
```bash
# Install in development mode (once pyproject.toml exists)
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Running Experiments
```bash
# Run a single benchmark experiment
python -m mebench run --config configs/experiment.yaml

# Run with specific device
python -m mebench run --config configs/experiment.yaml --device cuda:0
```

### Testing
```bash
# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_contract_validation.py

# Run a specific test
pytest tests/test_budget_accounting.py::test_budget_increments_by_images

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=mebench --cov-report=html
```

### Code Quality
```bash
# Lint with ruff (if configured)
ruff check mebench/

# Format with ruff
ruff format mebench/

# Type check with mypy
mypy mebench/
```

---

## Code Style Guidelines

### Import Conventions
```python
# Standard library imports first
import os
from pathlib import Path

# Third-party imports
import torch
import torch.nn as nn
import numpy as np
import yaml

# Local imports
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
```

### Type Hints (Mandatory)
All functions must include type hints. Use explicit containers from `mebench.core.types`:
```python
def propose_queries(k: int, state: BenchmarkState) -> QueryBatch:
    """Propose k queries to send to the oracle."""
    ...

def observe_output(
    query_batch: QueryBatch,
    oracle_output: OracleOutput,
    state: BenchmarkState
) -> None:
    """Observe oracle response and update attack state."""
    ...
```

### Naming Conventions
- **Functions/Methods**: `snake_case` - `propose_queries()`, `train_substitute()`
- **Classes**: `PascalCase` - `OracleWrapper`, `SubstituteModel`, `RandomAttack`
- **Constants**: `UPPER_SNAKE_CASE` - `DEFAULT_BATCH_SIZE`, `MAX_BUDGET`
- **Variables**: `snake_case` - `query_batch`, `victim_model`
- **Private methods**: `_leading_underscore()` - `_validate_config()`, `_cleanup_cache()`

### Function Size & Complexity
- Keep functions **small and explicit** (≤ 50 lines preferred)
- One responsibility per function
- Use helper functions to avoid nesting
- Avoid hidden global state; pass state/config explicitly

### Error Handling
```python
# Use specific exceptions with clear messages
def validate_output_mode(victim_mode: str, attack_mode: str) -> None:
    if victim_mode != attack_mode:
        raise ValueError(
            f"Output mode mismatch: victim={victim_mode}, attack={attack_mode}"
        )

# Fail-fast validation in __init__
class RandomAttack:
    def __init__(self, config: dict):
        self._validate_config(config)
        ...

    def _validate_config(self, config: dict) -> None:
        if "data_mode" not in config:
            raise KeyError("RandomAttack requires 'data_mode' in config")
```

### Documentation
```python
def evaluate_metrics(
    substitute: nn.Module,
    test_loader: DataLoader,
    victim: nn.Module,
    device: str
) -> dict:
    """
    Evaluate substitute model against victim.

    Args:
        substitute: The trained substitute model
        test_loader: Test dataset DataLoader
        victim: The victim model for agreement metrics
        device: torch device ('cuda:0' or 'cpu')

    Returns:
        dict with keys: acc_gt, agreement, kl_mean (if soft mode)
    """
    ...
```

---

## Architecture Principles

### Engine is Stable, Attacks are Plugins
- `core/engine.py` is the stable, non-negotiable benchmark engine
- Attacks (`attackers/*.py`) are plugins implementing a minimal interface
- Engine should NOT know about attack internals (pool selection vs generation)

### Explicit State Management
- Never use global mutable state
- Pass state objects explicitly: `propose(k, state)` instead of `propose(k)`
- Use dataclasses from `core/types.py` for all complex containers

### Centralized Validation
- All config validation in `core/validate.py`
- Fail-fast with clear error messages
- No silent defaults that violate contract

### Determinism Requirements
- Victim inference MUST use `model.eval()` and `torch.no_grad()`
- All RNG sources controlled: Python, NumPy, Torch (CPU + CUDA)
- Track A from-scratch training must be deterministic with fixed seeds

---

## Key Contract Requirements (Non-Negotiable)

### Budget Definition
- **1 query = 1 image** sent to oracle (regardless of batching)
- Budget checkpoints: `1k, 10k, 100k, 1000k`
- Budget accounting is **image-count only**, not API-call count

### Oracle Output Modes
- `soft_prob`: Raw softmax probabilities at T=1.0 (DEFAULT)
- `hard_top1`: Top-1 class label only
- Temperature T=1.0 is enforced for default oracle profile

### Two-Track Reporting
- **Track A (Primary)**: Standard From-Scratch Protocol
  - Train fresh substitute from scratch at each checkpoint
  - Fixed batch size: 128 (unless dataset profile override)
  - Training steps: `S(B) = ceil(0.2 × B)`
  - Reset optimizer + scheduler state at each checkpoint

- **Track B (Secondary)**: Attacker-Original Protocol
  - Use attacker's native loop (round-based or online)
  - Supplementary/appendix-style

### Temporary Caching Policy
- Temporary caching of query data is permitted (for Track A at large budgets)
- Cache MUST be deleted after run completion
- Cache location must be under run-scoped directory
- Run logs must record whether caching was used

---

## File Organization

### Core Modules (Stable)
- `core/engine.py` - Main benchmark orchestration
- `core/types.py` - Data structures (QueryBatch, OracleOutput)
- `core/state.py` - BenchmarkState class
- `core/validate.py` - Config validation logic
- `core/registry.py` - Attack/oracle registration

### Attack Plugins (Extensible)
- `attackers/base.py` - Base attack interface
- `attackers/random_baseline.py` - Random selection baseline
- `attackers/activethief.py` - ActiveThief implementation
- `attackers/dfme.py` - Data-free generative attack

### Output Artifacts (Per Seed)
- `run_config.yaml` - Exact config used
- `summary.json` - Key metrics at each checkpoint
- `metrics.csv` - Long-format: `seed,checkpoint_B,track,acc_gt,agreement,kl_mean,...`
- `final_substitute.ckpt` - Final checkpoint (optional)

---

## Testing Requirements

1. **Budget Accounting Test**: Verify budget increments exactly by image count
2. **Config Validation Test**: Invalid configs fail fast (DFME + surrogate, T≠1.0 default, etc.)
3. **Reproducibility Test**: Track A results deterministic with fixed seeds
4. **Victim Determinism Test**: All queries use `eval()` + `no_grad()`
5. **Cache Cleanup Test**: Temporary caches deleted after run

---

## When to Consult the Oracle

Consult Oracle (senior engineering advisor) for:
- Complex architecture decisions (e.g., data-free attack state coupling)
- Multi-system tradeoffs (e.g., caching strategy vs memory constraints)
- After 2+ failed attempts at fixing a bug
- Unfamiliar PyTorch patterns for victim/substitute handling

---

## Quick Reference

### Attack Interface
```python
class BaseAttack:
    def propose(self, k: int, state: BenchmarkState) -> QueryBatch:
        """Return k queries to send to oracle."""
        ...

    def observe(
        self,
        query_batch: QueryBatch,
        oracle_output: OracleOutput,
        state: BenchmarkState
    ) -> None:
        """Observe oracle response and update internal state."""
        ...
```

### Oracle Interface
```python
class Oracle:
    def query(self, x_batch: torch.Tensor) -> OracleOutput:
        """Query victim model with batch of images."""
        # Apply normalization, run in eval/no_grad, enforce output mode
        # Decrement budget by x_batch.shape[0]
        ...
```

### Config Validation Pattern
```python
def validate_config(config: dict) -> None:
    # Check data mode compatibility
    attack = config["attack"]["name"]
    data_mode = config["dataset"]["data_mode"]
    if attack == "dfme" and data_mode != "data_free":
        raise ValueError("DFME requires data_free mode")

    # Check output mode compatibility
    victim_mode = config["victim"]["output_mode"]
    attack_mode = config["attack"]["output_mode"]
    if victim_mode != attack_mode:
        raise ValueError(f"Mode mismatch: victim={victim_mode}, attack={attack_mode}")

    # Check temperature for default oracle
    if config["victim"]["temperature"] != 1.0:
        raise ValueError("Default oracle requires T=1.0 in v1.0")
```

---

## Implementation Bootstrap (최소 필수)

### 필수 초기 파일 (최소 세트)
```
pyproject.toml              # 의존성, 패키지명, pytest/ruff 설정
mebench/
  __init__.py
  core/
    __init__.py
    types.py                 # QueryBatch, OracleOutput
    state.py                 # BenchmarkState
  attackers/
    __init__.py
    base.py                  # BaseAttack 인터페이스
    random_baseline.py       # 랜덤 기준 구현
  oracles/
    __init__.py
    oracle.py                # Oracle 인터페이스
configs/
  experiment.yaml            # 최소 실행 예시 1개
tests/
  test_budget_accounting.py  # budget 1쿼리=1이미지 검증
  test_contract_validation.py # 유효하지 않은 config fail-fast
  test_reproducibility.py    # 시드 고정 결과 동일성 검증
```

### 실행 커맨드 (최소 표준)
```bash
# 기본 실행
python -m mebench run --config configs/experiment.yaml

# 결과 디렉토리 구조
runs/<run_name>/<timestamp>/seed_<s>/
  run_config.yaml
  summary.json
  metrics.csv
  final_substitute.ckpt (선택)
```

### 재현성 규칙 (간단)
```python
# core/seed.py - 전체 시드 고정
import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

### Artifacts 스키마 (최소)

**summary.json**
```json
{
  "run_name": "string",
  "seeds": [0, 1, 2],
  "victim_id": "string",
  "attack": "string",
  "data_mode": "string",
  "output_mode": "string",
  "checkpoints": {
    "1000": {"track_a": {"acc_gt": float, "agreement": float}},
    "10000": {...}
  }
}
```

**metrics.csv**
```
seed,checkpoint_B,track,acc_gt,agreement,attack,data_mode,output_mode,victim_id,substitute_arch
0,1000,track_a,0.65,0.70,random,seed,soft_prob,cifar10_resnet18,resnet18
0,10000,track_a,0.72,0.78,...
```

---

## Attack Implementation Checklist

### Random Baseline
- **후보 풀 정의**: TBD (seed 또는 surrogate 중 선택)
- **샘플링 방식**: TBD (uniform, replacement 여부)
- **학습 루프**: TBD
- **하이퍼파라미터**: TBD

### ActiveThief (based on docs/reference/activethief_details.md)

**알고리즘 루프**
1. 초기화: Thief dataset에서 랜덤 초기 시드 S0 선택
2. 쿼리: Victim 모델 f에 S_i 입력하여 라벨 D_i 획득
3. 학습: 현재까지 수집된 모든 라벨 데이터 ∪D_t 로 대체 모델 f~ 을 **from scratch 재학습**
4. 후보 평가: 남은 Thief 데이터(Pool)에 대해 f~로 예측 수행
5. 선택: 능동 학습 전략(Entropy, K-center 등) 사용해 다음 쿼리 S_{i+1} 선택
6. 반복: 예산 소진 시까지 2~5단계 반복

**쿼리 생성/선택 방식**
- **방식**: Selection from Pool (합성 데이터 생성 아님)
- **선택 전략** (Top-k 방식):
  - **Uncertainty**: 엔트로피가 가장 높은 샘플 선택
  - **K-center**: 현재 선택된 센터들로부터 feature space 상 거리가 가장 먼 샘플 선택 (Greedy)
  - **DFAL**: DeepFool 섭동(perturbation)이 가장 작은(경계에 가까운) 샘플 선택
  - **DFAL+K-center**: DFAL로 ρ개 후보 선정 후, K-center로 k개 최종 선택

**학습 루프 구조**
- **순서**: Query(S_i) → Labeling → Train Substitute(f~) → Predict Pool → Select(S_{i+1}) 반복
- **재학습**: 매 라운드마다 대체 모델을 **from scratch** 다시 학습 (초기화)

**하이퍼파라미터 기본값**
- **Optimizer**: Adam (lr: 논문 미기재 → 보통 0.001)
- **Batch Size**: 150 (이미지), 50 (텍스트)
- **Max Epochs**: 1,000
- **Early Stopping**: Patience 100 (이미지), 20 (텍스트)
- **Initial Seed Size**: 전체 예산의 10%
- **Validation Split**: 전체 쿼리 예산의 20%를 검증용으로 사용
- **Regularization**: L2 (0.001), Dropout (0.1, CIFAR-10은 0.2)

**정보량 측정 방식 (Uncertainty Metric)**
- **방식**: 엔트로피(Entropy) 기반 불확실성 샘플링
- **수식**:
  ```
  H_n = -∑_j ỹ_{n,j} log ỹ_{n,j}
  ```
  - ỹ_{n,j}: 대체 모델 f~가 샘플 x_n에 대해 예측한 클래스 j의 확률
  - 엔트로피 H_n이 가장 높은(가장 불확실한) 샘플 선택

**후보 풀(Pool) 구성 및 갱신 규칙**
- **구성**: Thief dataset(공개 데이터) 중 아직 쿼리하지 않은 나머지 샘플들
  ```
  D_pool = X_thief \ ∪S_i
  ```
- **갱신**:
  1. 대체 모델 f~를 사용하여 풀에 있는 모든 샘플에 대해 근사 라벨(approximate labels)과 확률 벡터 계산
  2. 선택 전략에 의해 k개가 선정되면, 이를 풀에서 제거하고 쿼리된 세트(Labeled Set)로 이동

**Selection Rule (선택 규칙)**
- **Top-k 방식**: 각 전략별 점수(Score)를 계산하여 상위 k개 선택
  - **Uncertainty**: 엔트로피 상위 k개
  - **DFAL**: 섭동 크기 α_n 하위 k개 (결정 경계에 가장 가까운 샘플)
  - **K-center**: 현재 센터(Labeled set)로부터의 거리가 가장 먼 샘플을 하나씩 Greedy하게 k번 선택
  - **DFAL+K-center**: DFAL로 상위 ρ개(예: 전체 예산)를 먼저 필터링 후, K-center로 k개 선택

**라운드별 학습/Selection 순서**
- **순서**: Train → Predict Pool → Select → Query → Train (반복)
  1. (초기) Random Selection → Query → Collect Labels
  2. Train: 수집된 데이터로 모델 학습 (from scratch)
  3. Predict: 학습된 모델로 풀 데이터 예측
  4. Select: 전략(Strategy)을 사용해 다음 쿼리셋 선정
  5. Query: 선정된 샘플을 피해 모델에 쿼리하여 반복

**데이터/전처리**
- **Thief Dataset**: 레이블이 없는 공개 데이터(NNPD) 사용 (예: ImageNet downsampled, WikiText-2)
- **Victim과의 관계**: 피해 모델의 학습 데이터(Problem Domain)와 분포가 달라도 됨(Natural Non-Problem Domain)
- **전처리**: 표준 입력 크기 조정 등 (victim profile 기반으로 적용)

**모델 아키텍처**
- **Student (Substitute)**:
  - **이미지**: Conv 블록(2×Conv, 1×Pool) ×l개 반복 (기본 l=3). 각 Conv 뒤 ReLU, Batchnorm 적용
  - **텍스트**: Word2vec → CNN (Kim) 또는 RNN (GRU, 64 units)
- **Victim**: 공격자는 Victim의 아키텍처를 정확히 몰라도 되며, 다른 아키텍처(CNN vs RNN)를 사용해도 추출 가능

**Augmentation 사용 여부**
- **사용 여부**: 논문에 명시적인 언급 없음 (데이터 증강 미사용)
- **참고**: DFAL 전략 설명 시 "perturbed sample x̂_n"을 생성하지만, 이는 선택 기준(metric)을 계산하기 위함이며, 훈련 데이터로 추가하거나 쿼리하는 용도가 아님. 피해 모델에는 오직 원본 x_n만 쿼리됨

**구현 주의사항**
- **재학습 비용**: 매 반복마다 모델을 초기화하고 재학습해야 하므로 연산 비용이 높음
- **Validation Set**: 쿼리 예산의 20%를 검증용으로 할당하여 모델 선택에 사용 (F1 Score 기반)
- **탐지 회피**: 합성 데이터가 아닌 자연 데이터(NNPD)를 쿼리하므로 PRADA와 같은 분포 기반 탐지 기법을 회피할 수 있음

**의사코드 (논문 요약)**
```
Algorithm: ACTIVETHIEF Framework
Input: Thief Dataset (Unlabeled), Query Budget B, Initial Seed Size |S0|, Step Size k
Output: Substitute Model f_tilde

1. S_0 = Select_Random_Subset(Thief_Dataset, size=|S0|)
2. D_labeled = Query_Victim(f, S_0)  # 초기 쿼리

3. While (Current_Query_Count < B):
    # 대체 모델 학습 (매 라운드 초기화)
    f_tilde = Initialize_Model()
    Train(f_tilde, D_labeled) # Early stopping & F1 score selection

    # 후보군(Unlabeled Pool)에 대한 예측
    Unlabeled_Pool = Thief_Dataset - D_labeled.inputs
    Predictions = Predict(f_tilde, Unlabeled_Pool) # Softmax vector

    # 능동 학습 전략으로 다음 쿼리 선택
    S_next = Active_Learning_Strategy(Predictions, k)
             # Strategies: Random, Uncertainty, K-center, DFAL, etc.

    # 피해 모델 쿼리 및 데이터셋 업데이트
    D_new = Query_Victim(f, S_next)
    D_labeled = D_labeled + D_new
```

### DFME (based on docs/reference/dfme_details.md)

**알고리즘 루프**
Generator(G)와 Student(S)의 적대적 학습 (Min-Max Game):
1. G가 노이즈 z로부터 쿼리 x 생성 (x = G(z))
2. S는 V의 출력을 모방하도록 학습 (Disagreement 최소화)
3. G는 S와 V의 출력이 달라지도록 학습 (Disagreement 최대화)
4. V의 미분값 부재를 해결하기 위해 Zeroth-order gradient estimation 사용
5. 예산 Q가 소진될 때까지 G-step과 S-step을 교차 반복

**쿼리 생성/선택 방식**
- **방식**: 합성 쿼리 생성 (Generative)
- **정의**: x = G(z) (z는 랜덤 노이즈)
- **기준**: Student와 Victim 간의 불일치(Disagreement)를 최대화하는 x를 생성
- **제약**: 출력은 Tanh를 통해 [−1,1] 범위로 제한

**학습 루프 구조**
- **구조**: Online Iterative Training
- **순서**: Query → Train G → Query → Train S 구조 (배치 단위 반복)
- **반복 조건**: Query Budget Q가 소진될 때까지

**하이퍼파라미터 기본값**
- **Batch Size**: 256
- **Student Optimizer**: SGD, LR=0.1, Weight Decay=5e-4, Momentum: 통상값 추정
  - LR Scheduler: 10%, 30%, 50% 지점(전체 학습 진행도 기준)에서 0.3배 감쇠
- **Generator Optimizer**: Adam, LR=5e-4
  - LR Scheduler: 10%, 30%, 50% 지점에서 0.3배 감쇠
- **Steps**: n_G = 1 (Generator), n_S = 5 (Student)
- **Gradient Approx**: m=1 (random directions), ϵ=10^(-3)
- **Query Budget (Q)**: SVHN 2백만(2M), CIFAR-10 2천만(20M)

**Loss 구성 (정확한 수식, 각 항의 의미)**
- **수식**:
  ```
  L_l1(x) = ∑_{i=1}^K |v_i - s_i|
  ```
- **구성 요소**:
  - v_i: Victim 모델의 i번째 클래스에 대한 Logit (근사값)
  - s_i: Student 모델의 i번째 클래스에 대한 Logit
  - 의미: 두 모델의 Logit 값 간의 절대 차이 합(Mean Absolute Error)
- **L1 vs KL**: KL Divergence는 Student가 Victim에 수렴할수록 기울기가 소실(vanishing)되므로 L1 Loss가 수렴성과 안정성에 더 유리함
- **Logit Recovery**: Victim이 확률 p만 줄 경우, `l_i ≈ log(p_i) - mean(log(p))` 공식을 사용해 Logit을 복원하여 Loss 계산에 사용

**입력 노이즈 분포/샘플링 방식**
- **분포**: 표준 정규 분포 (Standard Normal Distribution)
- **방식**: z ~ N(0,1)

**Generator/Student/Victim 업데이트 순서와 스케줄**
- **순서**: Generator Update (n_G 회 반복) → Student Update (n_S 회 반복) → 반복
- **스케줄**: 전체 쿼리 예산 Q가 0이 될 때까지 무한 루프 (Epoch 개념이 아닌 쿼리 예산 기준)

**G-step/S-step 비율, Iteration 수**
- **비율**: n_G : n_S = 1 : 5
- **이유**: Generator 학습 시 Gradient Approximation 비용(쿼리)이 발생하므로, G를 적게 업데이트하고 S를 많이 업데이트하는 것이 쿼리 효율적임
- **Gradient Approx Iteration**: m=1 (Forward Difference 방향 수). 논문은 m=10과 m=1의 성능 차이가 크지 않음을 확인

**모델 아키텍처**
- **Student**: ResNet-18-8x (ResNet-18보다 채널이 8배 넓은 모델)
- **Generator**:
  - 3개의 Convolutional Layer
  - 각 Conv 층 사이에 Linear Up-sampling 배치
  - Batch Normalization 및 ReLU 사용 (마지막 층 제외)
  - 마지막 활성화 함수: Hyperbolic Tangent (Tanh) → 출력 범위 [−1,1]
- **Victim**: 실험에서는 ResNet-34-8x 사용

**데이터/전처리**
- **Data-Free**: 외부 데이터 없음
- **Input**: 표준 정규분포 z ~ N(0,1)
- **Logit Recovery**: Victim의 Softmax 확률 출력에서 평균을 빼서 Logit을 근사함 (Mean Correction)

**구현 주의사항**
- **KL Divergence 사용 금지**: 학습 후반부에 Gradient Vanishing 문제 발생
- **Gradient Approximation**: m=1(부정확한 미분)이어도 충분하며 쿼리 효율이 더 중요함
- **Victim Output**: 단순 확률(Probability)을 그대로 쓰지 말고 반드시 Logit으로 변환해야 함

**의사코드 (논문 Algorithm 1 요약)**
```
Algorithm 1: DFME
Input: Query budget Q, Generator iters n_G, Student iters n_S,
       Learning rate eta, random directions m, step size epsilon
Result: Trained Student S

While Q > 0 do:
    # 1. Generator Update (Disagreement Maximization)
    For i = 1 to n_G do:
        z ~ Normal(0, 1)
        x = G(z)
        # Victim(V)은 블랙박스이므로 Forward Differences로 G의 gradient 근사
        grad_G = Approximate_Gradient(V, S, x, m, epsilon)
        # Gradient Ascent (Disagreement 증가 방향)
        theta_G = theta_G + eta * grad_G
    End

    # 2. Student Update (Disagreement Minimization)
    For i = 1 to n_S do:
        z ~ Normal(0, 1)
        x = G(z)
        # V와 S의 출력(Logit) 계산
        loss = L1_Loss(V(x), S(x))
        grad_S = Calculate_Gradient(loss, theta_S)
        # Gradient Descent (Disagreement 감소 방향)
        theta_S = theta_S - eta * grad_S
    End

    Update Q (consumed queries)
End
```

---

## Current Experimental Setup (Matrix Protocol)

### 1. Design Matrix
| Set ID | Victim (Data/Arch) | Surrogate Dataset | Substitute Arch | Budget | Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SET-A1** | MNIST / LeNet | **EMNIST (balanced)** | LeNet | 10k | 0, 1, 2 |
| **SET-A2** | MNIST / LeNet | **FashionMNIST** | LeNet | 10k | 0, 1, 2 |
| **SET-B1** | CIFAR10 / ResNet18 | **SVHN** | ResNet18 | 10k | 0, 1, 2 |
| **SET-B2** | CIFAR10 / ResNet18 | **GTSRB (32x32)** | ResNet18 | 10k | 0, 1, 2 |

### 2. Global Contract Updates
- **Normalization**: Additional mean/std normalization has been **removed**. All inputs are assumed to be in **[0, 1] scale** to ensure compatibility with Data-Free attacks (DFME, MAZE, etc.).
- **Learning Rate**: Default substitute LR is fixed to **0.01** across all experiments.
- **Data Loaders**:
  - EMNIST uses the `balanced` split.
  - GTSRB is automatically resized to `32x32`.
- **Reproducibility**: Track A trains from scratch at each checkpoint using fixed seeds (init_seed + seed_offset).

### 3. Automation Tools
- `generate_configs.py`: Generates 168 YAML files in `configs/matrix/`.
- `run_matrix.sh`: Sequentially executes experiments with a skip-if-exists logic.
- `aggregate_matrix.py`: Aggregates seed results into Mean ± Std format (LaTeX/CSV/Markdown).

### 4. Known Implementation Fixes (Session 1)
- **NaN Loss**: Fixed by applying `clamp(min=1e-10)` and `log_softmax` in KLDivLoss calculations.
- **Gradient Clipping**: Added `max_norm=1.0` to Track A training to prevent divergence.
- **Multinomial Sampling**: Fixed GAME attack crash by handling zero-probability tensors with `nan_to_num` and `clamp`.
- **Indentation/Indirection**: Fixed various hardcoded LR and normalization parameters in `activethief.py`, `knockoff_nets.py`, and `evaluator.py`.

### 5. Running the Matrix
```bash
# 1. Generate configs
python generate_configs.py

# 2. Run all experiments
bash run_matrix.sh

# 3. View aggregated results
python aggregate_matrix.py
```
