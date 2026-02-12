# Benchmark Protocol (v1.3, Unified)

본 문서는 현재 벤치마크의 단일 운영 원칙을 정리한다.

핵심 목표는 아래 3가지다.

1. **총 쿼리수 대비 성능 비교 가능성**
2. **논문 제안 방식의 본질 보존**
3. **특정 공격에 유리하지 않은 공정한 실험 설정**

이 문서는 우선 **pool-based attacks**를 대상으로 한다.

---

## 1) 운영 원칙

- 비교 지표는 `query budget -> acc/agreement` 곡선으로 본다.
- 쿼리 회계는 절대 원칙으로 유지한다.
  - **1 query = 1 image**
  - seed 쿼리, validation 쿼리, selection 보조 쿼리 모두 budget에 포함
- 공격 간 비교는 가능한 공통 조건을 최대한 통일하되,
  논문 방법론을 붕괴시키는 통일은 금지한다.

---

## 2) 통일/비통일 기준

### 2.1 통일해야 할 것 (강제)

- Victim 모델/체크포인트
- 평가셋 및 metric 계산 코드
- Query budget, checkpoint 지점, seed set
- Thief pool 데이터 소스/전처리/샘플링 인덱스
- 단순 substitute 학습 하이퍼파라미터 (가능한 범위)
  - optimizer
  - lr
  - scheduler
  - batch_size
  - max_epochs
  - patience
  - early stopping metric

### 2.2 통일하면 안 되는 것 (보존)

- 공격 고유 정책/메커니즘 하이퍼파라미터
  - `policy_lr`, `inversion_lr`, `generator/discriminator lr`
  - bandit/policy update 관련 하이퍼
  - contrastive/KD 전용 하이퍼
  - data-free 전용 `n_g/n_s`, `m`, `epsilon` 등
- 공격 고유 루프 구조
  - online/offline, inversion loop, SSL/KD loop, generator loop

요약하면,

- **통일 대상**: "일반적인 substitute supervised training"
- **비통일 대상**: "공격 방법론의 핵심 동역학"

---

## 3) 공통 실험 프로토콜 (Pool-based)

### 3.1 Budget 분해

- 전체 budget: `B`
- 기본 분해:
  - `B_seed = 0.1B`
  - `B_val = 0.2B` (고정 validation set)
  - `B_train = B - B_seed - B_val = 0.7B`

### 3.2 반복/step 규칙

- 기본 iteration 수: `N = 10`
- `step_size`를 명시하지 않으면:
  - `k = ceil(B_train / N)`
- `step_size`를 명시하면 그 값을 사용하되,
  실험 보고서에 명시한다.

### 3.3 데이터 정책

- 동일 surrogate root 사용
- 동일 resize/transform 사용
- 동일 subset index 사용 (`subset_seed` 또는 별도 index manifest)
- train/val 분할 방식도 run 단위로 고정

### 3.4 substitute 학습 통일 정책

공격 내부에서 supervised substitute를 학습하는 구간은 아래 기본값을 따른다.

- Optimizer: SGD
- LR: 0.01
- Momentum: 0.9
- Weight Decay: 5e-4
- Scheduler: MultiStep (0.5, 0.75)
- Batch Size: 공통값
- Max Epochs / Patience: 공통값

단, 공격의 핵심 메커니즘에 결합된 내부 루프는 예외 처리한다 (Section 4).

### 3.5 평가/보고

- 주요 지표:
  - `acc_gt`
  - `agreement`
- 예산 체크포인트별 long-format CSV 필수
- 아래 항목은 표의 비고로 명시:
  - hard/soft output mode
  - dataset label 사용 여부
  - 공격 고유 하이퍼 유지 항목

---

## 4) 예외 정책 (Method-preserving)

통일 원칙을 적용하되, 아래는 예외로 허용한다.

허용 조건:

1. 해당 하이퍼/루프가 논문 기여 핵심에 직접 해당
2. 통일 시 공격 동작이 붕괴하거나 방법론 의미가 사라짐
3. 예외 항목을 실험표 비고에 명확히 공개

예:

- KnockoffNets: `policy_lr`, bandit update
- InverseNet: inversion optimizer/lr/loop
- SwiftThief: contrastive/KD 루프 및 관련 하이퍼

---

## 5) 공격별 적용 매트릭스 (Pool-based)

| Attack | Substitute Supervised Train 통일 | 고유 메커니즘 보존 |
|---|---:|---:|
| random_baseline | Yes | N/A |
| activethief | Yes | Yes (selection strategy) |
| knockoff_nets | Yes | Yes (policy/bandit) |
| cloudleak | Yes | Yes (query synthesis details) |
| copycatcnn | Yes | Yes (data query pattern) |
| inversenet | Yes (substitute stage) | Yes (inversion stage) |
| blackbox_dissector | Yes | Yes (selection/erasing logic) |
| swiftthief | Partial (가능 범위) | Yes (CL/KD core loop) |

---

## 6) 공정성 체크리스트 (Reviewer-facing)

제출 전 아래를 모두 만족해야 한다.

- [ ] 동일 victim, 동일 test set 사용
- [ ] 모든 query가 budget에 계상됨
- [ ] thief pool source/transform/subset 인덱스 고정
- [ ] 통일 hyperparameter 표 제공
- [ ] 예외 항목(비통일)과 이유 명시
- [ ] hard/soft, label usage 비고 명시
- [ ] seed별 결과 + mean/std 보고

---

## 7) 재현성 산출물

각 run은 아래를 남긴다.

- `run_config.yaml` (최종 실험 설정)
- `metrics.csv` (checkpoint별 결과)
- `summary.json`
- 선택적으로
  - validation query count
  - seed/val/train budget breakdown

---

## 8) 변경 이력

- v1.3: 단일 프로토콜로 정리. 통일/비통일 경계와 예외 정책 명시.
- v1.2: Track 분기 규칙 중심 초안.
