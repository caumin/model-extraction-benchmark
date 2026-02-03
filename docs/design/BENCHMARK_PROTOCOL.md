# Benchmark Protocol Draft (v1.2)

본 문서는 **공정한 비교를 위한 분기 규칙**, **공격별 분류표**, **통제 실험 템플릿**을 정의한다.
목표는 쿼리 전략의 성능을 표준화된 조건에서 비교하면서, 구조적으로 분리가 불가능한 공격도
명확한 기준과 보고 규칙으로 포함하는 것이다.

---

## 1) 목적과 범위

- 공격 알고리즘의 **쿼리 전략**과 **학습 루프**를 분리하여 비교 가능성을 확보한다.
- 분리가 불가능한 공격은 Track B로 별도 보고하되, 공통 불변조건을 강제한다.
- loss 함수는 공격 고유 로직으로 간주하여 공격별로 유지한다.

---

## 2) 용어 정의

- **Track A (Standard From-Scratch)**: 표준 substitute 학습 루프(아키텍처/옵티마/스텝/배치/조기종료)가 고정된 비교 트랙.
- **Track B (Attacker-Original)**: 공격의 원래 프로토콜(온라인/SSL/생성/조합 루프 포함)을 허용하는 트랙.
- **Hybrid**: 쿼리 생성은 공격 고유 방식, 학습 루프는 표준화가 가능한 공격.

---

## 3) 분기 규칙 (Decision Tree)

다음 규칙을 **강제** 적용한다. 공격 개발자가 임의 선택하지 않는다.

1. 학습 루프가 표준 지도학습(CE/KL)이며, 혁신이 **쿼리 선택**에 있는가?
   - YES → **Track A**
   - NO → 2로 이동
2. 쿼리 생성이 학습 루프와 **강하게 결합**되어 분리 불가능한가?
   - YES → **Track B**
   - NO → **Hybrid** (쿼리 생성은 공격 고유, 학습 루프는 표준화)

**예외 승인 기준**
- 표준화가 논문 핵심 기여를 훼손하는 경우만 예외 허용
- 예외는 반드시 통제 실험(Section 6)을 동반

---

## 4) 공통 불변조건 (All Tracks)

- Victim 모델 및 평가셋 고정
- **Substitute 아키텍처 고정** (폭/채널 포함)
- Budget 회계: **1 query = 1 image** (배치 기반 할인 금지)
- Oracle 출력 모드 및 Temperature 고정
- 데이터 전처리 규칙 고정 (입력 스케일/정규화)

---

## 5) 보고 규칙

- **Table 1: Track A (Query Efficiency)**
  - 표준 학습 루프 하에서 쿼리 전략 비교
- **Table 2: Track B (Full System Performance)**
  - 원본 공격 프로토콜 성능 비교
- **Delta Report**
  - Track A vs Track B 차이를 별도 보고 (학습 루프 의존도 측정)
- 핵심 지표 권장
  - AUBAC (Budget-Accuracy Curve 면적)
  - Final Accuracy / Agreement @ budget checkpoints

---

## 6) 공격별 분류표 (초안)

| Attack | Category | Track | Rationale | Notes |
|---|---|---|---|---|
| Random | Pool-based | A | 쿼리 선택만 존재 | 표준 루프 고정 |
| ActiveThief | Active Learning | A | 선택 전략이 핵심 | 표준 루프 고정 |
| KnockoffNets | Pool + Bandit | A | 선택 정책이 핵심 | 학습 루프 표준화 가능 |
| CopycatCNN | Pool + Aug | A | 쿼리 데이터 조합이 핵심 | Aug는 데이터 전처리로 간주 |
| Blackbox Dissector | Pool + CAM | A | 선택/변형이 핵심 | Hard mode 유지 |
| CloudLeak | Adv Query | Hybrid | 쿼리 생성은 공격 고유 | 학습 루프는 표준화 가능 |
| InverseNet | Inversion | B | 쿼리 생성과 학습 결합 | soft-only 유지 |
| Blackbox Ripper | GAN/EA | B | 생성 루프 결합 | Generator/Student 동시 최적화 |
| DFME | Data-Free GAN | B | 쿼리 생성과 학습 결합 | L1 loss 고유 |
| DFMS | Data-Free GAN | B | 생성 루프 결합 | Generator/Student 동시 최적화 |
| MAZE | Data-Free ZO | B | 쿼리 생성과 학습 결합 | gradient-estimation 기반 |
| GAME | Data-Free GAN | B | 생성 루프 결합 | class-wise GAN |
| ES Attack | Data-Free ES | B | 생성 루프 결합 | ES 최적화 포함 |
| SwiftThief | SSL + KD | B | 학습 루프가 기여의 핵심 | CL/KD 파이프라인 고정 |

**Hybrid 적용 기준**
- 쿼리 생성은 공격 고유 방식을 유지하되, substitute 학습 루프는 표준화 가능하다고 판단되는 경우

---

## 7) 통제 실험 설계 템플릿

아래 템플릿을 **Track B/Hybrid 공격**에 필수 적용한다.

```
### Control Experiment: <Attack Name>

Goal:
- 학습 루프 vs 쿼리 전략의 성능 기여도 분리

Setup:
- Victim: <victim_id>
- Substitute Arch: <arch>
- Output Mode: <soft_prob/hard_top1>
- Budget: <B>
- Seeds: <list>

Conditions:
1) Track A (Standard Trainer + Attack Queries)
2) Track B (Native Loop + Attack Queries)
3) Native Loop + Random Queries (Loop Contribution Control)
4) Standard Trainer + Random Queries (Baseline)

Metrics:
- Accuracy@B, Agreement@B
- AUBAC (if multiple checkpoints)
- Query Efficiency (acc per 1k queries)

Expected Interpretation:
- (2) - (1): 학습 루프 기여도
- (1) - (4): 쿼리 전략 기여도
- (2) - (3): 쿼리 전략 vs 루프 상호작용
```

---

## 8) 예외 처리 및 승인 기준

- 예외는 반드시 **이유 + 통제 실험 결과**와 함께 보고한다.
- 예외 허용 여부는 아래 기준으로 결정한다.
  - 공격의 핵심 기여가 학습 루프 자체인가?
  - 표준화 시 공격의 동작 원리가 붕괴하는가?
  - 통제 실험에서 루프/쿼리 기여도가 분리되는가?

---

## 9) 변경 이력

- v1.2: Track 분기 규칙/통제 실험 템플릿 추가
