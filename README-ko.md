# 모델 추출 벤치마크 (Model Extraction Benchmark)

<div align="center">

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

**모델 추출 공격(Model Extraction Attacks)의 공정한 비교를 위한 엄밀한 논문급 벤치마크**

[English](README.md) • [Paper](https://arxiv.org/abs/2305.14890) • [문서](Model_Extraction_Benchmark_v1.0.1_Contract_and_Implementation_Guide.md) • [에이전트 가이드](AGENTS.md)

</div>

---

## 🎯 개요

**Model Extraction Benchmark**는 모델 훔치기(Model Stealing) 연구 분야의 "평가 위기(Evaluation Crisis)"를 해결하기 위해 설계된 PyTorch 기반 프레임워크입니다. **엄격한 검증 우선(Validation-First) 계약**을 통해, 공격 간의 성능 차이가 일관성 없는 훈련 설정이나 예산 계산 방식이 아닌, 실제 알고리즘의 개선에서 비롯됨을 보장합니다.

### 왜 이 벤치마크인가요?
- **⚖️ 공정한 비교**: "눈속임" 변수 제거 (예: 서로 다른 대체 모델 구조, 훈련 스케줄, 정규화 방식 등 통일).
- **🔒 엄격한 계약 (Contract)**: **1 쿼리 = 1 이미지** 원칙을 강제합니다. 숨겨진 API 호출이나 배치 꼼수를 허용하지 않습니다.
- **🔄 이중 트랙 평가 (Two-Track Evaluation)**:
  - **Track A (표준 초기화 프로토콜)**: *쿼리의 품질*만을 격리하여 평가하기 위해 매 체크포인트마다 대체 모델을 처음부터(From-Scratch) 재학습합니다.
  - **Track B (공격자 원본 프로토콜)**: 온라인/능동 학습 등 공격 고유의 *네이티브 루프* 동적 특성을 보존하여 평가합니다.
- **🧪 재현성**: 초기화, 데이터 샘플링, 피해 모델(Victim) 추론에 대해 결정론적(Deterministic) 시드를 사용합니다.

---

## 🛡️ 지원되는 공격 (v1.0)

원 논문을 충실히 재구현한 **13가지 최신 공격 기법**을 지원합니다:

| 카테고리 | 공격 이름 | 논문 | 핵심 전략 |
| :--- | :--- | :--- | :--- |
| **베이스라인** | **Random** | - | 후보 풀(Pool)에서의 균등 무작위 샘플링. |
| **능동 학습 (Active Learning)** | **ActiveThief** | [Pal et al. (2020)](https://arxiv.org/abs/2002.05646) | 불확실성(Uncertainty), K-Center, DFAL 샘플링 전략. |
| | **Blackbox Dissector** | [Wang et al. (2021)](https://arxiv.org/abs/2105.13733) | Grad-CAM 기반 중요 영역 삭제 및 샘플 선택. |
| | **CloudLeak** | [Yu et al. (2020)](https://ndss-symposium.org/ndss-paper/cloudleak-large-scale-deep-learning-models-stealing-publicly-available-images/) | 적대적 예제(Adversarial Example) 생성 (FeatureFool). |
| | **InverseNet** | [He et al. (2021)](https://arxiv.org/abs/2104.04324) | 역변환(Inversion) 기반 샘플 복원. |
| | **SwiftThief** | [Miao et al. (2024)](https://arxiv.org/abs/2405.15286) | 대조 학습(Contrastive Learning, SimSiam) + 능동 선택. |
| **데이터 프리 / 생성형 (Generative)** | **DFME** | [Truong et al. (2021)](https://arxiv.org/abs/2010.12758) | GAN 기반 쿼리 합성 (Min-Max 게임). |
| | **DFMS** | [Sanyal et al. (2022)](https://arxiv.org/abs/2205.12760) | 엔트로피 최대화를 통한 다양한 쿼리 합성. |
| | **MAZE** | [Kariyappa et al. (2021)](https://arxiv.org/abs/2005.03161) | Zeroth-order 기울기 추정 (Gradient Estimation). |
| | **ES-Attack** | [Zhang et al. (2022)](https://arxiv.org/abs/2209.11797) | 진화 전략(Evolutionary Strategy)을 통한 쿼리 합성. |
| | **GAME** | [Zhang et al. (2023)](https://arxiv.org/abs/2301.12759) | 적응형 카테고리 선택 및 GAN 학습. |
| **하이브리드 / 기타** | **KnockoffNets** | [Orekondy et al. (2019)](https://arxiv.org/abs/1812.02766) | 강화학습(Bandit)을 이용한 클래스 선택. |
| | **CopycatCNN** | [Correia-Silva et al. (2018)](https://arxiv.org/abs/1806.05476) | 자연 이미지의 대규모 오프라인 증강(Augmentation). |
| | **Blackbox Ripper** | [Barbalau et al. (2020)](https://arxiv.org/abs/2003.04884) | 사전 학습된 GAN의 잠재 공간(Latent Space) 진화. |

---

## ⚡ 설치 방법

```bash
git clone https://github.com/caumin/model-extraction-benchmark.git
cd model-extraction-benchmark

# 개발 모드로 설치
pip install -e .

# 개발 의존성 설치 (테스트, 린팅 등)
pip install -e ".[dev]"
```

---

## 🚀 사용법

### 1. 단일 실험 실행
특정 공격 설정을 실행합니다. 결과물(지표, 로그)은 `runs/` 디렉토리에 저장됩니다.

```bash
python -m mebench run --config configs/cifar10_resnet18_soft_dfme_10k_seed0.yaml --device cuda:0
```

### 2. 전체 벤치마크 매트릭스 실행
v1.0 벤치마크의 전체 실험(매트릭스 프로토콜)을 재현합니다.

```bash
# 1. 모든 설정 파일 생성
python generate_configs.py

# 2. 실험 실행 (순차 실행 또는 쉘 스크립트로 병렬화 가능)
bash run_matrix.sh

# 3. 결과 집계 (CSV/LaTeX 포맷)
python aggregate_matrix.py
```

---

## 📂 디렉토리 구조

```
model-extraction-benchmark/
├── mebench/                 # 핵심 패키지
│   ├── attackers/           # 공격 구현체 (플러그인)
│   ├── core/                # 엔진, 상태 관리, 검증 로직 (안정화됨)
│   ├── models/              # 대체 모델 아키텍처, GAN
│   ├── oracles/             # 피해 모델(Victim) 래퍼
│   └── eval/                # 메트릭 및 평가 로직
├── configs/                 # YAML 설정 파일
│   ├── matrix/              # 생성된 전체 벤치마크 설정
│   └── debug/               # 테스트용 최소 설정
├── runs/                    # 실험 결과물 (메트릭, 로그)
├── data/                    # 데이터셋 (CIFAR, MNIST 등)
├── papers/                  # 참고 논문 (PDF)
└── tests/                   # 계약 검증 테스트 코드
```

---

## 📜 계약 및 방법론 (Contract & Methodology)

벤치마크의 철학, 정밀한 예산 정의, "Track A vs Track B" 프로토콜에 대한 자세한 내용은 **[계약 및 구현 가이드](Model_Extraction_Benchmark_v1.0.1_Contract_and_Implementation_Guide.md)**를 참조하세요.

### 핵심 규칙
1.  **예산 (Budget)**: `1 쿼리` = `1 이미지`. 배치 단위 쿼리는 `batch_size`만큼 예산이 차감됩니다.
2.  **오라클 (Oracle)**: 기본 `soft_prob` 모드는 온도 `T=1.0`을 사용합니다. `hard_top1`은 라벨을 반환합니다.
3.  **결정론 (Determinism)**: 피해 모델은 항상 `eval()`/`no_grad()` 상태로 실행됩니다. Track A의 시드는 고정됩니다.

---

## 🤝 기여하기 (Contributing)

새로운 공격 기법의 추가를 환영합니다! 구현 시 **[공격 인터페이스 가이드(AGENTS.md)](AGENTS.md#attack-interface)**를 따르고 계약 테스트를 통과해야 합니다.

1.  `mebench/attackers/`에 `BaseAttack`을 상속받아 구현합니다.
2.  `configs/debug/`에 테스트용 설정을 추가합니다.
3.  `pytest tests/test_contract_validation.py`를 실행하여 검증합니다.

---

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요. (참고: 일부 공격 구현체는 원본 오픈소스 레포지토리를 기반으로 합니다).
