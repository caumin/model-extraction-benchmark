# 모델 추출 벤치마크 (Model Extraction Benchmark)

<div align="center">

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

**모델 추출 공격(Model Extraction Attacks)의 공정한 비교를 위한 엄밀한 논문급 벤치마크**

[English](README.md) • [문서](Model_Extraction_Benchmark_v1.0.1_Contract_and_Implementation_Guide.md)

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

### 실험 세트 (현재 기준)

| Set | Victim | Substitute | Surrogate | 입력 | Query Budget |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SET-A1** | MNIST LeNet (프로젝트 내 스크립트로 학습) | LeNet | ILSVRC2012 train (ImageNet) | `1x28x28` | `10k` (pool/hybrid), `10m` (synthetic) |
| **SET-B1** | CIFAR10 ResNet34 (DFAD 공식 체크포인트) | ResNet18 | ILSVRC2012 train (ImageNet) | `3x32x32` | `20k` (pool/hybrid), `20m` (synthetic) |

참고:
- 매트릭스 생성 기본 시드는 `0,1,2` 입니다.
- `SET-B1` victim 체크포인트는 DFAD 레포 기준입니다: `https://github.com/VainF/Data-Free-Adversarial-Distillation`.
- 매트릭스 설정은 `generate_configs.py`가 `configs/matrix/`에 생성합니다.

### 매트릭스 하이퍼파라미터 정책 (현재)

- **SET-B1 (`substitute.arch=resnet18`)**:
  - 공식/논문 근거가 충분히 비교 가능한 공격은 `generate_configs.py`에서 optimizer 계열과 기준 LR/batch를 정렬합니다.
  - 기본 정렬 경로는 **batch 512 + LR 선형 스케일**(SGD 계열 학습의 처리량 최적화 목적)입니다.
  - 예외 경로(`keep_reference_pair=True`)는 안정성 확보를 위해 공식/논문 LR-batch 쌍을 그대로 유지합니다.
  - 현재 keep-reference 공격: `dfme`, `ds`, `maze`, `swiftthief`.
- **SET-A1 (`substitute.arch=lenet_mnist`)**:
  - 이 벤치마크 설정과 일대일로 대응되는 LeNet 공식 프로파일이 일관되게 확보되지 않아,
  - 매트릭스 생성은 통합 heuristic substitute 기본값을 사용하고 공격 의미론적 파라미터만 유지합니다.
- 대상 설정에서 신뢰 가능한 일대일 근거가 부족한 공격은 heuristic 기본값을 유지합니다 (`activethief`, `marich`, `cloudleak`, `inversenet`, `random`, `dfms`).

---

## 🛡️ 지원되는 공격 (v1.0)

원 논문을 충실히 재구현한 **17가지 최신 공격 기법**을 지원합니다:

| 카테고리 | 공격 이름 | 논문 | Official Repo | 핵심 전략 |
| :--- | :--- | :--- | :--- | :--- |
| **베이스라인** | **Random** | - | - | 후보 풀(Pool)에서의 균등 무작위 샘플링. |
| **능동 학습 (Active Learning)** | **ActiveThief** | [Pal et al. (2020)](https://ojs.aaai.org/index.php/AAAI/article/view/5432) | [GitHub](https://github.com/iisc-seal/activethief) | 불확실성(Uncertainty), K-Center, DFAL 샘플링 전략. |
| | **MARICH** | [Basu et al. (2023)](https://arxiv.org/abs/2302.08466) | [GitHub](https://github.com/Debabrota-Basu/marich) | 엔트로피/손실/그래디언트 기반 다단계 능동 질의. |
| | **Blackbox Dissector** | [Wang et al. (2021)](https://arxiv.org/abs/2105.00623) | [GitHub](https://github.com/yxwang-10/Blackbox-Dissector) | Grad-CAM 기반 중요 영역 삭제 및 샘플 선택. |
| | **CloudLeak** | [Yu et al. (2020)](https://www.ndss-symposium.org/wp-content/uploads/2020/02/24178.pdf) | [GitHub](https://github.com/yunyuntsai/DNN-Model-Stealing) | 적대적 예제(Adversarial Example) 생성 (FeatureFool). |
| | **InverseNet** | [Gong et al. (2021)](https://www.ijcai.org/proceedings/2021/336) | - | 역변환(Inversion) 기반 샘플 복원. |
| | **SwiftThief** | [Lee et al. (2024)](https://www.ijcai.org/proceedings/2024/47) | [GitHub](https://github.com/ku-air/SwiftThief) | 대조 학습(Contrastive Learning, SimSiam) + 능동 선택. |
| **데이터 프리 / 생성형 (Generative)** | **DFME** | [Truong et al. (2021)](https://arxiv.org/abs/2011.14779) | [GitHub](https://github.com/cake-lab/datafree-model-extraction) | GAN 기반 쿼리 합성 (Min-Max 게임). |
| | **Dual Students (DS)** | [Beetham et al. (2023)](https://arxiv.org/abs/2309.10058) | [GitHub](https://github.com/James-Beetham/dual_students) | 두 학생 모델 간 불일치 기반 데이터 프리 추출. |
| | **DFMS** | [Sanyal et al. (2022)](https://arxiv.org/abs/2204.11022) | [GitHub](https://github.com/val-iisc/Hard-Label-Model-Stealing) | 엔트로피 최대화를 통한 다양한 쿼리 합성. |
| | **DisGUIDE** | [Rosenthal et al. (2023)](https://ojs.aaai.org/index.php/AAAI/article/view/26150) | [GitHub](https://github.com/lin-tan/disguide) | 불일치(disagreement) 기반 데이터 프리 추출 + replay/diversity 손실. |
| | **MAZE** | [Kariyappa et al. (2021)](https://arxiv.org/abs/2005.03161) | [GitHub](https://github.com/sanjaykariyappa/MAZE) | Zeroth-order 기울기 추정 (Gradient Estimation). |
| | **ES-Attack** | [Yuan et al. (2022)](https://arxiv.org/abs/2009.09560) | - | 진화 전략(Evolutionary Strategy)을 통한 쿼리 합성. |
| | **GAME** | [Xie et al. (2022)](https://link.springer.com/chapter/10.1007/978-3-031-17140-6_28) | [GitHub](https://github.com/xythink/game-attack) | 적응형 카테고리 선택 및 GAN 학습. |
| **하이브리드 / 기타** | **KnockoffNets** | [Orekondy et al. (2019)](https://arxiv.org/abs/1812.02766) | [GitHub](https://github.com/tribhuvanesh/knockoffnets) | 강화학습(Bandit)을 이용한 클래스 선택. |
| | **CopycatCNN** | [Correia-Silva et al. (2018)](https://arxiv.org/abs/1806.05476) | [GitHub](https://github.com/jeiks/Stealing_DL_Models) | 자연 이미지의 대규모 오프라인 증강(Augmentation). |
| | **Blackbox Ripper** | [Barbalau et al. (2020)](https://arxiv.org/abs/2010.11158) | [GitHub](https://github.com/antoniobarbalau/black-box-ripper) | 사전 학습된 GAN의 잠재 공간(Latent Space) 진화. |

---

공격별 구현 출처(제안 논문 + 공식 레포 이식 링크)는
`docs/reference/ATTACK_REFERENCES.md`에서 확인할 수 있습니다.

### 추가 개선 필요 사항

- `MARICH`는 현재 `mebench/attackers/marich.py`에서 논문/원본 단계형 프로토콜을 유지하고 있습니다.
- 따라서 다른 pool-based 공격과 완전히 통합된 공정 비교 프로토콜에는 아직 맞춰지지 않았습니다.
- 향후 과제: pool-based 공격 공통 프로토콜 계층(선택/학습/평가 스케줄)을 통합해 MARICH를 정렬.

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

## ✅ 빠른 시작 (공개 배포 워크플로우)

```bash
# 1) 스모크 실행 (권장 시작점)
bash scripts/launch/run_smoke.sh cuda:0

# 2) 매트릭스 설정 생성
# ImageNet surrogate 기본 경로는 C:/imagenet 입니다.
# 각자 로컬 경로에 맞게 IMAGENET_ROOT를 바꿔서 생성하세요.
IMAGENET_ROOT=C:/imagenet python generate_configs.py

# WSL 예시
IMAGENET_ROOT=/mnt/c/imagenet python generate_configs.py

# 3) (선택) 같은 경로로 전체 매트릭스 실행
IMAGENET_ROOT=C:/imagenet bash scripts/launch/run_matrix.sh

# 4) 완료된 실행 결과 집계
python aggregate_matrix.py
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
IMAGENET_ROOT=C:/imagenet python generate_configs.py

# 2. 실험 실행 (순차 실행 또는 쉘 스크립트로 병렬화 가능)
IMAGENET_ROOT=C:/imagenet bash scripts/launch/run_matrix.sh

# 3. 결과 집계 (CSV/LaTeX 포맷)
python aggregate_matrix.py
```

공식 런처 스크립트는 `scripts/launch/`에 정리되어 있습니다.
`IMAGENET_ROOT`는 사용자 로컬 ImageNet 경로에 맞게 반드시 수정해야 합니다.

### 3. 스모크 실행 (권장 시작점)

```bash
bash scripts/launch/run_smoke.sh cuda:0
```

### 4. 공개 배포 범위 안내

공개 배포본은 위의 `mebench` 프레임워크 실행 워크플로우에 한정됩니다.
`papers/`, `repro/`, `official_repo_clones/`는 내부/연구 자산으로
오픈소스 배포 범위에서 제외됩니다.

---

## 📂 디렉토리 구조

```
model-extraction-benchmark/
├── mebench/                 # 핵심 벤치마크 패키지
├── configs/
│   └── smoke/               # 추적되는 최소 설정
├── scripts/
│   └── launch/              # 공식 실행 런처
├── docs/
│   └── reference/
├── tests/                   # 계약/회귀 테스트
├── generate_configs.py
└── aggregate_matrix.py
```

## 공개 배포 범위

- 공개 배포는 `mebench` 실험 프레임워크 및 실행/검증에 필요한 자산만 포함합니다.
- `papers/`, `repro/`, `official_repo_clones/`는 배포 대상에서 제외합니다.
- `data/`, `runs/`, 로그, 체크포인트, 생성 리포트 등 로컬 런타임 산출물은 배포 대상이 아닙니다.
- 비공개 로컬 계획/설정 파일은 `.gitignore`로 배포에서 제외합니다.

---

## 📜 계약 및 방법론 (Contract & Methodology)

벤치마크의 철학, 정밀한 예산 정의, "Track A vs Track B" 프로토콜에 대한 자세한 내용은 **[계약 및 구현 가이드](Model_Extraction_Benchmark_v1.0.1_Contract_and_Implementation_Guide.md)**를 참조하세요.

### 핵심 규칙
1.  **예산 (Budget)**: `1 쿼리` = `1 이미지`. 배치 단위 쿼리는 `batch_size`만큼 예산이 차감됩니다.
2.  **오라클 (Oracle)**: 기본 `soft_prob` 모드는 온도 `T=1.0`을 사용합니다. `hard_top1`은 라벨을 반환합니다.
3.  **결정론 (Determinism)**: 피해 모델은 항상 `eval()`/`no_grad()` 상태로 실행됩니다. Track A의 시드는 고정됩니다.
4.  **BlackBox MLaaS 입력 계약**: 공격자는 Victim의 내부 정규화 정보를 알지 못한다고 가정합니다. 쿼리 이미지는 그대로 전송되며, 런타임 쿼리 경로에서는 wrapper 변환 없이 Victim 추론으로 직접 전달됩니다. Pool 기반 공격은 surrogate 표준 정규화 공간(`dataset.surrogate_normalization`, 기본 `standard`)에서 query/train을 수행하고, Data-free 공격은 `[-1,1]` 스케일로 쿼리하며 victim 쿼리 경로에서 attacker-side tanh->unit 변환을 하지 않습니다. 평가는 공정 비교를 위해 공통 정규화 test loader를 사용합니다.

---

## 🤝 기여하기 (Contributing)

새로운 공격 기법의 추가를 환영합니다! `mebench/attackers/`의 공격 인터페이스를 따르고 계약 테스트를 통과해야 합니다.

1.  `mebench/attackers/`에 `BaseAttack`을 상속받아 구현합니다.
2.  `configs/debug/`에 테스트용 설정을 추가합니다.
3.  `pytest tests/test_contract_validation.py`를 실행하여 검증합니다.

---

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요. (참고: 일부 공격 구현체는 원본 오픈소스 레포지토리를 기반으로 합니다).
