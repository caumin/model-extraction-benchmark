# 모델 추출 벤치마크

모델 추출 공격을 통일된 실험 설정 아래에서 비교하기 위한 PyTorch 벤치마크입니다.

[English](README.md) • [계약 가이드](Model_Extraction_Benchmark_v1.0.1_Contract_and_Implementation_Guide.md)

## 개요

- 목표: 공격 간 상대 비교
- 공식 프로토콜: `Track B` 단일 체계
- 런타임 계약: `engine -> attack.run(ctx) -> checkpoint/final eval -> artifact logging`
- 예산 계약: `1 query = 1 image`

## Track B only 의미

- 모든 공격은 native benchmark runtime으로 평가됩니다.
- victim 설정, substitute 기본값, budget, seed, reporting은 벤치마크가 통일합니다.
- 예전의 체크포인트별 from-scratch 재학습 프로토콜은 더 이상 공식 계약이 아닙니다.
- 결과 저장과 집계는 `track_b` 기준으로만 해석합니다.

## 빠른 시작

```bash
pip install -e ".[dev]"

# smoke
bash scripts/launch/run_smoke.sh cuda:0

# 설정 생성
IMAGENET_ROOT=D:/imagenet python generate_configs.py

# 매트릭스 실행
IMAGENET_ROOT=D:/imagenet bash scripts/launch/run_matrix.sh

# 결과 집계
python aggregate_matrix.py
```

## 핵심 규칙

1. `1 query = 1 image`
2. victim과 attack의 `output_mode`는 반드시 일치
3. 기본 oracle 온도는 `T=1.0`
4. pool-based 공격은 공유 substitute 기본값을 사용
5. data-free 공격은 native loop를 유지하되 동일한 artifact/reporting 계약으로 평가

## 구현된 공격

현재 벤치마크에는 네 가지 계열에 걸쳐 `17`개의 공격이 포함되어 있으며, 모두 동일한 Track-B 런타임 및 리포팅 계약 아래에서 평가됩니다.

| 계열 | 개수 | 개요 |
| :--- | ---: | :--- |
| Baseline | 1 | surrogate pool 기준선 샘플링 |
| Active Learning | 6 | 적응형 샘플 선택과 질의 효율화 |
| Pool-based / Offline | 2 | 대규모 라벨 수집 후 오프라인 substitute 학습 |
| Data-Free / Generative | 8 | 공개 surrogate pool 없이 질의 합성 기반 추출 |

### Baseline

| 공격 | 논문 | 핵심 전략 |
| :--- | :--- | :--- |
| Random | - | surrogate pool에서 균등 무작위 샘플링. |

### Active Learning

| 공격 | 논문 | 핵심 전략 |
| :--- | :--- | :--- |
| ActiveThief | [Pal et al. (2020)](https://ojs.aaai.org/index.php/AAAI/article/view/5432) | Uncertainty, K-Center, DFAL 기반 pool selection. |
| Blackbox Dissector | [Wang et al. (2021)](https://arxiv.org/abs/2105.00623) | Grad-CAM 스타일 마스킹을 활용한 erase-and-score active querying. |
| CloudLeak | [Yu et al. (2020)](https://www.ndss-symposium.org/wp-content/uploads/2020/02/24178.pdf) | 공개 이미지에서 FeatureFool 계열 adversarial query 생성. |
| InverseNet | [Gong et al. (2021)](https://www.ijcai.org/proceedings/2021/336) | inversion-inspired sample recovery와 staged querying. |
| MARICH | [Karmakar et al. (2023)](https://arxiv.org/abs/2302.08466) | staged sample selection 기반 multi-round hard-label querying. |
| SwiftThief | [Lee et al. (2024)](https://www.ijcai.org/proceedings/2024/47) | SimSiam 스타일 representation learning과 active selection 결합. |

### Pool-based / Offline

| 공격 | 논문 | 핵심 전략 |
| :--- | :--- | :--- |
| KnockoffNets | [Orekondy et al. (2019)](https://arxiv.org/abs/1812.02766) | bandit-style class sampling과 offline substitute training. |
| CopycatCNN | [Correia-Silva et al. (2018)](https://arxiv.org/abs/1806.05476) | 대규모 자연 이미지 라벨링 기반 offline stealing. |

### Data-Free / Generative

| 공격 | 논문 | 핵심 전략 |
| :--- | :--- | :--- |
| DFME | [Truong et al. (2021)](https://arxiv.org/abs/2011.14779) | generator/student를 교대로 최적화하는 min-max data-free extraction. |
| DFMS | [Sanyal et al. (2022)](https://arxiv.org/abs/2204.11022) | diverse hard-label query synthesis와 staged stealing. |
| DisGUIDE | [Tan et al. (2023)](https://ojs.aaai.org/index.php/AAAI/article/view/26150) | disagreement-guided data-free extraction과 replay. |
| MAZE | [Kariyappa et al. (2021)](https://arxiv.org/abs/2005.03161) | zeroth-order gradient estimation 기반 query synthesis. |
| ES-Attack | [Yuan et al. (2022)](https://arxiv.org/abs/2009.09560) | black-box victim을 겨냥한 evolutionary query synthesis. |
| GAME | [Xie et al. (2022)](https://link.springer.com/chapter/10.1007/978-3-031-17140-6_28) | adaptive class-conditioned generation과 extraction. |
| Dual Students | [Beetham et al. (2023)](https://arxiv.org/abs/2309.10058) | 두 student의 disagreement를 활용한 data-free extraction. |
| Blackbox Ripper | [Barbalau et al. (2020)](https://arxiv.org/abs/2010.11158) | pretrained GAN prior 위 latent-space evolution. |

## 산출물

seed별 실행 디렉터리는 아래 구조를 따릅니다.

```text
runs/<run_name>/<timestamp>/seed_<seed>/
  run_config.yaml
  summary.json
  metrics.csv
  experiment.log
```

- `run_config.yaml`: 실제 실행에 사용된 설정 스냅샷
- `summary.json`: 예산 체크포인트 기준 `track_b` 핵심 metrics
- `metrics.csv`: `track=track_b` long-form 결과 행
- `experiment.log`: 진행 상황, 이벤트, 경고, 예외를 담는 사람이 읽기 쉬운 실행 로그

## 기여

새 공격이나 문서 변경을 보낼 때는 아래 기준을 맞춰 주세요.

- 구현 위치: 새 공격은 `mebench/attackers/` 아래에 추가
- 런타임 계약: `AttackRunner.run(ctx)` 인터페이스와 Track-B 평가 흐름 유지
- 검증: `python -m pytest tests/ -q` 실행 후 결과 확인

## 라이선스

MIT. 자세한 내용은 `LICENSE` 참고.
