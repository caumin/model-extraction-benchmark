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

## 산출물

```text
runs/<run_name>/<timestamp>/seed_<seed>/
  run_config.yaml
  summary.json
  metrics.csv
  metrics_history.csv
```

- `summary.json`: 예산 체크포인트별 `track_b` metrics
- `metrics.csv`: `track=track_b` long-form row
- `run_config.yaml`: 실제 실행 config

## 기여

- 새 공격은 `mebench/attackers/`에 추가
- `AttackRunner.run(ctx)` 계약 준수
- `python -m pytest tests/ -q` 통과

## 라이선스

MIT. 자세한 내용은 `LICENSE` 참고.
