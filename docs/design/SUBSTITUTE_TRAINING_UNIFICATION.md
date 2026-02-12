# Substitute Training Unification Design (Draft)

## 1) 목표

- 공격 간 공정 비교를 위해 substitute 학습 루프를 **단일 표준 프로토콜**로 통일한다.
- loss 함수는 공격 고유 로직으로 유지한다.
- Track A/Track B 모두 동일한 학습 파라미터를 사용 가능하도록 한다.

---

## 2) Non-goals

- 공격 고유 쿼리 생성/선택 로직의 변경
- 데이터프리 공격의 생성 루프(GAN/ZO/SSL)의 일괄 표준화

---

## 3) 현 상태 문제점

- 공격별로 학습 루프(epochs, patience, optimizer, batch size)가 분산 구현됨
- 동일한 substitute 조건을 보장하기 어렵고, 재현성/공정성 리스크 존재
- config 규약은 있으나 실제 적용 지점이 공격마다 다름

---

## 4) 설계 원칙

1. **Single Training Engine**: 학습 루프는 중앙화
2. **Loss Injection**: 공격은 loss 함수만 주입
3. **Config-Driven**: 모든 학습 파라미터는 `substitute` 섹션에서 제공
4. **Track Compatibility**: Track A와 Track B 모두 동일한 Trainer 사용 가능

추가로, 공정 비교를 위해 early stopping 기준(metric/mode)도 표준화한다.
- Metric: validation loss
- Mode: min

---

## 5) 제안 구조

### 5.1 신규 모듈

- `mebench/training/substitute_trainer.py`
  - `SubstituteTrainerConfig`
  - `SubstituteTrainer`
  - `train()` 메서드 (표준 루프)

- `mebench/training/loss_registry.py` (선택)
  - 공격별 loss 빌더 등록 (optional)

### 5.2 표준 API (개념)

```python
class SubstituteTrainerConfig:
    arch: str
    optimizer: dict
    batch_size: int
    max_epochs: int
    patience: int
    init_seed: int
    grad_clip: float | None

class TrainRequest:
    model: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader | None
    loss_fn: Callable
    output_mode: str
    device: str

class SubstituteTrainer:
    def train(self, req: TrainRequest) -> nn.Module: ...
```

### 5.3 공격 통합 방식

- 각 공격은 **loss_fn**만 제공
- 데이터셋 구성 및 쿼리 수집 로직은 공격에 유지
- 학습 호출은 `SubstituteTrainer.train(...)`로 통일

---

## 6) 구성 스키마 (config)

```yaml
substitute:
  arch: resnet18
  init_seed: 1234
  batch_size: 256
  max_epochs: 1000
  patience: 100
  grad_clip: 1.0
  optimizer:
    name: sgd
    lr: 0.01
    momentum: 0.9
    weight_decay: 5e-4
```

Track A는 기존 `trackA.steps_coeff_c`를 유지하되, 학습 루프 자체는 동일 엔진 사용.

단, Track A는 `S(B)` 고정 step을 끝까지 수행하는 것이 목적이므로
early stopping / best checkpoint selection은 사용하지 않는다.

---

## 7) 마이그레이션 계획

### Phase 1: 공통 Trainer 도입
- `SubstituteTrainer` 구현
- 기존 Track A의 `_train_track_a`를 신규 Trainer로 치환

### Phase 2: Pool-based 공격 이관
- Random, ActiveThief, Knockoff, Copycat, Blackbox Dissector
- loss 함수 주입 방식으로 교체

### Phase 3: Hybrid 공격 이관
- CloudLeak (학습 루프 통일, 쿼리 생성 유지)

### Phase 4: Track B 전용 공격 정리
- DFME/MAZE/DFMS/GAME/ES/SwiftThief는 불가/부분 적용 명시

---

## 8) 테스트 계획

- 표준 Trainer로 학습 시 동일 seed에서 결과 일치
- patience/early stop 동작 검증
- optimizer/lr 계약 위반 시 fail-fast
- 공격별 로직과 결합 시 성능 회귀 테스트

---

## 9) 리스크 및 대응

- 공격별 고유 학습 최적화 손실 → Track B에서 별도 보고
- 학습 루프 통일로 성능 하락 가능 → 통제 실험으로 기여도 분리

---

## 10) 결정사항 요약

- 학습 루프는 중앙화한다.
- loss 함수는 공격별로 유지한다.
- Track A/Hybrid는 표준 Trainer 사용을 기본으로 한다.
