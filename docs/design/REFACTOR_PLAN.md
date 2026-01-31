# Attack IOC Refactor Plan (Track B First)

## 0) 목적과 결정사항

- propose/observe 구조를 완전히 제거한다.
- 공격이 자체 프로토콜로 실행되도록 IOC(Inversion of Control)를 채택한다.
- Track B(공격 원형 루프)만 우선 구현한다. Track A는 추후 재도입한다.
- 예산, 로깅, 체크포인트 관리는 BenchmarkContext에 집중한다.

Non-goals:
- Track A 학습/평가 복원
- legacy propose/observe 호환성 유지

---

## 1) 배경 및 문제 정의

기존 엔진은 propose/observe 루프를 강제하여 모든 공격을 동일한 스텝 모델에 끼워 넣는다.
이 구조는 다음 문제를 유발한다.

- 공격별 프로토콜(Generative, RL, multi-stage)을 왜곡한다.
- train-only 단계(질의 없는 학습)를 표현할 수 없다.
- step_size 고정과 같은 제약이 공격 로직과 충돌한다.

따라서 공격이 루프를 주도하는 IOC 구조로 전환해야 한다.

---

## 2) 신규 인터페이스

### 2.1 BenchmarkContext

```python
class BenchmarkContext:
    def query(self, x: torch.Tensor, meta: Optional[dict] = None) -> OracleOutput:
        """Oracle query + budget check + logging + checkpoint hook."""

    @property
    def budget_remaining(self) -> int: ...

    @property
    def query_count(self) -> int: ...

    def log_event(self, name: str, payload: dict) -> None: ...

    def on_checkpoint(self, query_count: int) -> None:
        """Track B에서는 no-op. Track A 재도입 시 사용."""
```

Responsibilities:
- 예산 체크 및 차감(1 query = 1 image)
- Oracle 호출 및 결과 반환
- 체크포인트 도달 시 훅 호출
- (옵션) query 기록(Track A 대비, 기본 비활성)

### 2.2 AttackRunner

```python
class AttackRunner(ABC):
    @abstractmethod
    def run(self, ctx: BenchmarkContext) -> None:
        """Attack protocol owns the loop."""
```

Train-only 단계는 ctx.query를 호출하지 않으면 된다.

---

## 3) 엔진 실행 흐름 (Track B Only)

```python
def run_experiment(config, device="cpu"):
    state = BenchmarkState(...)
    oracle = Oracle(victim, config["victim"], state)
    ctx = BenchmarkContext(state, oracle, logger, config)
    attack = create_runner(config["attack"]["name"], config, state)
    attack.run(ctx)
```

변경점:
- propose/observe 루프 삭제
- Evaluator/Track A 호출 제거
- QueryStorage는 기본 비활성(Track A 재도입 시 사용)

---

## 4) 예산 및 체크포인트 정책

- ctx.query는 남은 예산을 초과하면 에러를 발생시킨다(엄격 모드).
- train-only 단계는 예산 소비 0.
- 체크포인트는 ctx.query 내부에서 검출하고 on_checkpoint를 호출한다.
- Track B 우선 구현에서는 on_checkpoint를 no-op로 둔다.

---

## 5) 공격 마이그레이션 가이드 (Track B)

### 5.1 Pool-based (ActiveThief, Random, Copycat 등)

```python
while ctx.budget_remaining > 0:
    x = select_from_pool(k, state)
    y = ctx.query(x)
    update_internal_state(x, y)
```

### 5.2 Generative/Iterative (DFME, DFMS, MAZE, ES, GAME 등)

- 논문 루프를 그대로 유지한다.
- query 호출 지점만 ctx.query로 교체한다.
- 배치 크기는 ctx.budget_remaining으로 클램프한다.

### 5.3 SwiftThief

- seed query -> CL -> (SL loop)
- SL loop:
  - KD 1 epoch 수행(train-only)
  - sl_aug_interval마다 query/sampling + CL 재학습

이 구조는 IOC에서 자연스럽게 표현된다.

---

## 6) 변경 파일 목록

Add:
- mebench/core/context.py
- mebench/attackers/runner.py

Modify:
- mebench/core/engine.py
- mebench/attackers/*.py (전부 run(ctx)로 전환)
- mebench/attackers/__init__.py
- mebench/core/validate.py (필요 시 인터페이스 검증)

Delete:
- mebench/attackers/base.py (propose/observe 제거 후 대체)
- docs/design/REFACTOR_PLAN2.md

---

## 7) 테스트 계획 (Track B)

- BenchmarkContext 예산 정확성 테스트
- train-only 단계에서 budget 소비 0 테스트
- Runner 스모크 테스트(랜덤, DFME, SwiftThief)
- propose/observe 기반 테스트 제거 또는 재작성

---

## 8) 마이그레이션 순서

1. BenchmarkContext + AttackRunner 도입
2. engine.py 실행 경로 전환 (Track B only)
3. Random 공격 runner 전환
4. DFME runner 전환
5. SwiftThief runner 전환
6. 나머지 공격 순차 이관
7. 테스트/문서 정리

---

## 9) 후속 작업 (Track A 재도입)

- ctx.query 기록 활성화(QueryStorage 연동)
- 체크포인트에서 Evaluator 호출
- from-scratch Track A 루프 복구

---

## 10) 명시적 결정사항

- legacy propose/observe 경로는 유지하지 않는다.
- Track B만 우선 구현한다.
- ctx.query는 엄격한 예산 모드로 동작한다.
