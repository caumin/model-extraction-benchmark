# Project Compendium — Model Extraction Benchmark

**Last updated:** 2026-06-09
**Scope:** 8주간 (2026-05-12 → 2026-06-04) 세션 7개 통합. 추후 작업 시 이 문서만 보면 됨. 자세한 분석/디버깅 과정은 `docs/session_notes/`의 개별 문서 참조.

---

## 0. Project at a Glance

3 sets × multiple pool-based + data-free attacks의 **paper-fair** model extraction 벤치마크.

| Set | Victim domain | Input | Substitute arch | Default optimizer |
|-----|---------------|------:|-----------------|-------------------|
| **SET-A1** | MNIST classification | 28×28×1 | LeNet | SGD lr=0.05 (paper-tested) |
| **SET-B1** | CIFAR-10 classification | 32×32×3 | ResNet-18 (width_mult=1, ~11M) | AdamW lr=1e-3 + cosine |
| **SET-C1** | SewerML binary defect detection | 224×224×3 | xie2019 (3 conv + 3 FC, ~9M, dropout=0.6) | SGD lr=0.05 |

**Attacks covered** (전체 n=3 seed):
- Pool-based: Random (hard/soft), KnockoffNets, ActiveThief (Uncertainty/DFAL/DFAL+kC), CloudLeak, MARICH, BlackBoxDissector, SwiftThief
- Data-free: DFME, DFMS, MAZE, ES_Attack, DisGuide, **DualStudents (DS)**

> **Correction (2026-06-09):** Earlier compendium versions misclassified DS as pool-based. DS (Beetham et al., ICLR 2023 *Dual Student Networks for Data-Free Model Stealing*) uses a DFMEGenerator (`mebench/attackers/ds.py:179`) and runs at budget=20M like other data-free attacks. The Q3 decision (data-free = SGD only) applies to DS — it is excluded from phase 3–5 4-col rerun.

**Held / Deferred (paper 외 범위):** INVERSENET, BLACKBOX_RIPPER, GAME, COPYCAT-CNN.

**Hardware:** H200 143 GB GPU + 2 TB system RAM + 128 CPU cores.

**autoyield는 자원 가정에서 제외:** `gpu_autoyield/background_train.py`는 평소 49 GB VRAM + 1 CPU core를 점유하지만 **다른 프로세스가 자원을 요구하면 자동 종료 후 양보**하는 background workload. 따라서 mebench는 GPU 143 GB / CPU 128 cores / RAM 2 TB 전부를 가용 자원으로 가정하고 계획하면 됨 (실제로 mebench가 자원을 더 요구하면 autoyield가 알아서 빠짐).

**진짜 contention 주범 — multi-process oversubscription:**
- GPU: 1 GPU 위 N processes → SM context-switch 비용 (5/24 진단 시 10 processes에서 45x slowdown)
- CPU: mebench 1 job 당 num_workers (pool=8 + train=4 + val=4) + persistent_workers → **R-state ~25 cores/job**. 5 jobs × 25 = 125 cores ≈ 128 cores 시스템 포화 → 단순 victim verify에도 13분 (단독 1분)

→ SET-C에서 이론 한도는 `floor(128 / 25) = 5`이지만 GPU SM 분할 + I/O wait + load avg 안전 마진 위해 **MAX_CONCURRENT=3** (`scripts/run_setc_aug.py`).

---

## 1. Timeline (milestone 한 줄)

| 날짜 | Milestone | Session note |
|------|-----------|--------------|
| 05-12 | SubstituteTrainer scheduler bug 발견 + 수정 (warmup_steps→warmup_epochs), Random one-shot 재설계, SwiftThief fairness 가설 정립 | `2026-05-12_scheduler_fix_and_random_ablation.md` |
| 05-13/14 | SET-B1 random_soft 17%p chaotic-SGD 진단 → AdamW lr=1e-3 cosine recipe, SubstituteTrainer 기반 attack 전체 재실행 | `2026-05-14_adamw_recipe_and_rerun.md` |
| 05-14 (저녁) | 재실행 launch, SET-C planning (n=3 / 20k budget / soft label 우선) | `2026-05-14_execution_and_setc_planning.md` |
| 05-14 (밤) | KnockoffNets RL repatch — action-space 축소, phase (a) cold-start, reward Δ, pool fast-path | `2026-05-14_knockoffnets_repatch_analysis.md` |
| 05-19 | SET-C 5종 data-free 공통 collapse (binary OOD), DFMS는 quasi-data-free (50K real ImageNet) 분리, `analyze_results.py` 학회 style 1차 개편 | `2026-05-19_setc_retry_and_analysis_insights.md` |
| 05-27 | SET-A/B aug n=3 완성, SET-C baseline fills 7건 (DisGuide replay buffer 600 GB CPU RAM 진단), MAZE 84% eval bottleneck fix (`eval_interval_queries=0`), paper_tables 학회 final style | `2026-05-27_aug_setc_fills_disguide_paper_tables.md` |
| 06-04 | DisGuide n=3 완성 (단독 sequential 101h), SET-C random sweep (SGD lr=0.05이 이미 최적), SET-C aug 33 runs launch | `2026-06-04_setc_disguide_n3_sweep_aug_start.md` |
| 06-07 | SET-C aug 33 runs 완료 (2.5일 wall), 평균 Δ −3 %p (§2.7) — SET-B와 정반대 (domain-dependent aug). MAX_CONCURRENT=3 강제 + `active = running + external` cap fix | `2026-06-08_setc_aug_done_arch_unification_plan.md` |
| 06-08 | SET-B substitute resnet18→resnet34 변경 + LR sweep R1 (12 runs): **AdamW lr=1e-3 best (58.5 ± 4.7 %)**, **SGD ~44% (학습 부족)** | 동 노트 §5 |
| 06-08/09 | LR sweep R2 추가 (15 runs, 5 lr × 3 seeds): **AdamW lr=5e-4 mean best (61.1 ± 6.9)**, SGD 모든 lr 44-47% (lr만으로 회복 불가). 사용자 결정: **AdamW lr=5e-4 + SGD lr=0.02 (mean-best) 채택, SGD column 유지하고 negative finding으로 paper 보고**. 360 phase configs 생성 + LR patch 적용. Phase 3 (SET-B 180 runs) launch | 동 노트 §5.0.2, [PHASE3_5_SCHEDULE](experiments/PHASE3_5_SCHEDULE.md) |

---

## 2. Critical findings (paper-level)

### 2.1 SubstituteTrainer scheduler bug
**`mebench/training/substitute_trainer.py`** — 학습률 스케줄러가 warmup을 **step** 단위로 받았는데 attack은 **epoch** 단위로 전달 → epoch=10일 때 warmup이 10 steps 만에 끝나서 lr이 즉시 base lr 도달, 이후 cosine decay 정상이지만 학습 dynamics 왜곡.

수정 후 일부 attack의 SET-A/B 결과가 5–10%p 이동. **pre-fix 결과는 `runs_legacy_preschedfix/` 아카이브**, 절대 비교 금지 (자동 기록: [[scheduler-bug-history]] memory).

### 2.2 SET-B1 random_soft 17%p chaotic SGD → AdamW
SGD lr=0.1 (paper default) 3 seeds 편차: **8.5%p std**, mean 0.71 ± 0.085. Bifurcation: 2 seeds는 학습 정상, 1 seed는 chunking 경계에서 gradient 폭발 후 부분 collapse.

**Recipe:** AdamW lr=1e-3, betas=(0.9, 0.999), cosine schedule, warmup=1 epoch. n=3 std 0.6%p로 축소, mean +6%p 상승.

→ **SET-B1 전체 AdamW 재실행**. 모든 pool-based attack에 적용. SwiftThief는 CL stage SGD 유지 (논문 prescription).

### 2.3 SET-C1 data-free 공통 collapse (binary OOD)
DFME / DFMS / MAZE / ES_Attack / DisGuide 5종 모두 SewerML binary domain에서 **roc_auc ≈ 0.500, base-rate constant predictor**로 수렴.

세 가지 기여 인자:
1. Single-logit BCE saturation (1-class output 특성)
2. Generator OOD (random/learned noise → binary class boundary와 무관)
3. Ensemble disagreement signal (DisGuide) → OOD에서 무력화

**DFMS exception:** 50K real ImageNet seed pool 사용 → 진정한 data-free 아님. paper에서는 별도 표기 ("quasi-data-free").

**paper message:** binary OOD high-resolution 도메인에서 data-free attack의 구조적 한계. paper 본문 caveat 1 paragraph 명시.

### 2.4 SwiftThief augmentation fairness 격차
SwiftThief는 공식 코드의 **CL stage**에서 RandomResizedCrop + HFlip + ColorJitter + Grayscale 강 augmentation을 사용 (KD stage는 raw). 다른 pool-based attack은 raw image만.

**조치:** 동일 aug pipeline을 다른 pool-based attack에 적용하여 **+Aug column**을 paper table에 추가. SwiftThief 자체는 재실험 안 함 (해당 attack의 baseline column이 이미 augmented). 즉, SwiftThief 효과 ÷ aug 효과 분리.

**SET-A는 strong/soft 두 variant** (MNIST에서 hflip + 강한 colorjitter는 ill-defined → soft = random_crop만). **SET-B는 strong 1종.** **SET-C는 binary-aware (saturation=0, hue=0)** — SwiftThief 공식 `num_classes<=2` 분기 일치.

### 2.5 KnockoffNets RL 재패치 — RL 효과 미미
원본 구현은 action space = full label space (10 actions on SET-A/B) → RL gradient noise 과대. 패치:
- action space = aggregator (random / least-confident / margin 3 categories)
- phase (a): 첫 N samples 균등 cold-start, RL bandit 시작 전
- reward Δ: per-sample acc 변화 vs running baseline
- pool fast-path: action≠RL일 때 즉시 sample (RL forward skip)

**결과:** Random baseline 대비 +0.5~1.5%p 그치는 marginal 효과. **paper 본문에서 "query selection vs sample efficiency" 음성 finding으로 보고**: KnockoffNets RL이 ActiveThief uncertainty selection과 동급(둘 다 Random 대비 작은 우위).

### 2.6 DualStudents SET-B1 hard label chaotic
| Set | seed0 | seed1 | seed2 | mean ± std |
|-----|------:|------:|------:|-----------:|
| SET-A1 DS_hard | 0.954 | 0.950 | 0.954 | 0.953 ± 0.003 ✓ |
| **SET-B1 DS_hard** | 0.531 | 0.547 | **0.100** | **0.393 ± 0.254** ⚠ |
| SET-B1 DS_soft | 0.880 | 0.885 | 0.901 | 0.889 ± 0.011 ✓ |

seed2가 random-수준 collapse. 가설: student_lr=0.3 + hard-label CE + 두 student random init disagreement signal seed-sensitive bifurcation. **paper에 as-measured 보고, 1문장 caveat.** 추가 seed 재실험은 `seed_variance_deferred` 메모리에 따라 보류.

### 2.7 SET-C aug ablation — SET-B와 정반대 (domain-dependent aug)

SET-C 33 aug runs 완료 (2026-06-07, n=3). 12 pool attacks 평균 **Δ ≈ −3.0 %p** (도움 3 / 동등 1 / 손해 8).

| Pattern | SET-A (MNIST) | SET-B (CIFAR-10) | **SET-C (SewerML binary)** |
|---------|---------------|------------------|---------------------------:|
| +Aug 효과 (avg over pool attacks) | strong harm (hflip ill-defined) → soft variant | **+10~20 %p 우위** | **−3 %p 손해 (8/12 attacks)** |

대표 cells:
| Attack | SGD baseline | SGD+Aug | Δ |
|--------|-----------:|--------:|--:|
| KNOCKOFF_NETS_SOFT | 68.5 ± 0.6 | 63.2 ± 3.4 | **−5.3** |
| ACTIVETHIEF_UNCERTAINTY_SOFT | 69.1 ± 0.1 | 60.2 ± 5.2 | **−8.9** |
| BLACKBOX_DISSECTOR_HARD | 65.7 ± 1.4 | 56.9 ± 4.3 | **−8.8** |
| ACTIVETHIEF_HARD | 65.2 ± 2.5 | 68.4 ± 2.0 | +3.2 (drop minority) |
| **SwiftThief_SOFT** (baseline = aug 내장) | **68.2 ± 1.8** | — | — |

**해석 가설 (paper 본문 후보):**
1. SewerML binary defect는 시각 단서가 작고 위치 특정 (defect 영역). `RandomResizedCrop scale=[0.2,1.0]` + HFlip이 단서 영역 절삭/거울 변형 → supervised label noise.
2. SwiftThief가 SET-C에서 우위인 이유 재해석: baseline 68.2 = 다른 attacks +Aug 결과와 동등. SwiftThief의 advantage는 **CL stage에서 aug를 representation learning에 한정** 사용한 데서 옴 (KD supervised loss는 raw image).
3. SET-B와 SET-C 정반대 결과 → **"augmentation 효과는 도메인 의존적"** message. multi-class natural image에서 우위, binary high-res defect에서 손해.

### 2.8 SET-B resnet34 substitute regime — SGD 불능, AdamW marginal

SET-B substitute resnet18 (~11M) → resnet34 (~21M) 통일 후 random_soft 27-run LR sweep (R1: 12, R2: 15) 결과:

| Optimizer | lr | Acc% mean ± std | vs resnet18 prior |
|-----------|---:|----------------:|-------------------:|
| **AdamW** | **5e-4** ⭐ | **61.08 ± 6.90** | resnet18: 62.5 → −1.4 %p |
| AdamW | 1e-3 | 58.46 ± 4.65 | −4 %p, but std smaller |
| AdamW | 2e-3 | 52.74 ± 2.76 | |
| AdamW | 3e-3 | 52.11 ± 2.41 | |
| **SGD** | **0.02** | **47.18 ± 4.45** ⚠ | resnet18 SGD 0.1: 71 → −24 %p |
| SGD | 0.01 | 46.94 ± 4.37 | |
| SGD | 0.05 | 44.53 ± 4.49 | |
| SGD | 0.1 | 43.86 ± 3.68 | |
| SGD | 0.2 | 44.18 ± 1.57 | most stable but lowest |

**핵심 발견:**
1. **SGD는 resnet34 (21M)에서 학습 불가** — lr ∈ [0.01, 0.02, 0.05, 0.1, 0.2] 5 점 grid 모두 mean 44-47% (random ≈ 50% wrt CIFAR-10 binary baseline). 추가 lr 탐색해도 의미 없음.
2. **AdamW lr=5e-4 mean best** — 그러나 std 6.9 %p (lr=1e-3의 4.7 %p보다 큼). 사용자 결정: lr=5e-4 채택 (mean 우선).
3. **resnet18 → resnet34 (capacity 2×)** 결과 적극적으로 떨어짐: AdamW −1.4 %p (acceptable), SGD −24 %p (catastrophic).

**Paper-level 메시지:** "Substitute capacity 증가 (resnet18 → resnet34) 시 optimizer choice가 critical해진다. SGD baseline은 큰 substitute에서 수렴 어려움, AdamW + cosine warmup이 필수."

### 2.9 SET-C SGD lr=0.05 안정 (xie2019 regime)
SET-B chaotic 패턴 검증을 위한 sweep 9 runs:

| Variant | mean ± std |
|---------|-----------:|
| **SGD lr=0.05** (baseline) | **0.6695 ± 0.0116** ⭐ |
| SGD lr=0.1 | 0.6426 ± 0.0383 |
| AdamW lr=1e-3 | 0.6641 ± 0.0133 |
| AdamW lr=3e-3 | 0.5675 ± 0.0566 |

SGD lr=0.05이 안정 + 최고. AdamW 도입 marginal. **SET-C는 SGD-only 진행 → paper table은 SGD baseline + SGD+Aug 2-column.**

원인 추정 (dropout=0.6 강 reg + skip-connection 없음 + 224×224 입력 dynamics가 SET-B와 다른 regime).

---

## 3. Engineering issues & fixes (재발 방지용)

| 이슈 | 원인 | 해결 |
|------|------|------|
| SubstituteTrainer scheduler step/epoch 혼선 | warmup_steps는 step 단위, 호출자는 epochs 전달 | `warmup_epochs` 명시 + `epochs * steps_per_epoch` 변환 |
| MAZE 19h → 3.7h (5x) | `eval_interval_queries=1024` 디폴트로 evaluator가 train wall 84% 차지 | attack config `extra={"eval_interval_queries": 0}` (end-only eval) |
| SET-C val_loader IPC bottleneck (~10x) | val_batch=32 + persistent_workers=False on 224×224 | `val_batch_size=128`, `persistent_workers=True`, `prefetch_factor=4` |
| DisGuide ~602 GB CPU RAM | `ClassicalMemory` `torch.zeros((1M, 3, 224, 224))` 사전 할당, 공식 `lin-tan/disguide`와 동일 (`/tmp/disguide/disguide/replay.py:49`) | 단독 sequential runner `scripts/run_setc_disguide_fills.py` (4 jobs × 25h = 101h) |
| Multi-process GPU contention 45x slowdown | 동일 GPU 위 10 concurrent processes의 SM context-switch (autoyield 점유는 yield 가능하므로 원인 아님) | VRAM-aware scheduler `committed_gb` 카운터, 4-5 jobs max, margin 5 GB |
| Scheduler over-launch (snapshot stale) | poll cycle 한 번에 5 jobs 동시 launch → `free` snapshot가 14s 동안 stale | `committed = sum(j.vram for j in running)` + 15s `time.sleep` 후 다음 launch |
| SET-C aug 5 jobs CPU 포화 (load 249, victim verify 13분) | 1 job당 ~25 R-state cores (num_workers 8+4+4 + persistent_workers) → 5 jobs × 25 = 125 cores → 128 코어 시스템 oversubscription | `MAX_CONCURRENT=3` (SET-C 한정), `LAUNCH_INTERVAL_S=90` (worker spawn 시간 확보) |
| Scheduler 재시작 시 외부 jobs 무시 → 중복 launch | `len(running)` 만 cap, pgrep으로 잡힌 SKIP external jobs 미카운트 | `total_active = len(running) + len(external)` cap, `committed`에도 external × 10 GB 추가 |
| CloudLeak missing `logging` import | refactor 누락 | `cloudleak.py` 상단에 `import logging` 추가 |
| SET-C aug strong 적용 시 MNIST 성능 폭락 | HFlip이 digit class label-breaking | SET-A는 **strong + soft** 두 variant; soft는 random_crop만 |
| `analyze_results.py` `_aug` suffix matching 충돌 | `_aug_soft` 가 `_aug` 매치 먼저 잡힘 | `_AUG_VARIANT_SUFFIXES = [("_aug_soft", "soft"), ("_aug", "strong")]` 순서 |

---

## 4. Methodology & scope decisions

| 결정 | 내용 |
|------|------|
| Budget | SET-A1/B1 pool=20k, data-free=20M; SET-C1 pool=20k, data-free=20M |
| Seeds | n=3 (seed=0,1,2) for all populated cells. n<3은 paper-fair 위해 채워야 함 |
| Optimizer per set | SET-A: SGD only / SET-B: SGD + AdamW (chaotic SGD 음성 finding 포함) / SET-C: SGD only |
| Augmentation columns | SET-A: Base + Strong + Soft / SET-B: Base + Strong (AdamW만) / SET-C: Base + Strong (binary-aware) |
| SwiftThief | 자체 재실험 안 함 (baseline column이 이미 aug 포함), 다른 attack에 같은 pipeline 적용해 분리 |
| Table 구조 | Per-set unified table: multi-row header (Optimizer × Base/+Aug × Acc/Fid) |
| Notation | $\mathcal{D}^{\mathrm{test}}$, Acc/Fid, primary caption (SET-A1)에 전체 정의 + 후속 set는 `Table~\ref{...}` cross-ref |
| Augmentation pipeline 정의 위치 | paper §Experimental Setup → Implementation Details (caption 길이 초과로 본문 이동) |
| Legacy 비교 금지 | `runs_legacy_*/` 디렉토리 (scheduler bug pre-fix, KnockoffNets pre-repatch 등)은 paper 비교용 아님 |

### Attack display names (final, paper용)

| 코드 키 | Display | Origin paper |
|---------|---------|--------------|
| random_baseline | Random | — |
| knockoff_nets | KnockoffNets | Orekondy 2019 |
| activethief | ActiveThief (k-center) | Pal 2020 |
| activethief_dfal | DFAL+k-Center | Ducoffe 2018 + Pal 2020 |
| activethief_uncertainty | Uncertainty | Pal 2020 |
| cloudleak | CloudLeak | Yu 2020 |
| marich | MARICH | Karmakar 2024 |
| blackbox_dissector | BlackBoxDissector | Wang 2022 |
| ds | DualStudents | Beetham 2023 |
| swiftthief | SwiftThief | (저자 후속) |
| dfme | DFME | Truong 2021 |
| dfms | DFMS-HL (quasi-data-free) | Sanyal 2022 |
| maze | MAZE | Kariyappa 2021 |
| es_attack | ES-Attack | Yuan 2022 |
| disguide | DisGuide | Tan 2023 |

---

## 5. Operational reference

### 5.1 Wall time (n=3 평균, single-process baseline)

| Attack | SET-A1 | SET-B1 | SET-C1 |
|--------|------:|------:|------:|
| Random / KnockoffNets / MARICH | <5 min | 15-30 min | 1-3 h |
| ActiveThief / DFAL / Uncertainty | 5-10 min | 30-60 min | 3-6 h |
| CloudLeak | 10-15 min | 30-60 min | 3-8 h |
| BlackBoxDissector | 5-10 min | 30-60 min | 3-6 h (+ end spike) |
| SwiftThief | 30 min | 2-4 h | 12-24 h |
| DFME / MAZE | 30 min | 4-8 h | 8-19 h (fix 후 3-5 h) |
| ES_Attack | 1 h | 6-10 h | 8-15 h |
| DisGuide | 2-4 h | 6-12 h | **25 h** (단독 sequential) |

### 5.2 Peak VRAM (single-process, SET-C1)

| Attack class | Sustained | Notes |
|--------------|----------:|-------|
| Random / KnockoffNets | 9-10 GB | val_loader 128 batch 포함 |
| MARICH | 12 GB | bandit overhead |
| ActiveThief / DFAL | 13-15 GB | k-center compute |
| Uncertainty | 14 GB | full-pool inference |
| CloudLeak | 10-15 GB | active sampling |
| BlackBoxDissector | 9 GB sustained / **70 GB transient** at attack_run_end | 학습 마지막 distillation pass |
| SwiftThief | 14-18 GB | CL + KD |
| DFME / MAZE | 30-45 GB | generator + student |
| ES_Attack | 30-40 GB | ensemble training |
| DisGuide | 35 GB GPU + **602 GB CPU RAM** | replay buffer |

### 5.3 Schedulers (script별 용도)

| Script | 용도 |
|--------|------|
| `scripts/run_setc_aug.py` | SET-C aug 33 runs (budget=90 GB, MAX=5, 6/4 launch 중) |
| `scripts/run_setc_disguide_fills.py` | DisGuide 단독 sequential (5/27 완료) |
| `scripts/run_priority_aug.py` | priority queue aug runner (SET-A/B용) |
| `scripts/run_setc_fills.py` | SET-C baseline fill 7건 runner (5/27 완료, archived) |

### 5.4 핵심 코드 path

```
mebench/training/
  substitute_trainer.py   # scheduler bug fix (warmup epoch 단위)
  augmentation.py         # build_augmentation_pipeline, SingleViewAugment, resolve_pool_norm_stats
mebench/attackers/
  random_baseline.py, activethief.py, knockoff_nets.py,
  blackbox_dissector.py, cloudleak.py, marich.py
                          # 모두 aug_spec → aug_fn → TrainRequest.preprocess_fn 패턴
  knockoff_nets.py:*      # RL action space + phase (a) + reward Δ + pool fast-path
  disguide.py:108         # _ClassicalReplayMemory 602 GB pre-alloc
analyze_results.py        # 학회-style: 멀티-row header, percent + subscript std, bolding, cross-ref
generate_configs.py       # SET-C val_batch=128, persistent_workers=True, MAZE eval_interval=0
mebench/models/
  substitute_factory.py:810-851  # xie2019 정의 (AlexNet-style)
```

---

## 6. Standing rules / gotchas

- **`runs_legacy_*/` 디렉토리는 절대 paper에 사용 금지.** scheduler bug pre-fix (`runs_legacy_preschedfix/`), KnockoffNets pre-repatch (`runs_legacy_knockoff_prepatch/`), 의심 runs (`runs_legacy_suspect_runs/`) 등.
- **`auto-memory`** 기록:
  - `[[experiment_hold_list]]` — INVERSENET, BLACKBOX_RIPPER, SET-C1 seed1/seed2 보류
  - `[[scheduler_bug_history]]` — 5/12 수정, pre-fix는 `runs_legacy_preschedfix/`
  - `[[seed_variance_deferred]]` — SET-B1 DS_hard, random_soft/knockoff chaotic은 paper 작성 시점 결정
- **SwiftThief는 재실험 안 함.** baseline column = augmented이므로 fair comparison의 reference point.
- **DisGuide는 동시 실행 금지** (602 GB × 2 = 1.2 TB → OOM). 단독 sequential 강제.
- **`autoyield`는 자동 양보**: 다른 프로세스가 자원 요구 시 background_train.py가 자동 종료. 자원 계획에서 차감 불필요 (GPU 143 GB / CPU 128 / RAM 2 TB 전부 사용 가능).
- **Phase config generation은 placeholder lr 패턴**: `scripts/gen_phase_rerun.py`는 SGD lr=0.1 / AdamW lr=1e-3 placeholder로 생성. sweep 결과로 결정된 lr은 `scripts/patch_lr_from_r2.py --apply [--sgd-lr X --adamw-lr Y]`로 일괄 patch. dry-run 먼저 사용.
- **VRAM-aware scheduler는 `committed_gb` 카운터 + `time.sleep(15)` 필수** — nvidia-smi snapshot이 14s까지 stale.
- **paper-untested SET** (특히 SET-C binary OOD)에서 collapse는 결과로 보고 (kill 금지 — 사용자 명시).

---

## 7. Outstanding items / next steps

### 7.1 진행 완료 (2026-06-07 종료)
- ✅ SET-C aug 33 runs (n=3) — 평균 Δ −3 %p (§2.7), `analyze_results.py` 갱신, paper tables 재생성.

### 7.2 다음 phase — surrogate-victim arch unification + 4-column unification (2026-06-08/09 진행 중)

**Phase 1-2 완료 (2026-06-08/09):**
- SET-B substitute resnet18 → resnet34 변경
- LR sweep R1+R2 = 27 runs, 결과: AdamW lr=5e-4 (mean best 61.1 ± 6.9), SGD lr=0.02 (mean best 47.2 — random-level, learning insufficient)
- 사용자 결정: **AdamW lr=5e-4 + SGD lr=0.02 (mean-best 둘 다)**, SGD column 유지 (negative finding for paper)
- 360 phase 3-5 configs 생성 (`scripts/gen_phase_rerun.py`), LR patch 적용 (`scripts/patch_lr_from_r2.py`)

**Phase 3 진행 중 (launch: 2026-06-09 00:02):**
- SET-B 180 runs (resnet34 sub): SGD baseline + SGD+Aug + AdamW + AdamW+Aug
- MAX_CONCURRENT=5, per-job VRAM 5 GB
- ETA: 3-5 일 (per-job wall ~6-10 min × 36 launch cycles)

**Phase 4-5 자동 chain (`scripts/chain_phases.py`):**
- Phase 3 종료 자동 감지 → Phase 4 (SET-A 90 runs, ~30분) → Phase 5 (SET-C 90 runs, ~4-5일)

**목표:** 모든 sets의 substitute arch를 **victim arch와 동일**하게 통일, 모든 sets의 컬럼을 **SGD / SGD+Aug / AdamW / AdamW+Aug** 4-column으로 통일. 기존 SET-B (resnet18 sub) 결과는 paper appendix로 archive.

**현재 victim/substitute 매칭:**
| Set | Victim arch | 현재 substitute | 통일 후 | Δ params |
|-----|-------------|-----------------|---------|---------:|
| SET-A1 | lenet_mnist | lenet_mnist | (변경 없음) | — |
| **SET-B1** | **resnet34** (w=1, ~21M) | resnet18 (w=1, ~11M) | **resnet34 (w=1, ~21M)** | **+10M (~2×)** |
| SET-C1 | xie2019 (~9M) | xie2019 | (변경 없음) | — |

**현재 컬럼 vs 통일 후 (Q1–Q5 결정 반영, 2026-06-08):**

| Set | 현재 | 통일 후 (main table) |
|-----|------|---------------------|
| SET-A1 | SGD / SGD+Aug / SGD+Aug-soft | SGD / SGD+Aug / SGD+Aug-soft / **AdamW** / **AdamW+Aug** (**5 col**) |
| SET-B1 | SGD / SGD+Aug / AdamW / AdamW+Aug (resnet18 sub) | SGD / SGD+Aug / AdamW / AdamW+Aug (**4 col**, **resnet34 sub로 전부 rerun**) |
| SET-C1 | SGD / SGD+Aug | SGD / SGD+Aug / **AdamW** / **AdamW+Aug** (**4 col**) |

**Cell-level 규칙:**
- **Pool attacks (11종):** 각 cell n=3.
- **SwiftThief:** SGD baseline + AdamW (2 cells per set), `+Aug` column에는 dash (baseline이 이미 augmented이므로 paper-fair convention).
- **Data-free attacks (5종):** SGD only로 SET-B/C에서 별도 row block (Budget=20M). AdamW/+Aug 컬럼은 dash. 사용자 결정 Q3=(a).
- **SET-B legacy (resnet18 sub) 결과:** main table에서 제외. paper **appendix 별도 표** "SET-B legacy (substitute=ResNet-18)"로 보존. `analyze_results.py`가 자동 분리 출력하도록 패치 필요.

**Phase 작업 순서 (사용자 명시):**
1. **SET-B substitute arch resnet18 → resnet34** + smoke test (peak VRAM/wall-time 측정)
2. **SET-B LR sweep** (random_soft × {SGD, AdamW} × multiple lr × n=3) — 새 arch 최적 hyperparam
3. **SET-B 4-cell full rerun** (~144 runs)
4. **SET-A AdamW + AdamW+Aug** (72 runs)
5. **SET-C AdamW + AdamW+Aug** (72 runs)

**자원 견적 (대략):**
| Phase | 기간 |
|-------|------|
| 1. SET-B arch change + smoke | 0.5 일 |
| 2. SET-B LR sweep (~9 runs) | 1 일 |
| 3. SET-B full rerun (144 pool) | 3-5 일 (resnet34 = resnet18 × ~1.7×) |
| 4. SET-A AdamW (72 runs) | 0.5-1 일 (SET-A는 매우 가벼움) |
| 5. SET-C AdamW (72 runs) | 4-5 일 (SET-C 무거움) |
| **합계** | **9-12 일** |

**결정 사항 (2026-06-08 확정):** 위 cell-level 규칙 표 참조.
- SwiftThief: SGD + AdamW 2 cells, +Aug 칸은 dash.
- SET-A: 5 col (`SGD+Aug-soft` 유지).
- Data-free: SGD only block (별도 row).
- SET-B legacy resnet18: paper appendix 별도 table로 보존 (`analyze_results.py` 분리 출력 패치 필요).
- LR sweep: 4 variants × n=3 = 12 runs (확장 없음).

### 7.3 Paper writing (실험 완료 후)
- Aug pipeline 정의 §Experimental Setup → Implementation Details
- SwiftThief fairness narrative 1 paragraph (SET-C 결과 포함, §2.7 해석)
- DisGuide collapse caveat 본문 1 paragraph
- DS_HARD SET-B1 high variance 1 문장
- **"Augmentation effect is domain-dependent"** 1 paragraph (SET-B vs SET-C 정반대 결과)

---

## 8. 개별 session note 참조 (deep dive)

| 주제 | 문서 |
|------|------|
| Scheduler bug 진단/수정 과정, SwiftThief fairness 1차 가설 | `session_notes/2026-05-12_*` |
| SET-B1 chaotic-SGD 17%p butterfly 추적, AdamW recipe 도출 | `session_notes/2026-05-14_adamw_recipe_*` |
| 재실행 launch operational details, SET-C planning | `session_notes/2026-05-14_execution_*` |
| KnockoffNets RL action/phase/reward/fast-path 패치 분석 | `session_notes/2026-05-14_knockoffnets_repatch_*` |
| SET-C data-free 5종 collapse mechanism, DFMS quasi-data-free, analyze_results 1차 개편 | `session_notes/2026-05-19_*` |
| SET-A/B aug 완성 운영, SET-C baseline fills, DisGuide 600 GB 진단, paper_tables final | `session_notes/2026-05-27_*` |
| DisGuide n=3 완성, SET-C random sweep, SET-C aug launch | `session_notes/2026-06-04_*` |
| SwiftThief augmentation fairness 분석 | `experiments/swiftthief_augmentation_fairness.md` |
| SET-C1 random_soft sweep 설계 | `experiments/setc1_random_soft_sweep.md` |
| SET-B1 optimizer 비교 | `experiments/setb1_optimizer_comparison.md` |
| Seed variance issues 정리 | `experiments/seed_variance_issues.md` |
| Metrics 정의 | `docs/METRICS.md` |
