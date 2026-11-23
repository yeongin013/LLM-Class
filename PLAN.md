# 프로젝트 계획: Prompt Injection 분류기의 추론 비용 줄이기

> 작성일 2026-11-11. 주제는 추천안의 1순위(BERT에 강의 기법 4가지 적용)와 3순위(cascade)를 결합한 것이다.

## 1. 목표와 연구 질문

사전학습 BERT로 만든 prompt injection 분류기를 기준으로, 강의에서 배운 기법을 적용해 **정확도를 거의 유지하면서 추론 시간을 줄인다.**

- **Q1.** 이 데이터에서 BERT 추론 시간의 병목은 어디인가? Attention의 O(n²)인가, 아니면 n·d²에 비례하는 투영·FFNN인가?
- **Q2.** 강의 기법 4가지(동적 padding, early exit, token pruning, GQA)가 각각 연산량과 지연 시간을 얼마나 줄이는가? 정확도는 얼마나 잃는가?
- **Q3.** TF-IDF → BERT cascade로 평균 비용을 얼마나 더 줄일 수 있는가?

**성공 기준 (목표치):** 기준 모델 대비 F1 하락 0.5%p 이내에서 평균 지연 시간 3배 이상 단축.

## 2. 이미 확인한 사실 (실측)

| 항목 | 값 |
|---|---|
| 행 수 | train 11,089 / validation 2,101 / test 2,101 |
| WordPiece 토큰 수 (`bert-base-uncased`, [CLS]·[SEP] 포함) | train 평균 14.1 (p95 29, 최대 110), validation 평균 11.0 (최대 40), test 평균 10.0 (최대 36) |
| 128토큰 초과 | 0건 |
| validation ∩ test 중복 | 116문장 |
| train ∩ validation 중복 | 4문장 |
| train 내부 중복 | 38행 |
| 라벨 충돌 | 1건 |
| TF-IDF + LogisticRegression 기준선 | test accuracy 0.950, ROC-AUC 0.989, 공격 recall 0.911 |

여기서 바로 나오는 관찰이 두 가지 있다.
- `max_length=128`로 고정 padding하면 test 기준으로 토큰의 약 92%가 [PAD]다. 동적 padding의 이득이 클 것으로 예상된다.
- train 문장이 test보다 평균 40% 길다. 분포 차이로 기록해 둔다.

## 3. 이론 복잡도 분석 (Ch5 §6 행렬 크기 기준)

BERT-base 한 층의 곱셈 횟수 (n = 토큰 수, d = 768, FFNN hidden = 4d):

| 부분 | 곱셈 횟수 |
|---|---|
| Q·K·V·출력 투영 | 4nd² |
| FFNN | 8nd² |
| QKᵀ + Attention 가중합 | 2n²d |

n²항의 비중은 n/(6d+n)이다. n=14이면 0.3%, n=128이면 2.7%, n=512이면 10%다.
→ 이 데이터에서 연산량은 사실상 **n × L × 12d²**에 비례한다. 줄일 대상은 n(padding, pruning), L(early exit), K·V 투영(GQA)이다.

각 기법의 이론 감소율을 먼저 계산하고, `torch.utils.flop_counter.FlopCounterMode`로 측정한 FLOPs와 대조한다.

**결과 (2026-11-19, `src/bench.py` → `results/complexity.json`, `results/latency_pilot.json`, `results/flops_vs_n.png`):**
- 이론식 `2·[L(12nd² + 2n²d) + d² + 2d]`는 BERT-base와 DistilBERT 모두 validation·`test_clean`에 나오는 모든 길이에서 `FlopCounterMode` 측정값과 정확히 일치했다 (오차 0).
- 128 고정 padding 대비 실제 길이로 처리하면 FLOPs가 validation에서 11.9배, `test_clean`에서 13.0배 줄어든다 (BERT-base 기준, DistilBERT도 같다).
- 그러나 batch 1 지연 시간(무작위 가중치, validation 200문장, median)은 FLOPs만큼 줄지 않았다.

| 모델 | 장치 | 128 고정 | 실제 길이 | 단축 배수 |
|---|---|---|---|---|
| BERT-base | MPS | 9.85 ms | 6.82 ms | 1.44배 |
| BERT-base | CPU 1스레드 | 24.97 ms | 10.17 ms | 2.46배 |
| DistilBERT | MPS | 5.09 ms | 3.51 ms | 1.45배 |
| DistilBERT | CPU 1스레드 | 12.51 ms | 5.18 ms | 2.41배 |

- 해석: 입력이 짧으면 연산 자체보다 연산마다 붙는 고정 비용(커널 실행 준비, Python 호출)이 지연 시간을 지배한다. DistilBERT가 BERT의 약 절반 시간인 것으로 보아, 이 고정 비용은 층 수에 비례한다.
- 이후 단계에 대한 가설: batch 1에서는 n을 줄이는 기법(T3 token pruning)보다 L을 줄이는 기법(T2 early exit)이 실제 시간을 더 많이 줄일 것이다.

## 4. 실험 설계

### 공통 규칙
- 하이퍼파라미터와 threshold는 **validation으로만** 정한다. 최종 평가는 정리된 test로 **한 번만** 한다.
- 학습이 들어가는 실험은 seed 3개(예: 13, 42, 2026)로 반복하고 평균 ± 표준편차를 보고한다.
- 정확도 지표: accuracy, macro F1, 공격 recall, 정상 오탐률(FPR), ROC-AUC
- 비용 지표:
  - FLOPs/샘플 (이론값과 측정값)
  - 지연 시간: batch 1에서 샘플당 median과 p95. warm-up 20회 후 측정하고, GPU는 측정 전에 동기화한다(`torch.cuda.synchronize` / `torch.mps.synchronize`)
  - 처리량: batch 32에서 초당 샘플 수
  - 토큰화 시간은 따로 보고한다
- 장비: Mac M4 Pro(MPS, CPU)에서 개발한다. 장비별 결과는 섞지 않고 따로 표로 만든다.

### 단계 0. 환경
- `uv venv --python 3.12` (시스템 Python 3.14는 쓰지 않는다)
- 패키지: `torch`, `transformers`, `datasets`, `scikit-learn`, `matplotlib`

### 단계 1. 데이터 정리 → `data/clean/`
1. train: 중복 39행 제거, 라벨 충돌 1건 제거, validation과 겹치는 4문장 제거
2. test: validation과 겹치는 116문장을 뺀 `test_clean`(약 1,985행)을 만든다. 원본 test 결과도 참고용으로 함께 보고한다.
3. 정리 전후 행 수와 라벨 비율을 기록한다.

**결과 (2026-11-11, `results/data_stats.json`):**
- train 11,089 → 11,045. 소문자화와 공백 정규화 기준으로 중복 39행(정확히 같은 문자열 기준으로는 38행), 라벨 충돌 1행, validation과 겹치는 4행을 제거했다.
- test 2,101 → `test_clean` 1,985. 빠진 116문장 중 109개가 공격이다. 그래서 `test_clean`의 라벨 비율이 1,043(정상) : 942(공격)로 50/50이 아니다. accuracy보다 macro F1과 클래스별 지표를 중심으로 본다.
- validation은 그대로다 (2,101행, 50/50).

### 단계 2. TF-IDF 재측정과 기준 모델 (B0)
- `bert-base-uncased` + `[CLS]` Dense(768×2), Ch7 §4 Single Text Classification 구조
- 시작값(관례): lr 2e-5, batch 32, 3 epoch, warm-up 10%, validation F1 기준 조기 종료
- **비교 기준은 `max_length=128` 고정 padding 추론이다.** 이것을 1배로 놓고 아래 기법을 비교한다.
- TF-IDF 기준선도 정리된 데이터로 다시 측정한다.
- **비교 모델 DistilBERT** (`distilbert-base-uncased`, 6층, 67M): 같은 설정으로 fine-tuning한다(seed 3개). "강의 기법으로 BERT를 줄인 것"과 "처음부터 층을 반으로 줄여 증류한 공개 모델"을 비교하기 위해서다. 특히 T2 early exit와 직접 비교된다. 2026-11-19 추가.

**TF-IDF 재측정 결과 (2026-11-19, `src/tfidf.py` → `results/tfidf.json`, `results/tfidf_probs.csv`):**
- C는 validation macro F1 기준으로 5를 골랐다 (C=2와 소수점 다섯째 자리까지 같음).

| split | macro F1 | 공격 recall | 정상 FPR | ROC-AUC |
|---|---|---|---|---|
| validation | 0.975 | 0.960 | 0.010 | 0.997 |
| `test_clean` | 0.951 | 0.910 | 0.011 | 0.988 |

- CPU batch 1 지연 시간은 median 0.40 ms다. BERT-base(MPS, 실제 길이)의 6.82 ms보다 약 17배 빠르다. cascade의 1단계로 쓸 근거가 된다.
- validation과 `test_clean`의 차이(공격 recall 0.960 → 0.910)는 validation에서 하이퍼파라미터를 고른 효과와 split 간 분포 차이가 섞인 결과다.

**B0 학습 결과 (2026-11-19, `src/model.py` → `results/b0/`, MPS, seed 13·42·2026, validation 기준 평균 ± 표준편차):**

| 모델 | macro F1 | 공격 recall | 정상 FPR | ROC-AUC | 고른 epoch | 학습 시간/회 |
|---|---|---|---|---|---|---|
| TF-IDF (참고) | 0.975 | 0.960 | 0.010 | 0.997 | — | 수 초 |
| BERT-base | 0.979 ± 0.005 | 0.968 ± 0.004 | 0.010 ± 0.005 | 0.998 ± 0.001 | 1, 3, 1 | 3.9분 |
| DistilBERT | 0.980 ± 0.003 | 0.965 ± 0.005 | 0.004 ± 0.002 | 0.999 ± 0.000 | 1, 1, 1 | 2.1분 |

- 확인 지점("validation F1이 TF-IDF보다 높음")은 평균 기준으로 통과했다. 다만 차이(0.004~0.005)가 seed 간 표준편차와 비슷해서 확실한 우위라고 보기는 어렵다.
- DistilBERT가 BERT-base와 같거나 약간 낫다 (차이는 편차 범위 안). 이 데이터는 6층으로도 충분하다는 신호이며, T2 early exit의 근거가 된다.
- 6회 중 5회가 epoch 1에서 가장 좋았고, epoch 2에서 대부분 떨어졌다. 짧은 단문 데이터라 1 epoch면 거의 수렴하는 것으로 보인다. epoch마다 validation으로 최선을 고르므로 validation 점수에는 선택에 따른 낙관 편향이 약간 있다.
- ROC-AUC(0.998~0.999)에 비해 F1이 낮다. threshold 0.5가 최적이 아닐 수 있으므로 단계 5에서 validation으로 threshold를 정한다.

### 단계 3. 기법 적용 (각각 단독 효과 측정 → 누적 효과 측정)

**T1. 동적 padding** (Ch4 §11, Ch6 §5 Padding Mask)
- batch 1이면 실제 길이 그대로, batch 32이면 배치 안 최대 길이로 padding한다.
- 길이순으로 정렬해 배치를 만드는 방식(length bucketing)도 비교한다.
- 가중치는 바꾸지 않으므로 정확도가 B0와 같아야 한다. 이것이 구현이 맞는지 확인하는 검사가 된다.

**T1 결과 (2026-11-19, `python src/bench.py padding` → `results/t1_padding.json`):**
- 구현 검증 통과: 체크포인트 6개 모두 128 고정과 동적 padding의 확률이 **비트 단위로 같다** (최대 차이 0.0). 길이순 정렬은 배치 구성이 바뀌어 최대 5.4e-6 차이가 나지만 예측이 바뀐 문장은 0개다.
- 0.0이 나온 이유: 각 행의 투영·FFNN 계산은 n과 무관하고, mask된 위치는 softmax에서 정확히 0이 되어 합에 영향을 주지 않는다.

| 측정 | 모델 | 장치 | 128 고정 | 동적 | 길이순 정렬 + 동적 |
|---|---|---|---|---|---|
| batch 1 지연 시간 (median) | BERT-base | MPS | 10.02 ms | 7.27 ms (1.38배) | — |
| | BERT-base | CPU 1스레드 | 24.62 ms | 9.65 ms (2.55배) | — |
| | DistilBERT | MPS | 5.24 ms | 3.87 ms (1.35배) | — |
| | DistilBERT | CPU 1스레드 | 12.58 ms | 5.00 ms (2.51배) | — |
| batch 32 처리량 (문장/초) | BERT-base | MPS | 196 | 939 (4.8배) | 1,616 (8.2배) |
| | BERT-base | CPU 1스레드 | 58 | 336 (5.8배) | 612 (10.5배) |
| | DistilBERT | MPS | 391 | 1,793 (4.6배) | 3,115 (8.0배) |
| | DistilBERT | CPU 1스레드 | 116 | 663 (5.7배) | 1,227 (10.6배) |

- 토큰화는 문장당 0.026 ms로 무시할 수 있다.
- 해석: batch 32에서는 고정 비용이 여러 문장에 나뉘어, 이득이 FLOPs 감소(11.9배)에 가까워진다 (CPU 길이순 정렬 10.5배). batch 1에서는 고정 비용 때문에 1.4~2.5배에 그친다. 길이순 정렬은 요청을 모아서 처리할 수 있는 일괄 검사(offline) 상황에서만 쓸 수 있다.

**T2. Early exit** (Ch7 §4 `[CLS]` + Dense, Ch5 층 구조)
1. 먼저 probing을 한다. B0의 backbone을 고정하고, 층 1~12의 `[CLS]` 벡터마다 분류기를 학습해 층별 정확도 곡선을 그린다.
2. 그 분류기를 exit head로 쓴다. 추론할 때 각 층에서 최대 확률이 τ 이상이면 그 층에서 멈춘다.
3. τ를 validation에서 바꿔가며 평균 통과 층 수와 F1의 관계 곡선을 그린다.
4. 시간이 남으면 모든 exit head와 backbone을 함께 학습(손실 가중합)하는 방식과 비교한다.
- batch 1 기준으로 측정한다. 배치에서는 샘플마다 멈추는 층이 달라 비교가 복잡해진다.

**T2 결과 (2026-11-19, `python src/model.py --stage exit` → `results/t2_exit.json`, `results/t2_exit.png`):**
- exit head: 층마다 `[CLS]` 벡터에 LogisticRegression을 학습했다 (train으로 학습, B0 고정). 마지막 층은 B0의 원래 분류 헤드를 그대로 쓴다. head는 `checkpoints/<run>/exit_heads.pt`에 저장했다.
- 구현 검증 통과: τ > 1(절대 멈추지 않음)로 층별 루프를 돌리면 B0 확률을 재현한다 (최대 차이 BERT 5.7e-6, DistilBERT 6.7e-7).
- probing: 1층 `[CLS]`만으로 macro F1 0.939(BERT) / 0.945(DistilBERT)다. DistilBERT는 3층에서 0.976으로 거의 최고에 이른다. 이 과제가 상당 부분 단어 수준 정보로 풀린다는 뜻이며, TF-IDF가 강한 이유와 맞는다. probing 곡선의 마지막 점(0.970~0.973)이 τ 탐색의 마지막 점(0.979~0.980)보다 낮은 것은, 전자가 `[CLS]`에 붙인 LogisticRegression이고 후자가 fine-tuning된 원래 헤드이기 때문이다.

τ 탐색 (validation, seed 3개 평균):

| 모델 | τ | 평균 통과 층 | macro F1 | 공격 recall | 정상 FPR | FLOPs (전체 층 대비) |
|---|---|---|---|---|---|---|
| BERT | 0.9 | 1.80 / 12 | 0.975 | 0.963 | 0.012 | 15% |
| BERT | 0.95 | 2.38 / 12 | 0.979 | 0.969 | 0.011 | 20% |
| BERT | 0.99 | 4.00 / 12 | 0.981 | 0.972 | 0.010 | 33% |
| BERT | 전체 (B0) | 12 | 0.979 | 0.968 | 0.010 | 100% |
| DistilBERT | 0.9 | 1.51 / 6 | 0.979 | 0.969 | 0.011 | 25% |
| DistilBERT | 전체 | 6 | 0.980 | 0.965 | 0.004 | 100% |

- 선택 규칙("F1 하락 0.005 이내에서 평균 층 수 최소")으로 고른 τ는 6개 중 5개가 0.9, 1개(BERT s2026)가 0.95다.
- **주의:** τ를 validation으로 고르고 같은 validation으로 평가했으므로 위 F1은 낙관적이다. `test_clean` 최종 평가에서 확인한다.

batch 1 지연 시간 (seed 13, validation 500문장, 단위 ms, median / **mean** / p95):

| 모델 | 장치 | HF 전체 실행 | early exit (τ=0.9) | 층별 루프, 멈추지 않음 |
|---|---|---|---|---|
| BERT | MPS | 7.36 / **7.43** / 8.66 | 0.92 / **1.52** / 3.91 | 8.70 / 8.88 / 10.63 |
| BERT | CPU 1스레드 | 9.82 / **10.03** / 11.13 | 0.86 / **1.55** / 4.08 | 9.85 / 10.24 / 12.51 |
| DistilBERT | MPS | 4.01 / **4.36** / 6.41 | 0.89 / **1.24** / 2.94 | 4.35 / 4.36 / 4.94 |
| DistilBERT | CPU 1스레드 | 4.95 / **4.99** / 5.26 | 0.84 / **1.25** / 3.27 | 4.91 / 5.02 / 5.44 |

- 멈추는 층이 문장마다 달라 분포가 치우쳐 있으므로 **평균**으로 비교한다. BERT 평균 지연 시간은 MPS에서 4.9배(7.43 → 1.52 ms), CPU에서 6.5배(10.03 → 1.55 ms) 줄었다. B0 기준(128 고정, MPS 10.02 ms) 대비로는 6.6배다.
- **BERT + early exit(1.52 ms)가 DistilBERT 전체 실행(4.36 ms)보다 빠르다.** batch 1 지연 시간에서는 강의 기법을 적용한 BERT가 공개 압축 모델보다 낫다.
- MPS에서 층별 루프가 HF 실행보다 약 1.4 ms 느리다(8.88 vs 7.43). 층마다 멈출지 판단하려고 장치를 동기화(`.item()`)하는 비용이다. 이 비용을 포함하고도 위 이득이 나왔다.
- 확인 지점("층별 정확도 곡선")을 통과했다. 성공 기준(F1 하락 0.5%p 이내, 평균 지연 시간 3배 이상 단축)도 validation 기준으로는 충족했다.

**T3. Attention 점수 기반 token pruning** (Ch4 §8·9 Attention Score)
- 층 k에서 head 평균 `[CLS]` → 토큰 attention이 낮은 토큰을 제거하고 상위 r%만 남긴다. `[CLS]`와 `[SEP]`는 항상 남긴다.
- k ∈ {4, 6, 8}, r ∈ {50%, 70%}로 탐색한다. 학습 없이 추론에서만 적용하는 것부터 시작한다.
- **가설:** 입력이 이미 짧아(평균 10~14토큰) 이득이 작다. 이득이 작다는 결과도 그대로 보고한다.

**T3 결과 (2026-11-19, `python src/model.py --stage prune` → `results/t3_prune.json`):**
- 구현 검증 통과: 직접 계산한 `[CLS]` attention이 HF eager attention 가중치와 일치하고(최대 차이 1.8e-7), 남기는 비율 100%이면 B0 출력을 재현한다.
- **가설이 확인됐다. FLOPs는 최대 32% 줄지만 지연 시간은 줄지 않았고, 정확도는 떨어졌다.**

validation, seed 3개 평균 (지연 시간은 seed 13, 500문장, batch 1 평균):

| 모델 | 설정 | macro F1 | 예측이 바뀐 문장 (seed 평균) | FLOPs | 지연 시간 MPS | 지연 시간 CPU 1스레드 |
|---|---|---|---|---|---|---|
| BERT | 자르지 않음 | 0.979 | — | 100% | 6.79 ms | 9.73 ms |
| BERT | 4층 뒤 50% | 0.946 | 116 | 68% | 6.44 ms | 10.46 ms |
| BERT | 6층 뒤 70% | 0.978 | 19 | 87% | 6.86 ms | 9.96 ms |
| BERT | 8층 뒤 70% | 0.978 | 16 | 91% | 6.91 ms | 9.83 ms |
| DistilBERT | 자르지 않음 | 0.980 | — | 100% | 3.40 ms | 4.90 ms |
| DistilBERT | 2층 뒤 50% | 0.952 | 93 | 68% | 3.38 ms | 5.37 ms |
| DistilBERT | 3층 뒤 50% | 0.978 | 32 | 76% | 3.48 ms | 4.99 ms |

- 지연 시간이 줄지 않은 이유: (1) batch 1에서는 토큰 수보다 층별 고정 비용이 지배한다(T1과 같은 결론). (2) attention 점수 계산과 토큰 선택(`topk`, 인덱싱)이 연산을 추가한다.
- 정확도: 일찍(4층 뒤) 많이(50%) 자를수록 크게 떨어진다. 짧은 문장에서는 단어 하나하나가 판정에 중요하다.
- T2와 함께 쓰는 실험은 하지 않는다. early exit에서 대부분 문장이 1~2층에서 끝나므로, 4층 이후 pruning까지 가지 않는다.
- 결론: 이 데이터의 batch 1 추론에서 token pruning은 쓸모가 없다. 긴 입력(512토큰 근처)이나 큰 배치에서만 의미가 있을 것이다.

**T4. MHA → GQA 변환** (Ch8 §8)
- B0의 K·V head 12개를 g개 그룹으로 평균 내서 묶는다(g ∈ {6, 4, 2, 1}). 그 뒤 짧게 추가 학습(uptraining, 약 1 epoch)한다.
- 이론 감소율은 층당 2nd²(1 − g/12) / 12nd²이다. g=4이면 약 11%다.
- **가설:** GQA는 decoder의 KV cache를 줄이려는 기법이라, encoder 분류에서는 이득이 작다.
- 위험 요소: HF `BertSelfAttention` 내부 구현이 버전마다 다르다. 자체 attention 모듈로 교체해 구현하고, g=12(변환 없음)일 때 출력이 B0와 같은지 먼저 확인한다.

**T4 결과 (2026-11-23, `python src/model.py --stage gqa` → `results/t4_gqa.json`, BERT만):**
- 구현: 자체 attention 모듈 대신, HF SDPA attention이 지원하는 `num_key_value_groups`를 쓴다. key·value 선형층만 g head 크기로 줄이고 평균으로 초기화한다.
- 구현 검증 통과: g=12 변환은 B0 출력과 완전히 같다(차이 0). `FlopCounterMode` 실측 FLOPs가 이론식과 정확히 같다. 저장 후 다시 불러와도 출력이 같다.
- uptraining: B0 checkpoint에서 변환한 뒤 B0와 같은 설정으로 1 epoch 학습한다(약 79초/회, 12회).
- **결과: 정확도는 거의 그대로 회복되고, FLOPs 감소만큼 시간도 조금 줄었다. 그러나 감소 폭이 작다(최대 약 10%).**

validation, seed 3개 평균 (지연 시간은 seed 13, 500문장, batch 1 평균):

| K·V head 수 g | 변환 직후 F1 | uptraining 후 F1 (±std) | FLOPs | 지연 시간 MPS | 지연 시간 CPU 1스레드 |
|---|---|---|---|---|---|
| 12 (B0) | — | 0.979 ± 0.005 | 100% | 7.74 ms | 11.11 ms |
| 6 | 0.928 | 0.971 ± 0.005 | 92% | 7.29 ms | 10.43 ms |
| 4 | 0.798 | 0.976 ± 0.004 | 89% | 7.17 ms | 9.98 ms |
| 2 | 0.620 | 0.977 ± 0.003 | 86% | 7.48 ms | 9.73 ms |
| 1 (MQA) | 0.384 | 0.978 ± 0.002 | 85% | 7.05 ms | 9.72 ms |

- 변환만 하면 정확도가 크게 떨어진다(g=1이면 0.38로 거의 무작위). 1 epoch uptraining이면 모든 g에서 B0와의 차이가 seed 편차 수준(≤0.8%p)으로 회복된다. g=6이 가장 낮은 것은 seed 편차로 본다.
- CPU에서는 시간이 FLOPs와 비슷한 비율로 줄었다(g=1: FLOPs 85%, 시간 87%). MPS에서는 고정 오버헤드 때문에 감소가 더 작다(g=1: 91%).
- 주의: 지연 시간은 g 순서대로 한 번씩 측정했다. 측정 순서와 발열의 영향을 배제하지 못했다. g=2 MPS 평균(7.48 ms)은 이상치 때문에 높고, 중앙값은 7.06 ms다.
- 가설은 대체로 맞다. GQA가 줄이는 것은 K·V 투영뿐이고, 이 부분은 encoder 연산의 일부다. KV cache가 없는 encoder 분류에서는 이득이 작다. T2(early exit, 약 5배)에 비하면 의미가 적다.
- 다만 정확도 손실 없이 FLOPs를 15% 줄이므로, T2와 함께 쓰면 추가 이득이 조금 있다. 단계 4·5에서는 B0와 GQA g=1을 후보로 함께 둔다.

### 단계 4. Cascade (TF-IDF → BERT)
- TF-IDF 확률이 [τ_low, τ_high] 구간 밖이면 TF-IDF가 그대로 판정한다. 구간 안이면 BERT로 보낸다.
- BERT 쪽은 단계 3의 최선 조합을 쓴다.
- 구간을 바꿔가며 BERT로 보내는 비율, F1, 평균 지연 시간을 기록한다. 기대 평균 지연 시간은 t_tfidf + p·t_bert이다(p = BERT로 보내는 비율).

**단계 4 결과 (2026-11-23, `python src/final.py cascade` → `results/final/cascade_validation.json`):**
- 구현: TF-IDF 확신도 max(p, 1−p)가 t 미만인 문장만 BERT 쪽(early exit, τ는 T2에서 고른 값)으로 보낸다. t는 BERT로 보내는 비율 q가 되도록 validation 확신도의 분위수로 정한다. 지연 시간은 TF-IDF → 토큰화 → early exit 전체 경로를 batch 1로 실제 측정했다.
- GQA g=1에도 T2를 적용했다(`--stage exit --models bert_gqa1 --name t2_exit_gqa1`). 얕은 층 probe 정확도가 B0보다 낮아(1층 F1 0.89 vs 0.94) 평균 3.2층까지 실행되고, MPS 평균 2.30 ms로 B0+exit(1.52 ms)보다 느리다.

validation 전체, seed 3개 평균 (지연 시간은 seed 13, 500문장, batch 1 평균, MPS):

| BERT로 보낸 비율 | BERT+exit F1 | 지연 시간 | DistilBERT+exit F1 | 지연 시간 | GQA1+exit F1 | 지연 시간 |
|---|---|---|---|---|---|---|
| 0% (TF-IDF만) | 0.975 | 0.51 ms | 0.975 | 0.45 ms | 0.975 | 0.54 ms |
| 5% | **0.982** | 0.79 ms | **0.982** | 0.64 ms | 0.980 | 1.01 ms |
| 10% | 0.982 | 0.96 ms | 0.981 | 0.80 ms | 0.978 | 1.25 ms |
| 20% | 0.981 | 1.30 ms | 0.981 | 0.91 ms | 0.977 | 1.39 ms |
| 50% | 0.978 | 2.19 ms | 0.980 | 1.51 ms | 0.976 | 2.39 ms |
| 100% (exit만) | 0.977 | 1.62 ms | 0.979 | 1.32 ms | 0.976 | 2.39 ms |

- 기준: B0 BERT HF 추론은 128 고정 padding 10.4 ms, 동적 padding 7.51 ms다.
- **cascade가 TF-IDF 단독, BERT 단독보다 모두 정확하다.** TF-IDF가 확신하지 못하는 5~20% 문장에서 BERT가 TF-IDF의 오답을 바로잡는다. 두 모델의 오답이 서로 다르다는 뜻이다.
- 50%에서 100%보다 느린 이유: 보내는 문장이 TF-IDF가 어려워한 문장이라 early exit도 깊이 내려간다. 거기에 TF-IDF 비용이 더해진다.
- 단계 5 규칙에 주는 영향: validation에서는 TF-IDF 단독이 B0와 0.5%p 이내다. 그래서 "0.5%p 이내에서 가장 싼 설정" 규칙을 쓰면 q=0(TF-IDF 단독)이 뽑힌다. 단계 5에서는 q를 "tune F1이 가장 높은 값"으로 고른다.

### 단계 5. 통합과 최종 평가
- 모든 변형 모델을 정확도-지연 시간 산점도에 놓고 Pareto 곡선을 그린다.
- 최종 후보 2~3개만 `test_clean`으로 한 번 평가한다.
- 오분류 분석: 라벨 노이즈, 키워드 의존 판정, 변형 표기, early exit에서 너무 일찍 멈춘 경우
- 2026-11-23 추가 (평가 신뢰도 보강): validation을 층화 50/50으로 tune(1,050)과 check(1,051)로 나눈다. τ, 보내는 비율, 판정 threshold는 tune에서만 고르고, 후보 비교와 최종 후보 선정은 check에서 한다. test_clean에서는 B0 대비 macro F1 차이의 paired bootstrap 95% CI(2,000회, seed 평균 차이)와 seed별 McNemar exact test를 보고한다.

**단계 5 결과 (2026-11-23, `python src/final.py final` → `results/final/final.json`, `pareto_check.png`, `errors_test.csv`):** `report.md`에 정리했다. 요약:
- 최종 후보(check에서 B0 − 0.5%p 이상인 후보 중 MPS 평균 지연 시간이 가장 짧은 3개): DistilBERT cascade, BERT cascade, GQA1 cascade.
- test_clean에서 셋 다 B0보다 F1이 높고(+0.7~1.2%p, bootstrap 95% CI가 모두 0보다 큼), B0 동적 padding보다 7~9배, 128 고정 padding보다 9~11배 빠르다(MPS). **성공 기준을 충족했다.**
- 절차상 공개: 첫 실행에서는 지연 시간을 seed 13 설정으로만 쟀다. 그런데 seed마다 보내는 비율이 0~35%로 달라 대표성이 없었다. 그래서 seed 3개 평균으로 고쳐 다시 실행했다. 이때 test 결과를 이미 한 번 본 상태였다. 최종 후보는 check만으로 정하며, 바뀐 것은 지연 시간 측정 방식뿐이다. 첫 실행의 후보는 DistilBERT cascade, GQA1 cascade, DistilBERT exit였다.

## 5. 파일 구성 (최소)

```
LLM_Class/
  PLAN.md
  data/prompt-injection-dataset/   원본 (수정 금지)
  data/clean/                      단계 1 결과
  src/
    prepare_data.py   단계 1
    tfidf.py          TF-IDF 기준선과 cascade 1단계
    model.py          B0 학습, exit head, token pruning, GQA 변환
    bench.py          FLOPs, 지연 시간, 처리량 측정
    final.py          단계 4 cascade, 단계 5 최종 평가
  results/            실험별 JSON과 그림
  checkpoints/        학습된 모델 (git 제외)
  report.md           최종 보고서
```

파일은 필요해질 때만 만든다. 기법별로 파일을 나누지 않고, `model.py` 안에서 옵션으로 켜고 끈다.

## 6. 진행 순서와 확인 지점

단계 번호는 4장 실험 설계의 번호와 같다.

| 단계 | 작업 | 확인 지점 (통과해야 다음 단계로) | 상태 |
|---|---|---|---|
| 0·1 | 환경, 데이터 정리 | 정리 후 행 수 기록, 중복 0건 확인 | 완료 (2026-11-11) |
| 2 | 복잡도 분석(3장), TF-IDF 재측정, B0 학습 | 이론 FLOPs와 측정 FLOPs의 차이가 10% 이내 / B0 validation F1이 TF-IDF보다 높음 | 완료 (2026-11-19) |
| 3-T1 | 동적 padding | 정확도가 B0와 동일 (구현 검증) | 완료 (2026-11-19) |
| 3-T2 | probing → early exit | 층별 정확도 곡선 | 완료 (2026-11-19) |
| 3-T3·T4 | token pruning, GQA 변환 | g=12 출력이 B0와 일치 (T4 구현 검증) | T3 완료 (2026-11-19), T4 완료 (2026-11-23) |
| 4 | Cascade | BERT로 보내는 비율-F1 곡선 | 완료 (2026-11-23) |
| 5 | 통합, 최종 평가, 보고서 | 성공 기준 달성 여부를 그대로 보고 | 완료 (2026-11-23) |

## 7. 위험 요소와 대응

| 위험 | 대응 |
|---|---|
| MPS 지연 시간 측정이 불안정함 | 동기화 후 반복 측정하고 median을 쓴다. CPU 단일 스레드 결과를 함께 보고한다 |
| batch 1에서는 GPU 실행 준비 오버헤드가 커서 FLOPs 감소가 시간 감소로 이어지지 않을 수 있음 | FLOPs와 실제 시간을 둘 다 보고하고, 차이의 원인을 분석에 넣는다 |
| validation이 작아(2,101행) F1 차이가 우연일 수 있음 | seed 3개를 반복하고 편차를 함께 보고한다 |
| 라벨 노이즈 | 오분류 분석에서 사람이 검토한 의심 라벨 목록을 따로 정리한다 |
| HF 내부 구현 변경 | `transformers` 버전을 고정하고 기록한다 |
