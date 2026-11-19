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

**T4. MHA → GQA 변환** (Ch8 §8)
- B0의 K·V head 12개를 g개 그룹으로 평균 내서 묶는다(g ∈ {6, 4, 2, 1}). 그 뒤 짧게 추가 학습(uptraining, 약 1 epoch)한다.
- 이론 감소율은 층당 2nd²(1 − g/12) / 12nd²이다. g=4이면 약 11%다.
- **가설:** GQA는 decoder의 KV cache를 줄이려는 기법이라, encoder 분류에서는 이득이 작다.
- 위험 요소: HF `BertSelfAttention` 내부 구현이 버전마다 다르다. 자체 attention 모듈로 교체해 구현하고, g=12(변환 없음)일 때 출력이 B0와 같은지 먼저 확인한다.

### 단계 4. Cascade (TF-IDF → BERT)
- TF-IDF 확률이 [τ_low, τ_high] 구간 밖이면 TF-IDF가 그대로 판정한다. 구간 안이면 BERT로 보낸다.
- BERT 쪽은 단계 3의 최선 조합을 쓴다.
- 구간을 바꿔가며 BERT로 보내는 비율, F1, 평균 지연 시간을 기록한다. 기대 평균 지연 시간은 t_tfidf + p·t_bert이다(p = BERT로 보내는 비율).

### 단계 5. 통합과 최종 평가
- 모든 변형 모델을 정확도-지연 시간 산점도에 놓고 Pareto 곡선을 그린다.
- 최종 후보 2~3개만 `test_clean`으로 한 번 평가한다.
- 오분류 분석: 라벨 노이즈, 키워드 의존 판정, 변형 표기, early exit에서 너무 일찍 멈춘 경우

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
| 3-T3·T4 | token pruning, GQA 변환 | g=12 출력이 B0와 일치 (T4 구현 검증) | |
| 4 | Cascade | BERT로 보내는 비율-F1 곡선 | |
| 5 | 통합, 최종 평가, 보고서 | 성공 기준 달성 여부를 그대로 보고 | |

## 7. 위험 요소와 대응

| 위험 | 대응 |
|---|---|
| MPS 지연 시간 측정이 불안정함 | 동기화 후 반복 측정하고 median을 쓴다. CPU 단일 스레드 결과를 함께 보고한다 |
| batch 1에서는 GPU 실행 준비 오버헤드가 커서 FLOPs 감소가 시간 감소로 이어지지 않을 수 있음 | FLOPs와 실제 시간을 둘 다 보고하고, 차이의 원인을 분석에 넣는다 |
| validation이 작아(2,101행) F1 차이가 우연일 수 있음 | seed 3개를 반복하고 편차를 함께 보고한다 |
| 라벨 노이즈 | 오분류 분석에서 사람이 검토한 의심 라벨 목록을 따로 정리한다 |
| HF 내부 구현 변경 | `transformers` 버전을 고정하고 기록한다 |
