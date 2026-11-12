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
1. train: 중복 38행 제거, 라벨 충돌 1건 제거, validation과 겹치는 4문장 제거
2. test: validation과 겹치는 116문장을 뺀 `test_clean`(약 1,985행)을 만든다. 원본 test 결과도 참고용으로 함께 보고한다.
3. 정리 전후 행 수와 라벨 비율을 기록한다.

**결과 (2026-11-11, `results/data_stats.json`):**
- train 11,089 → 11,045. 소문자화와 공백 정규화 기준으로 중복 39행(정확히 같은 문자열 기준으로는 38행), 라벨 충돌 1행, validation과 겹치는 4행을 제거했다.
- test 2,101 → `test_clean` 1,985. 빠진 116문장 중 109개가 공격이다. 그래서 `test_clean`의 라벨 비율이 1,043(정상) : 942(공격)로 50/50이 아니다. accuracy보다 macro F1과 클래스별 지표를 중심으로 본다.
- validation은 그대로다 (2,101행, 50/50).

### 단계 2. 기준 모델 (B0)
- `bert-base-uncased` + `[CLS]` Dense(768×2), Ch7 §4 Single Text Classification 구조
- 시작값(관례): lr 2e-5, batch 32, 3 epoch, warm-up 10%, validation F1 기준 조기 종료
- **비교 기준은 `max_length=128` 고정 padding 추론이다.** 이것을 1배로 놓고 아래 기법을 비교한다.
- TF-IDF 기준선도 정리된 데이터로 다시 측정한다.

### 단계 3. 기법 적용 (각각 단독 효과 측정 → 누적 효과 측정)

**T1. 동적 padding** (Ch4 §11, Ch6 §5 Padding Mask)
- batch 1이면 실제 길이 그대로, batch 32이면 배치 안 최대 길이로 padding한다.
- 길이순으로 정렬해 배치를 만드는 방식(length bucketing)도 비교한다.
- 가중치는 바꾸지 않으므로 정확도가 B0와 같아야 한다. 이것이 구현이 맞는지 확인하는 검사가 된다.

**T2. Early exit** (Ch7 §4 `[CLS]` + Dense, Ch5 층 구조)
1. 먼저 probing을 한다. B0의 backbone을 고정하고, 층 1~12의 `[CLS]` 벡터마다 분류기를 학습해 층별 정확도 곡선을 그린다.
2. 그 분류기를 exit head로 쓴다. 추론할 때 각 층에서 최대 확률이 τ 이상이면 그 층에서 멈춘다.
3. τ를 validation에서 바꿔가며 평균 통과 층 수와 F1의 관계 곡선을 그린다.
4. 시간이 남으면 모든 exit head와 backbone을 함께 학습(손실 가중합)하는 방식과 비교한다.
- batch 1 기준으로 측정한다. 배치에서는 샘플마다 멈추는 층이 달라 비교가 복잡해진다.

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
  report.md           최종 보고서
```

파일은 필요해질 때만 만든다. 기법별로 파일을 나누지 않고, `model.py` 안에서 옵션으로 켜고 끈다.

## 6. 진행 순서와 확인 지점

| 순서 | 작업 | 확인 지점 (통과해야 다음 단계로) |
|---|---|---|
| 1 | 환경, 데이터 정리 | 정리 후 행 수 기록, 중복 0건 확인 |
| 2 | 복잡도 분석, TF-IDF 재측정 | 이론 FLOPs와 측정 FLOPs의 차이가 10% 이내 |
| 3 | B0 학습 | validation F1이 TF-IDF보다 높음 |
| 4 | T1 | 정확도가 B0와 동일 (구현 검증) |
| 5 | T2 probing → early exit | 층별 정확도 곡선 |
| 6 | T3, T4 | g=12 출력이 B0와 일치 (T4 구현 검증) |
| 7 | Cascade | BERT로 보내는 비율-F1 곡선 |
| 8 | 통합, 최종 평가, 보고서 | 성공 기준 달성 여부를 그대로 보고 |

## 7. 위험 요소와 대응

| 위험 | 대응 |
|---|---|
| MPS 지연 시간 측정이 불안정함 | 동기화 후 반복 측정하고 median을 쓴다. CPU 단일 스레드 결과를 함께 보고한다 |
| batch 1에서는 GPU 실행 준비 오버헤드가 커서 FLOPs 감소가 시간 감소로 이어지지 않을 수 있음 | FLOPs와 실제 시간을 둘 다 보고하고, 차이의 원인을 분석에 넣는다 |
| validation이 작아(2,101행) F1 차이가 우연일 수 있음 | seed 3개를 반복하고 편차를 함께 보고한다 |
| 라벨 노이즈 | 오분류 분석에서 사람이 검토한 의심 라벨 목록을 따로 정리한다 |
| HF 내부 구현 변경 | `transformers` 버전을 고정하고 기록한다 |
