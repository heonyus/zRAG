알겠습니다. 지금 상황에서 “교수님 씽크”에 정확히 맞추면서도 **RAG 비교가 가능**한 형태로, **이번에 딱 1개 실험**을 어떻게 돌리면 되는지 **실행 절차 중심**으로 정리하겠습니다. (그 다음에 바로 이어질 **후속 실험 로드맵**도 같이 드리겠습니다.)

---

## 0) 교수님 아이디어를 한 문장으로 정리하면

**Vector DB(문서 저장소) + Retriever를, “로컬 LLM이 학습한 연속 벡터(z)”로 대체해서**
`(retrieved z들 + query) → 로컬 LLM → evidence(또는 answer)`가 되게 만들고, 최종 답변은 필요하면 큰 LLM(ChatGPT)에 넘기는 구조입니다.

여기서 교수님이 “제일 단순하게라도 먼저 해보라”는 건,

* **문서(긴 텍스트) 하나가 들어오면**
* **random init된 z(learnable vector / learned token)를 붙여 놓고**
* **NLL(= -log likelihood) 최소화로 z를 최적화해서**
* **그 z를 넣으면 문서가 “나오게” 만들라**
  입니다. 이게 “write(저장)”의 가장 원형입니다.

---

## 1) MetaQueries랑 연결은 뭐냐

MetaQueries는 “**학습 가능한 query 벡터(learnable queries)** 를 LLM 앞에 붙여서, **LLM(대개 frozen)** 이 원하는 출력을 하도록 만드는 계열”입니다. 즉, **텍스트 토큰이 아니라 연속 벡터(query/token)를 최적화해서 모델을 조종한다**는 발상 자체가 연결점입니다. ([OpenReview][1])

당신 실험에서는 “learnable queries”가

* MetaQueries: **query 쪽에 붙는 learnable vector**
* 우리 방향: **문서/메모리 슬롯을 대표하는 learnable z (doc-conditioned vector)**
  로 바뀐다고 보면 됩니다.

---

## 2) “이번 주에 돌릴 1개 실험” — 가장 현실적인 설계

요구조건(3B~8B, L4, RAG 비교, 최신 RAG 평가 방식)을 만족시키면서 **진짜로 한 번에 결과가 나오는** 실험을 추천합니다.

### Experiment-1: **MIRAGE 기반 “z-RAG(=VectorDB→z)” vs 표준 RAG**

MIRAGE는 RAG에서 중요한 **노이즈 취약성/문맥 오해**까지 같이 평가하는 벤치마크입니다. (EM도 같이 봅니다.) ([GitHub][2])
이 실험이 좋은 이유는: “RAG다움”을 잃지 않으면서 **교수님이 말한 ‘LLM이 DB가 된다’**를 수치로 보여줄 수 있습니다.

---

### 2.1 데이터/태스크 선택

* **평가 벤치마크:** MIRAGE ([GitHub][2])
* **처음 1개만 고르라면:** MIRAGE 안에서 **NaturalQA(NQ)** 또는 **TriviaQA** 계열 파트부터 시작(ODQA 성격이라 가장 보편적)

MIRAGE는 **RAG에서 중요한 4가지 지표(NV/CA/CI/CM)** 와 QA 성능(EM loose/strict)을 같이 봅니다. ([GitHub][2])

---

### 2.2 비교 대상(베이스라인) — “동일 조건” 맞추는 법

이번 실험에서 비교는 최소 3개로 잡으시면 됩니다.

1. **Closed-book (No-RAG)**

* 입력: `query`만
* 목적: “로컬 LLM 자체 능력” 하한선

2. **Standard RAG (Vector DB + Retrieve text + LLM)**

* Retriever: BM25 또는 dense(예: bge 계열) 중 1개 고정
* 입력: `top-k passage text + query`
* 목적: 우리가 이길 상대(가장 정석)

3. **z-RAG (Vector DB 대신 z로 저장)**

* 입력: `top-k의 z 들 + query`
* 목적: “vectorstore를 LLM 메모리로 대체” 성립 여부

여기서 **Retriever는 2)와 3)에서 동일하게** 씁니다. 그래야 “저장/읽기 방식 차이”만 비교됩니다.

---

### 2.3 모델/컴퓨트 (L4 기준)

* 로컬 LLM: **3B로 시작 권장** (예: Qwen 계열 3B / Llama 계열 3B).
  7B도 가능하지만, 이번 실험은 **“저장/읽기 메커니즘 성립”**이 1순위라 3B가 안전합니다.
* z 형태:

  * `m`개의 **soft tokens** (예: m=16)
  * 실제 구현은 embedding vector `(m, hidden_dim)`을 학습하는 형태

---

## 3) z-RAG를 “진짜로 어떻게 돌리냐” — 단계별 실행 절차

핵심은 **Write(저장)** 와 **Read(사용)** 를 명시적으로 나누는 겁니다.

---

### Step A. Write(저장): passage → z 학습 (교수님이 말한 그거)

각 passage(문서/청크)마다 z를 하나 배정하고, 아래 loss로 z만 학습합니다.

* LLM 파라미터: **freeze**
* 학습 파라미터: **z만 업데이트**
* 목표: z를 prefix로 넣으면 passage가 잘 생성되게

손실:

* **NLL 최소화**
  [
  \mathcal{L}_{write}(z_i) = -\log p(D_i \mid z_i)
  ]
  이게 교수님이 말한 “로그라이클리후드 최소화(=NLL 최소화)로 토큰을 옵티마이즈”입니다.

실무적으로는:

* passage 길이를 256~512 토큰 정도로 잘라서 시작
* z 학습 step 수는 작게(예: 50~200 step)
* 그리고 **캐시**: 한 번 만든 z는 저장해두고 재사용
  → “LLM이 토큰 DB가 된다”를 실험적으로 구현

---

### Step B. Read(사용): query가 들어오면 z를 써서 evidence/answer 생성

이제 RAG처럼 합니다.

1. Retriever로 top-k passage id를 얻음 (Standard RAG와 동일)
2. 각 passage id에 대응하는 z를 가져옴
3. 로컬 LLM 입력:

   * `z_1, z_2, ..., z_k + query`
4. 출력은 두 가지 중 하나로 고정 (이번엔 **하나만**)

   * (추천) **answer 직접 생성**
   * 또는 evidence 먼저 생성 → (추후) 큰 LLM에 전달

이번 실험은 “비교”가 목적이라 **answer 직접 생성**이 더 깔끔합니다(큰 LLM을 끼면 변수가 늘어납니다).

---

## 4) 평가지표/리포트는 이렇게 내면 됩니다

### 4.1 1차 지표 (RAG 벤치마크 표준)

* **MIRAGE EM_loose / EM_strict** ([GitHub][2])
* **MIRAGE 4대 지표**

  * NV (Noise Vulnerability)
  * CA (Context Acceptability)
  * CI (Context Insensitivity)
  * CM (Context Misinterpretation) ([GitHub][2])
* MIRAGE는 이걸 묶어 overall score 형태도 씁니다. ([GitHub][2])

### 4.2 진단 지표 (왜 좋아졌는지 설명용)

* **RAGChecker**를 붙이면 “retrieval이 문제인지, generation이 문제인지”를 claim-level로 쪼개서 보여줄 수 있습니다. ([arXiv][3])
  (논문/미팅에서 설득력 올라갑니다.)

---

## 5) “기존 RAG보다 성능이 높아졌다”를 어떻게 설득하냐 (포인트)

이번 실험에서 노려야 하는 승리 포인트는 보통 둘 중 하나입니다.

1. **노이즈 상황(Mixed)에서의 강건성**

* Standard RAG는 top-k 문서에 노이즈가 섞이면 **CM/CI가 튀는** 경우가 많습니다.
* z-RAG는 “텍스트를 그대로 넣는 게 아니라, 압축된 z를 통해 읽는 구조”라 **노이즈 민감도가 달라질 가능성**이 있습니다.
  → MIRAGE 지표로 바로 증명 가능 ([GitHub][2])

2. **컨텍스트 예산 대비 성능**

* Standard RAG는 k개 passage를 넣으면 context length가 급증
* z-RAG는 `k*m` soft token으로 고정
  → **동일 context budget에서 더 많은 후보를 고려**하는 식의 설계로 확장 가능
  (이건 후속 실험으로 아주 깔끔합니다)

---

## 6) 후속 실험(다음 주) — 당신이 원한 “LLM weights까지 학습해서 initialization 가능한 z”

이번 주 실험은 교수님 방식(가장 단순)으로 “DB가 된다”를 보여주는 거고,
그 다음이 당신이 정리한 목표(LLM도 학습해서 z init이 되게)입니다.

### Follow-up 1: “z initializer” 학습 (online z 최적화 제거)

* 지금: passage마다 z를 매번 SGD로 만들었음(느림)
* 다음: `z = g(D)` 형태로 **문서에서 z를 한 번에 뽑는 initializer(encoder/MLP)** 를 학습
* 이 방향은 **DyPRAG(문서로부터 파라미터를 생성하는 translator)** 같은 계열과 직접적으로 맞닿습니다. ([OpenReview][4])

### Follow-up 2: “LLM weights + z”로 end-to-end 학습 (당신이 적은 형태)

[
\mathcal{L} = -\log p(\text{answer/evidence} \mid z_i, q)
]
이건 “parametric RAG” 계열(문서를 파라미터/모듈로 저장)과 강하게 유사합니다. ([arXiv][5])

### Follow-up 3: Retriever까지 LLM이 먹기 (generative retrieval)

“query가 들어오면 관련 문서를 (ID로) 생성”하는 계열은 이미 존재합니다. 예를 들어 DSI는 seq2seq로 **문서 식별자 자체를 생성**하는 형태로 “검색 인덱스 자체를 모델에 내장”하려는 흐름입니다.
당신이 하려는 건 “ID 생성”보다 더 강하게 “evidence 텍스트 생성”까지 가는 쪽이라 차별점이 생깁니다.

---

## 7) 관련 연구가 ‘똑같이’ 있었냐? (냉정하게)

* “**learnable vector를 앞에 붙여서 원하는 출력을 만든다**” → 너무 흔합니다(soft prompt/prompt tuning/MetaQueries). ([OpenReview][1])
* “**문서를 파라미터/모듈로 저장해서 RAG를 대체/보완**” → Parametric RAG / DyPRAG 계열이 이미 있습니다. ([arXiv][5])
* 다만 당신이 적은 형태처럼 **‘(z_i + query) → evidence’를 주 타겟으로, z를 “문서별 메모리 슬롯”으로 두고** RAG 평가(MIRAGE 같은)에서 정면승부하는 설계는 **조합 자체가 연구적으로 충분히 의미가 있습니다**(특히 write/read 분리 + new doc init까지 붙이면).

---

# 결론: 지금 당장 “어떻게 실험”하시면 되냐

1. **MIRAGE**로 평가 프레임부터 잡습니다. ([GitHub][2])
2. 베이스라인 3개를 동일 retriever 조건으로 준비합니다: **Closed-book / Standard RAG / z-RAG**
3. z-RAG는 이번 주에는 교수님 말대로 **LLM freeze + z만 NLL로 최적화(Write)** 를 먼저 구현합니다.
4. 그 z를 retrieval 결과에 대응시켜 **(z_topk + query)로 answer 생성(Read)** 을 돌립니다.
5. MIRAGE EM + NV/CA/CI/CM로 “RAG 대비 이점(특히 노이즈/문맥오해)”을 수치로 보고,
6. 다음 주에 **initializer + LLM finetune**으로 넘어갑니다. (이게 당신이 원하는 “init 가능한 z + RAG처럼 생성”의 본체)

---

원하시면, 제가 다음 메시지에서 **Experiment-1을 바로 코드 구조로 쪼개서**:

* 데이터 로딩(미라지)
* retriever 구성
* z write(최적화) 루프
* inference 프롬프트
* MIRAGE eval 실행 커맨드
  까지 “그대로 따라 치면 돌아가는 체크리스트” 형태로 드리겠습니다.

[1]: https://openreview.net/revisions?id=L2PjBCQCN2&utm_source=chatgpt.com "Revision History for Transfer between Modalities with..."
[2]: https://github.com/nlpai-lab/MIRAGE "GitHub - nlpai-lab/MIRAGE: MIRAGE is a light benchmark to evaluate RAG performance."
[3]: https://arxiv.org/html/2408.08067v2?utm_source=chatgpt.com "RagChecker: A Fine-grained Framework for Diagnosing ..."
[4]: https://openreview.net/forum?id=TWMiVfRwSF&utm_source=chatgpt.com "Dynamic Parametric Retrieval Augmented Generation"
[5]: https://arxiv.org/pdf/2501.15915?utm_source=chatgpt.com "Parametric Retrieval Augmented Generation"