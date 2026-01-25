아래는 지금까지 대화에서 정리된 **“교수님이 원하시는 방향(= Vector DB를 작은 LLM로 대체)”**을 기준으로, **왜 하는지 → 무엇을 만들지 → MetaQueries와의 연결 → 선행연구 포지셔닝 → 이번 주 1개 실험(비교대상/데이터셋/메트릭/절차) → 후속 실험 로드맵**까지 한 번에 읽히는 **보고서 형태**로 정리한 문서입니다.

---

# LLM-as-Memory Parametric RAG 실험 설계 보고서 (2026-01-25 기준)

## 0. 결론부터: “그래서 어떻게 실험하나요?”

이번 주 **단 1개 실험**은 아래로 고정합니다.

* **실험 1 (이번 주)**:
  **(문서 집합 C)**를 “Vector DB”에 넣는 대신,
  **작은 Local LLM(3B~8B)을 end-to-end로 학습**시켜
  **learnable vector `z_i`(문서별 latent key) + query**를 넣으면
  **evidence(근거 텍스트)를 RAG처럼 생성**하게 만들고,
  **동일 조건의 기존 RAG 대비 성능을 비교**합니다.

핵심은 “토큰만 학습”이 아니라, 사용자가 마지막에 정리한 대로 **`z_i` + query → evidence**를 **LLM weights까지 같이 학습**해서 “`z` 인터페이스 자체가 잘 작동하도록” 만드는 것입니다.

---

## 1. 왜 이걸 하나? (교수님 문제의식)

교수님 요지는 다음 2개로 압축됩니다.

1. **기존 RAG 파이프라인의 다단계 한계**

* “큰 LLM(예: ChatGPT)” + “별도 retriever” + “vector DB” + “reranker” …
  구조가 길어질수록 **오류 원인 추적/교정이 어렵고**, 시스템 복잡도가 커집니다.

2. **Vector DB를 LLM로 바꿔보자 (retriever 자체를 LLM로)**

* 문서가 들어오면 어떤 형태로든 **LLM 내부에 ‘기억 가능한 상태’**를 만들고
* query가 들어오면 그 상태를 통해 **관련 근거(evidence)를 생성**하게 해서
* “retrieval”을 **embedding similarity**가 아니라 **LLM 생성/추론 메커니즘**으로 대체해보자는 방향입니다.

---

## 2. 교수님이 말한 “가장 간단한 것부터”의 정확한 의미

교수님이 처음 강조한 simplest step은 “개념 증명”입니다.

* 랜덤 초기화된 벡터(토큰/연속 벡터 등)를 두고
* 이를 최적화(optimizer에 올려서)해서
* **log-likelihood 기반 목적함수(NLL; negative log-likelihood)를 최소화**
  → “그 벡터를 앞에 넣으면 특정 텍스트가 잘 나오게” 만들라는 것

이건 학계 용어로는 **soft prompt / prompt tuning / prefix tuning 계열**과 맞닿아 있습니다. ([Hugging Face][1])
다만 교수님 방향은 여기서 끝이 아니라, 당신이 정리한 대로 **RAG 태스크로 확장**해서 “vector DB를 대체”하는 쪽입니다.

---

## 3. MetaQueries(Transfer between Modalities with MetaQueries)와 무슨 관계인가?

교수님이 MetaQueries를 읽으라고 한 이유는 “형태(모달리티)가 달라도 **learnable query/vector가 강력한 인터페이스**가 된다”는 점 때문입니다.

* MetaQueries:
  **learnable queries**를 두고, 그것이 backbone 모델과 결합되면서 다른 모달리티로 transfer하는 “인터페이스” 역할을 합니다. ([arXiv][2])

* 우리(교수님 방향의 RAG 버전):
  문서별로 **learnable `z_i`**를 두고,
  **`z_i + query` → evidence**가 되도록 학습시켜
  `z_i`를 “문서 접근 키 + 생성 조건”으로 쓰는 구조입니다.

즉, MetaQueries는 “learnable vector로 모델을 조종/조건화하는 발상”을 RAG의 메모리/리트리빙으로 가져오라는 레퍼런스입니다.

---

## 4. “완전히 똑같은 연구가 있었나?”에 대한 냉정한 답

**완전히 동일한 조합(문서별 learnable z + query → evidence, 그리고 새 문서에 대해 z를 빠르게 초기화/적응시키는 RAG 대체 파이프라인)**은, 제가 확인 가능한 범위에서 **표준화된 대표 레퍼런스로 굳어져 있지는 않습니다.**
다만, 아래 연구들이 **가장 가까운 주변부**입니다.

### (A) Soft Prompt 계열: “learnable vector로 모델 출력 제어”

* Prompt Tuning / Prefix Tuning / P-tuning v2: learnable continuous prompt로 태스크를 제어 ([Hugging Face][1])
  → 하지만 보통 “태스크 단위”가 중심이고, “문서 저장/검색 RAG 대체”가 정면 타깃은 아닙니다.

### (B) Generative Retrieval / Parametric Index: “모델이 doc id를 생성해서 검색”

* DSI(Transformer Memory as a Differentiable Search Index): query → doc identifier를 생성하는 식의 “parametric retrieval” 라인 ([Emergent Mind][3])
* DSI++: indexing/업데이트 관점 확장(증분 인덱싱 아이디어에 힌트)
  → 하지만 “doc id 생성”이 중심이고, 당신이 말한 **`z_i`를 문서별 latent로 두고 evidence를 직접 생성**하는 형태와는 목적함수/인터페이스가 다릅니다.

### (C) RAG 고도화: retrieval이 잘못되었을 때를 다루는 라인

* Self-RAG, CRAG 등은 retrieval/생성 루프를 개선하지만 “vector DB를 없애는 것”이 본질 목적은 아닙니다. ([arXiv][4])

정리하면:
“없는 것 같다”는 직감은 **절반은 맞고**, 절반은 “가까운 이웃 연구(soft prompt, generative retrieval)가 이미 있다”가 정확합니다. 당신의 설계는 **이웃들을 조합해 ‘RAG의 DB를 z+LLM로 치환’**한다는 점에서 연구로서 충분히 설득 가능한 포지션입니다.

---

## 5. 이번 주 실험 1개: 무엇을, 어떤 비교대상으로, 어떤 메트릭으로, 어떻게 돌릴지

### 5.1 실험 목표 (1개만)

**Vector DB 기반 RAG 대비**,
**(Local LLM + 문서별 z 테이블)**이

* (i) evidence 생성 품질
* (ii) 최종 QA 정답률/faithfulness
  에서 이길 수 있는지 확인합니다.

### 5.2 모델/환경 제약

* Local LLM: **3B~8B** (예: Llama 3.x 8B, Qwen2.5 7B, Mistral 7B 등) ([Hugging Face][5])
* GPU: GCP **G2(L4 24GB)** 기준으로 설계 ([Google Cloud Documentation][6])
  → 현실적으로 **QLoRA/LoRA + z만 full precision** 같은 구성이 가장 안전합니다.

### 5.3 데이터셋 선택 (2026 기준 “RAG 평가에 많이 쓰이는 축”)

이번 실험은 “최신 벤치마크 프레임”을 따르는 게 좋습니다.

* **MIRAGE**: RAG 시스템의 취약점(노이즈/문맥 둔감/오독 등)을 체계적으로 평가하려는 벤치마크 축 ([ACL Anthology][7])
* **RAGBench / T2-RAGBench**: RAG 평가용 벤치마크/확장 흐름 ([alphaXiv][8])
* **CRAG(= Corrective RAG) 논문이 자주 쓰는 세팅/데이터(예: PopQA 평가 언급)**도 “현업 RAG 비교 프레임”으로 설득력 있습니다. ([arXiv][9])

**이번 주 1개 실험 추천(현실성 포함):**

* “MIRAGE에서 제공/정의하는 QA 세팅” 또는 “RAGBench류 QA 세팅” 중 하나로 고정하고,
* corpus 규모는 L4에 맞게 **작게 시작(N=2k~10k passage)** 하십시오.
  (스케일 확장은 후속 실험으로 넘기는 게 맞습니다.)

### 5.4 평가 메트릭 (2026 기준 RAG에서 ‘많이 쓰고’, 논문에서 납득되는 것)

RAG 평가는 보통 “정답률”만 보면 부족해서, **retrieval + generation + faithfulness**를 같이 봅니다.

* **Answer Quality (정답 품질)**

  * EM / F1 (open-domain QA 표준)
* **Retrieval Quality (검색 품질)**

  * Recall@k (gold evidence 포함 여부), MRR/nDCG (가능하면)
* **RAG-specific / Faithfulness (근거성/정합성)**

  * RAGChecker: retrieval/generation을 분해해 평가하는 프레임 ([arXiv][10])
  * RAGAS: context precision/recall, faithfulness, answer relevancy 등 ([arXiv][11])
  * ARES: 자동평가를 통계적으로 다루려는 흐름(신뢰구간 등) ([ACL Anthology][12])

**이번 주는 과욕 금지:**

* 논문 보고용 “최소 세트”는 **(EM/F1) + (Recall@k) + (RAGAS 또는 RAGChecker)**로 충분합니다.

---

## 6. 실험 1의 시스템 설계 (당신이 적어둔 수식 그대로 구현 가능한 형태)

### 6.1 우리가 만들 모델(= Vector DB 대체 버전)

당신이 정리한 그대로를 **실행 가능한 최소형(MVP)**로 바꾸면 아래입니다.

* 파라미터

  * Local LLM weights: θ
  * 문서별 learnable vector: {z_i} (i=1..N)

* 학습 데이터 (핵심)

  * (D_i, q, evidence) 튜플
  * input: [z_i ; q]
  * output: evidence

* Loss

  * **L = - log p(evidence | z_i, q; θ)**

여기서 “Write(D→z)”가 명시적이지 않다는 문제는, 이번 주에는 **학습 데이터 구성으로 흡수**합니다:

* 한 문서 D_i에 대해 여러 질문 q를 붙여서 학습시키면
  z_i가 D_i를 “답변 가능한 형태로 압축”하지 않으면 손실이 줄지 않습니다.
  → 결과적으로 z_i가 문서 메모리가 됩니다(암묵적 write).

### 6.2 “새 문서가 오면 z를 어떻게 초기화/적응?” (교수님이 말한 initialization의 실험화)

이번 주 실험 1에 **옵션으로 넣을 수 있는 가장 설득력 있는 추가 평가**는 이겁니다:

* Train 단계: θ와 기존 {z_i}를 학습
* Test 단계(holdout 문서 D_new):

  1. z_new를 random init
  2. θ는 freeze
  3. z_new만 몇 step 최적화해서 “D_new를 잘 재현/설명”하게 만든 뒤
  4. z_new + q → evidence 성능 측정

이건 교수님이 처음 말한 “랜덤 벡터를 옵티마이즈”와 정합적이고, 동시에 당신이 원하는 “LLM 전체학습으로 z 인터페이스 학습”도 반영합니다.

---

## 7. 비교 대상 RAG 시스템(동일 조건 비교) — 이번 주 실험 1에 넣을 것 vs 후속으로 뺄 것

### 7.1 이번 주 실험 1에 “반드시” 넣을 베이스라인 (최소 2개)

1. **Naive RAG (BM25)**

* BM25로 top-k passage → (동일한 최종 LLM)에게 넣고 답 생성

2. **Dense RAG (SOTA급 임베딩 + vector DB)**

* e5/bge 계열 dense retriever로 top-k → 동일 최종 LLM
  (논문/실무에서 “강한 기본선”으로 가장 설득됩니다)

3. **Hybrid(+ optional reranker)** 는 가능하면 넣되, 이번 주 시간이 촉박하면 후속으로 넘기십시오.
   (다만 “SOTA급 RAG”라고 말하려면 hybrid+rerkank가 보통 필요합니다.)

### 7.2 “최신 고도화 RAG(Self-RAG/CRAG)”는 후속 실험로 미루는 게 합리적

* Self-RAG, CRAG는 구현/세팅 변수가 많아서 “이번 주 1개 실험” 목표를 흐립니다. ([arXiv][4])
  → **후속 실험**에서 “우리 방식 vs 고도화 RAG”로 붙이는 게 맞습니다.

---

## 8. 이번 주 실험 1: 실행 절차(체크리스트)

아래 순서 그대로 진행하면, “보고 가능한 결과”가 나옵니다.

### Step A. Corpus 구성 (작게)

* N=2,000~10,000 passages로 시작
* 각 passage에 doc id i 부여, learnable z_i 생성

### Step B. 학습 데이터 구성

* 각 QA 샘플에 대해 “정답을 포함하는 passage(D_i)”를 gold로 지정
* evidence는

  * (가장 단순) D_i에서 answer span 주변 window 텍스트
  * 또는 데이터셋이 제공하는 supporting evidence 사용(가능하면)

### Step C. 학습

* Local LLM(3B~8B)을 **QLoRA/LoRA**로 학습(θ 업데이트)
* 동시에 {z_i}도 학습
* Objective: -log p(evidence | z_i, q)

### Step D. 추론

* (간단 버전) “정답 passage의 z_i를 oracle로 준다”는 통제 실험도 가능하지만,
  교수님 취지(=RAG 대체)를 위해서는 아래 중 하나는 해야 합니다.

  * 방식 1(현실적): top-k 후보 z를 “간단한 점수로” 선택(예: z_i별 프롬프트 likelihood proxy)
  * 방식 2(정합적): DSI류처럼 query→id를 생성하게 해서 z 선택 ([Emergent Mind][3])
    이번 주에는 **방식 1로 MVP를 만들고**, 방식 2는 후속으로 넣는 게 안전합니다.

### Step E. 평가

* EM/F1
* Recall@k(“gold passage가 top-k에 들어갔는지”)
* RAGAS 또는 RAGChecker로 faithfulness/answer relevancy까지 ([arXiv][11])

### Step F. 최종 LLM(예: ChatGPT) 결합은 “동일 조건”으로만

* 당신 말대로 최종 output을 큰 LLM로 넘길 거면,
  **모든 베이스라인도 동일하게 큰 LLM을 쓰고**,
  비교는 “evidence 품질 차이로 인해 최종 답이 얼마나 좋아졌는지”로 합니다.

---

## 9. 후속 실험 로드맵 (이번 주 이후)

이번 주 1개 실험이 끝나면, 다음은 “논문감”을 만드는 방향으로 갑니다.

1. **Retrieval을 완전히 parametric하게 만들기**

* query→(id 또는 z 선택)을 LLM이 생성하도록(D SI/DSI++ 라인 접목) ([Emergent Mind][3])

2. **새 문서 유입(online write) 성능**

* z_new를 few-step으로 적응시키는 “update latency vs 품질” 곡선 제시
* 이게 나오면 교수님이 말한 “initialization 가능한 vector”가 명확해집니다.

3. **스케일 확장**

* z 테이블이 커질 때 (N=100k) 어떻게 검색/라우팅할지(계층화, 코드북 등)

4. **노이즈/오답 문서 주입에 대한 강건성**

* MIRAGE류 설정으로 “RAG failure mode”를 정면 타격 ([ACL Anthology][7])

---

## 10. 이번 주 결과물(교수님 보고용) 형태

교수님께는 아래 4장 슬라이드면 충분합니다.

1. 문제정의: “Vector DB → (Local LLM + z) 치환”
2. 모델: z_i + query → evidence (학습식 1줄)
3. 실험셋업: corpus 크기, local LLM, 베이스라인(BM25/dense)
4. 결과표: EM/F1 + Recall@k + (RAGAS/RAGChecker 일부) + 코스트(시간/메모리)

---

# 마지막으로: “지금 당장 무엇부터 하면 되나요?”

당장 착수 순서를 한 줄로 고정하면:

1. **Corpus를 2k~10k passages로 확정**
2. **(q, gold passage, answer/evidence) 튜플**을 만들고
3. **`-log p(evidence | z_i, q)`**로 **Local LLM(3~8B) + z_i** 학습
4. **BM25 vs Dense-RAG vs Our(z+LLM)** 를 **EM/F1, Recall@k, RAGAS/RAGChecker**로 비교

---



[1]: https://huggingface.co/papers/2104.08691?utm_source=chatgpt.com "The Power of Scale for Parameter-Efficient Prompt Tuning"
[2]: https://arxiv.org/abs/2504.06256?utm_source=chatgpt.com "Transfer between Modalities with MetaQueries"
[3]: https://www.emergentmind.com/papers/2202.06991?utm_source=chatgpt.com "Transformer Memory as Search Index"
[4]: https://arxiv.org/abs/2310.11511?utm_source=chatgpt.com "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
[5]: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct?utm_source=chatgpt.com "meta-llama/Llama-3.1-8B-Instruct"
[6]: https://docs.cloud.google.com/compute/docs/machine-resource?utm_source=chatgpt.com "Machine families resource and comparison guide"
[7]: https://aclanthology.org/2025.findings-naacl.157.pdf?utm_source=chatgpt.com "MIRAGE: A Metric-Intensive Benchmark for Retrieval- ..."
[8]: https://www.alphaxiv.org/overview/2407.11005?utm_source=chatgpt.com "RAGBench: Explainable Benchmark for Retrieval- ..."
[9]: https://arxiv.org/pdf/2401.15884?utm_source=chatgpt.com "arXiv:2401.15884v3 [cs.CL] 7 Oct 2024"
[10]: https://arxiv.org/html/2408.08067v1?utm_source=chatgpt.com "RagChecker: A Fine-grained Framework for Diagnosing ..."
[11]: https://arxiv.org/html/2309.15217v1?utm_source=chatgpt.com "RAGAS: Automated Evaluation of Retrieval Augmented ..."
[12]: https://aclanthology.org/2024.naacl-long.20.pdf?utm_source=chatgpt.com "ARES: An Automated Evaluation Framework for Retrieval- ..."