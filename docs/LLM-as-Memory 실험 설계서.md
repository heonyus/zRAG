# LLM-as-Memory: Vector DB를 Local LLM으로 대체하는 실험 설계

## 1. 핵심 아이디어

### 1.1 문제 정의

기존 RAG 파이프라인의 한계:
- 큰 LLM + 별도 retriever + Vector DB + reranker → 복잡한 다단계 구조
- 오류 원인 추적/교정이 어려움
- 시스템 복잡도 증가

**교수님 방향**: Vector DB를 LLM으로 대체
- 문서가 들어오면 LLM 내부에 "기억 가능한 상태"를 만들고
- Query가 들어오면 그 상태를 통해 **관련 근거(evidence)를 생성**
- Retrieval을 embedding similarity가 아닌 **LLM 생성/추론 메커니즘**으로 대체

### 1.2 핵심 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    Local LLM 시스템                         │
│                                                             │
│   Memory Pool: Z = {z_i} (벡터/soft token)                 │
│                                                             │
│   Query ──→ [내부 routing/attention over Z] ──→ Evidence    │
│     │                                          (텍스트)     │
│     │                                              │        │
└─────┼──────────────────────────────────────────────┼────────┘
      │                                              │
      └──────────────┬───────────────────────────────┘
                     │
                     ▼
         Query + Evidence (텍스트) ──→ ChatGPT ──→ Answer
```

### 1.3 핵심 구분

| 구분 | 형태 | 설명 |
|------|------|------|
| **내부 메모리 (z_i)** | 벡터/soft token | 문서를 대표하는 learnable embedding |
| **외부 출력 (evidence)** | 텍스트 | ChatGPT가 이해할 수 있는 근거 텍스트 |
| **문서 선택** | 내부 routing | 외부 retriever/selector 없음 |

---

## 2. 기존 RAG vs 제안 방식

```
기존 RAG:
Query → [Retriever] → Vector DB → Retrieved Text → LLM → Answer
              ↑
         외부 모듈

제안 방식:
Query → [Local LLM + Internal Memory Z] → Evidence → (+Query) → ChatGPT → Answer
                    ↑
              외부 모듈 없음
              내부 routing으로 z 선택
```

### 차별점

| 기존 RAG | 제안 방식 |
|----------|-----------|
| 외부 retriever 필요 | **retriever 제거** |
| Vector DB에 텍스트 저장 | **z_i (벡터)로 압축 저장** |
| Embedding similarity로 검색 | **LLM 내부 attention/routing** |
| Retrieved text를 LLM에 전달 | **Evidence 텍스트 생성** |

---

## 3. 학습 설계

### 3.1 목표 함수

$$\min_{\theta, \{z_i\}} -\log P_\theta(\text{evidence} \mid \text{query}, Z)$$

- $\theta$: Local LLM 파라미터 (QLoRA로 학습)
- $Z = \{z_i\}$: 문서별 learnable memory vector
- evidence: 생성할 근거 텍스트

### 3.2 학습 데이터

```
(query, gold_document, evidence) 튜플

- query: 질문
- gold_document: 정답이 포함된 문서 (z_i와 매핑)
- evidence:
  - (단순) 문서에서 answer span 주변 텍스트
  - (가능하면) 데이터셋이 제공하는 supporting evidence
```

### 3.3 학습 파이프라인

```python
# 학습 루프 (개념적)
for query, z_i, evidence in dataloader:
    # z_i는 해당 문서의 memory vector
    # Z 전체가 모델 내부 메모리로 존재

    # Forward: query + Z → evidence 생성
    logits = local_llm(query, memory=Z)

    # Loss: evidence 생성 확률 최대화
    loss = -log_prob(evidence | query, Z)

    # Backward: θ (LoRA) + z_i 업데이트
    loss.backward()
    optimizer.step()
```

### 3.4 내부 Routing 메커니즘

외부 selector 없이, LLM 내부에서 query가 Z에 대해 attention/routing:

```
Option A: Cross-Attention
    Query embedding이 Z의 각 z_i에 attention
    → 관련 z_i들이 자동으로 가중치 높게

Option B: Soft Prompt Concatenation
    Z 전체를 prefix로 → LLM이 내부적으로 필요한 정보 선택

Option C: Memory-Augmented Transformer
    Z를 key-value memory로 → query가 memory read
```

---

## 4. 추론 파이프라인

```
Input:  Query만 (외부 document selection 없음)

Step 1: Local LLM이 내부 메모리 Z를 참조하여 evidence 생성
        query → Local LLM(Z) → evidence (텍스트)

Step 2: Query + Evidence를 ChatGPT에 전달하여 최종 답변
        (query + evidence) → ChatGPT → answer

Output: Answer
```

### 추론 코드 (개념적)

```python
def inference(query, local_llm, chatgpt_api):
    # Step 1: Local LLM이 evidence 생성
    # Z는 모델 내부에 이미 존재 (학습된 memory)
    evidence = local_llm.generate(query)  # 내부적으로 Z 참조

    # Step 2: Query + Evidence → ChatGPT로 최종 답변
    prompt = f"""Based on the following evidence, answer the question.

Evidence: {evidence}

Question: {query}

Answer:"""
    answer = chatgpt_api(prompt)

    return answer
```

---

## 5. 실험 설계

### 5.1 데이터셋

| 데이터셋 | 크기 | 특징 | 용도 |
|----------|------|------|------|
| **Natural Questions** | 79K QA pairs | Single-hop, Wikipedia 기반 | Primary |
| **HotpotQA** | 113K | Multi-hop reasoning | Extension |

### 5.2 Corpus 설정

- **POC**: 2K~5K passages
- **Full**: 10K~50K passages
- 각 passage에 대해 z_i 학습

### 5.3 Baselines

| Baseline | 설명 | 비교 목적 |
|----------|------|-----------|
| **No Retrieval** | LLM만 사용 | 하한선 |
| **BM25 + LLM** | Sparse retrieval | 전통적 방법 |
| **Dense RAG (E5)** | Dense retrieval + LLM | Standard RAG |
| **Our (z + Local LLM)** | 제안 방식 | 주요 비교 대상 |

### 5.4 평가 메트릭

**Answer Quality:**
- Exact Match (EM)
- Token F1

**Evidence Quality:**
- ROUGE-L (생성된 evidence vs gold evidence)
- Recall@K (gold document 포함 여부 - 내부 routing 평가용)

**RAG-specific (RAGAS):**
- Faithfulness: evidence에 기반한 답변인가
- Answer Relevancy: 질문과 관련 있는가
- Context Precision/Recall

**Efficiency:**
- Latency (ms/query)
- Storage (MB/1K docs): z_i vs Vector DB

### 5.5 실험 단계

| Phase | 기간 | 목표 | 성공 기준 |
|-------|------|------|-----------|
| **POC** | 3일 | 구조 작동 확인 | Evidence 생성 가능, ROUGE-L > 0.3 |
| **Ablation** | 4일 | 하이퍼파라미터 탐색 | 최적 z_dim, routing 방식 |
| **Full Eval** | 5일 | Baseline 비교 | EM within 5% of RAG |
| **Multi-hop** | 3일 | HotpotQA 확장 | Multi-hop 한계 분석 |

---

## 6. 하이퍼파라미터

### 6.1 Memory Vector

| 파라미터 | 탐색 범위 | 설명 |
|----------|-----------|------|
| `z_dim` | {64, 128, 256, 512} | 각 z_i의 차원 |
| `m_tokens` | {1, 4, 8, 16} | 문서당 memory token 수 |

### 6.2 모델

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `local_llm` | Qwen3-8B / Llama-3.1-8B | Local LLM |
| `quantization` | 4-bit (QLoRA) | 메모리 효율 |
| `lora_r` | {8, 32, 64} | LoRA rank |

### 6.3 학습

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `lr_llm` | 2e-5 | LLM (LoRA) learning rate |
| `lr_z` | 1e-3 | Memory vector learning rate |
| `batch_size` | 2~4 | GPU 메모리에 따라 |
| `epochs` | 3~5 | 학습 |

---

## 7. 새 문서 적응 (New Document Adaptation)

교수님 요구: "랜덤 벡터를 최적화"

새 문서가 들어왔을 때:
1. z_new를 랜덤 초기화
2. θ는 freeze
3. z_new만 몇 step 최적화 (문서 정보 인코딩)
4. z_new + query → evidence 성능 측정

```python
def adapt_new_document(new_doc, local_llm, steps=50):
    # 1. 랜덤 초기화
    z_new = nn.Parameter(torch.randn(m_tokens, z_dim) * 0.02)

    # 2. LLM freeze
    for p in local_llm.parameters():
        p.requires_grad = False

    # 3. z_new만 최적화
    optimizer = AdamW([z_new], lr=5e-3)
    for _ in range(steps):
        loss = -log_prob(new_doc | z_new, local_llm)
        loss.backward()
        optimizer.step()

    # 4. Memory pool에 추가
    memory_pool.add(z_new)
    return z_new
```

---

## 8. 기대 결과 및 의의

### 8.1 기대 결과

| 메트릭 | RAG Baseline | 제안 방식 (목표) |
|--------|--------------|------------------|
| EM | ~28% | ~25% (within 5%) |
| Latency | ~150ms | ~50ms (3x↓) |
| Storage/1K | ~150MB | ~3MB (50x↓) |

### 8.2 연구적 의의

1. **Retriever 제거**: 외부 모듈 없이 LLM 내부에서 memory access
2. **Vector DB → Learnable Memory**: 텍스트 저장 → 벡터 압축
3. **End-to-end 학습**: 분리된 retriever 없이 단일 모델 최적화
4. **빠른 적응**: 새 문서에 대해 z만 학습 (전체 재학습 불필요)

---

## 9. 주요 참고 문헌

- **Soft Prompt**: Prompt Tuning, P-tuning v2, Prefix Tuning
- **Generative Retrieval**: DSI, DSI++ (doc ID 생성 vs evidence 생성)
- **Document Compression**: Gist Tokens, ICAE, xRAG, 500xCompressor
- **Parametric RAG**: DyPRAG (가장 유사한 선행연구)
- **MetaQueries**: Learnable query가 모달리티 간 인터페이스 역할

---

## 10. 요약

```
핵심 한 줄:
Query → Local LLM (내부 memory Z routing) → Evidence (텍스트)
    → Query + Evidence → ChatGPT → Answer

학습 목표:
-log P(evidence | query, Z; θ)

차별점:
1. 외부 retriever/selector 없음 (내부 routing)
2. z_i는 벡터, 출력은 텍스트
3. Vector DB 완전 대체
```
