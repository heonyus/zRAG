# LLM as Memory 연구 종합 보고서

**연구자:** Honey  
**작성일:** 2026년 1월 25일  
**버전:** 3.0 (Final)

---

## 목차

1. [연구 배경: 왜 이 연구를 하는가?](#1-연구-배경-왜-이-연구를-하는가)
2. [연구 방향 변화 히스토리](#2-연구-방향-변화-히스토리)
3. [최종 연구 방향: Parametric QA](#3-최종-연구-방향-parametric-qa)
4. [핵심 아이디어 상세](#4-핵심-아이디어-상세)
5. [구현 계획](#5-구현-계획)
6. [실험 설계](#6-실험-설계)
7. [기대 결과 및 의의](#7-기대-결과-및-의의)
8. [일정 및 리소스](#8-일정-및-리소스)
9. [참고 문헌 및 관련 연구](#9-참고-문헌-및-관련-연구)

---

# 1. 연구 배경: 왜 이 연구를 하는가?

## 1.1 큰 그림: LLM as Memory 패러다임

### 현재 RAG의 한계

현재 대부분의 LLM 시스템은 **RAG (Retrieval-Augmented Generation)** 방식을 사용한다:

```
[기존 RAG 파이프라인]
Query → Vector DB 검색 → Top-K 문서 → LLM에 컨텍스트로 제공 → Answer
```

**RAG의 문제점:**
| 문제 | 설명 |
|------|------|
| **Latency** | 매 쿼리마다 retrieval 필요 (수십~수백 ms) |
| **Storage** | 전체 문서 + 임베딩 저장 필요 (수십 GB) |
| **Retrieval Error** | 검색이 잘못되면 답변도 잘못됨 |
| **Context Length** | 긴 문서는 잘라야 함, 정보 손실 |
| **Integration** | LLM과 retriever가 분리되어 최적화 어려움 |

### LLM as Memory: 새로운 패러다임

**핵심 철학:** 외부 벡터 DB나 검색 시스템에 의존하는 대신, **LLM의 파라미터 자체를 메모리 저장소로 활용**

```
[LLM as Memory 패러다임]
Write (저장): 문서 → LLM 파라미터/학습된 토큰에 인코딩
Read (검색):  Query → LLM Forward Pass → Answer (별도 retrieval 없음)
```

**이론적 기반:** Geva et al. (EMNLP 2021)
- 트랜스포머의 **FFN 레이어가 Key-Value 메모리로 작동**
- W₁ = Key (텍스트 패턴), W₂ = Value (출력 분포)
- 이 발견이 ROME, MEMIT 등 모델 편집 연구의 기초

## 1.2 연구 동기

### 교수님의 연구 비전

> "RAG를 LLM 내부 메모리로 대체하여, retriever 없이 LLM이 직접 정보를 저장하고 검색하는 시스템"

**두 가지 메모리 유형:**

| 유형 | 방법 | 속도 | 용도 |
|------|------|------|------|
| **Short-term Memory** | Token Learning (LLM freeze) | 빠름 | 최근/임시 정보 |
| **Long-term Memory** | Full LLM Fine-tuning | 느림 | 중요/영구 정보 |

### 왜 중요한가?

1. **효율성:** Retrieval latency 제거, inference만으로 정보 접근
2. **압축:** 문서를 compact vector로 압축 (50-100x 저장 공간 절약)
3. **통합:** LLM과 메모리가 하나의 시스템으로 end-to-end 최적화 가능
4. **확장성:** 외부 인프라(Vector DB) 없이 LLM만으로 운영

---

# 2. 연구 방향 변화 히스토리

## 2.1 Phase 1: M+ 기반 연구 (초기 계획)

### 원래 계획: Write-time Retrieval + Iterative Retrieval

**M+ 논문 (ICML 2025) 분석** 기반으로 두 가지 연구 아이디어 수립:

**아이디어 B: Write-time Retrieval**
```
문제: 새 정보를 저장할 때 기존 메모리와 연결 없이 독립 저장
     → 관련 정보가 함께 검색되지 않음 (Co-retrieval rate 35%)

해결: 저장 전에 관련 기존 메모리를 검색하고 연결
     related = retrieve(new_info)
     store(new_info, context=related)
```

**아이디어 D: Iterative Retrieval with Learned Termination**
```
문제: 고정 5회 검색 → 58%가 불필요한 검색
해결: 학습된 Sufficiency Predictor로 언제 멈출지 결정
     while not is_sufficient(context):
         retrieve(query)
```

### 준비한 것들

1. **M+ 코드 분석:** `inject_memory()` 함수의 Semantic Disconnection 문제 발견
2. **실험 설계:** B-1 ~ B-6, D-1 ~ D-6 총 13개 실험 계획
3. **데이터셋:** HotpotQA, MuSiQue 준비
4. **평가 지표:** Co-retrieval Rate, Sufficiency Accuracy 정의

## 2.2 Phase 2: 교수님 방향 전환 (1월 23일)

### 교수님 새로운 지시

> "M+ 분석보다 더 근본적인 걸 해봐. **Token Learning**으로 문서를 압축하는 거야."

**핵심 개념:**
```python
# 문서 D가 있을 때
# Random token z를 학습시켜서
# z를 넣으면 D가 생성되도록

z = torch.randn(hidden_size, requires_grad=True)  # learnable

for step in range(1000):
    loss = -log P(D | z)  # = Cross Entropy
    loss.backward()
    optimizer.step()  # z만 업데이트

# 학습 후: z → LLM → D 재생성
```

**교수님 언급 논문:** MetaQueries (2025)
- Learnable queries가 frozen MLLM과 diffusion model 사이의 interface 역할
- "LLM freeze, learnable vector만 학습"의 선례

## 2.3 Phase 3: 최종 방향 확정 - RAG 대체 시스템

### 교수님과의 논의 결과

**단순 Token Learning이 아니라:**
1. Token만 학습 (LLM freeze) ❌ - 이건 sanity check용
2. **LLM 전체 + learnable vector 함께 학습** ✅ - 이게 진짜 목표

**최종 정의된 목표:**

```
목표 A (Write): max log p(D_i | z_i)
  → z_i를 넣으면 문서 D_i가 나오도록 학습

목표 B (Read): max log p(evidence | q, M)  
  → Query q와 메모리 M으로 관련 정보 생성 (RAG처럼)
```

**"Initialization 가능한 Vector"의 의미:**
- 새 문서가 들어와도 학습 없이/few-step으로 z 생성 가능해야 함
- 형태 1: learnable embedding z_i (각 문서마다)
- 형태 2: encoder f(D_i) = z_i (범용 encoder)

---

# 3. 최종 연구 방향: Parametric QA

## 3.1 핵심 아이디어

### 기존 RAG vs Parametric QA

```
[기존 RAG]
문서 저장: D → Chunking → Embedding → Vector DB
질의 응답: Query → Retriever → Top-K Chunks → LLM → Answer

[Parametric QA (제안)]
문서 저장: D → Learnable Vector z 학습 (Write Phase)
질의 응답: Query → z 선택 → LLM(z, Query) → Answer (Read Phase)
```

### 왜 "Parametric"인가?

- 문서 정보가 **파라미터(z)에 저장**됨
- 별도 Vector DB, Retriever 불필요
- LLM이 z를 "읽어서" 정보 추출

## 3.2 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                    Parametric QA System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ════════════════════════════════════════════════════════════   │
│  ║              WRITE PHASE (Offline)                       ║   │
│  ════════════════════════════════════════════════════════════   │
│                                                                  │
│    Document D_i                                                  │
│         │                                                        │
│         ▼                                                        │
│    ┌─────────────┐                                              │
│    │ Initialize  │  z_i = random or encoder(D_i)                │
│    │    z_i      │                                              │
│    └─────────────┘                                              │
│         │                                                        │
│         ▼                                                        │
│    ┌─────────────┐                                              │
│    │  Optimize   │  min -log P(D_i | z_i; LLM)                  │
│    │    z_i      │  (LLM도 함께 학습 가능)                       │
│    └─────────────┘                                              │
│         │                                                        │
│         ▼                                                        │
│    ┌─────────────┐                                              │
│    │   Store     │  Memory Pool: {z_1, z_2, ..., z_N}           │
│    │    z_i      │                                              │
│    └─────────────┘                                              │
│                                                                  │
│  ════════════════════════════════════════════════════════════   │
│  ║              READ PHASE (Inference)                      ║   │
│  ════════════════════════════════════════════════════════════   │
│                                                                  │
│    Query q                                                       │
│         │                                                        │
│         ▼                                                        │
│    ┌─────────────┐                                              │
│    │  Select     │  Top-K z_i by similarity(q, z_i)             │
│    │    z_i      │  또는 learned router                         │
│    └─────────────┘                                              │
│         │                                                        │
│         ▼                                                        │
│    ┌─────────────┐                                              │
│    │    LLM      │  Input: [z_1, z_2, ..., z_k, query tokens]   │
│    │  Generate   │                                              │
│    └─────────────┘                                              │
│         │                                                        │
│         ▼                                                        │
│      Answer                                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 3.3 관련 연구와의 비교

| 방법 | Write | Read | LLM 학습 | 특징 |
|------|-------|------|----------|------|
| **RAG** | Vector DB 저장 | Retriever | ❌ | 표준 방식 |
| **MemoryLLM** | Memory pool 업데이트 | Cross-attention | ✅ | 1B 메모리 파라미터 |
| **ICAE** | Encoder로 압축 | Decoder 생성 | 부분 | 4-16x 압축 |
| **xRAG** | Retriever embedding 재사용 | Modality bridge | ❌ | 178x 압축 |
| **Ours** | z_i 최적화 | z_i 선택 + LLM | ✅ | RAG 완전 대체 목표 |

---

# 4. 핵심 아이디어 상세

## 4.1 Write Phase: 문서를 Vector로 압축

### 목표 함수

```
max log P(D_i | z_i; θ)

D_i: 문서 텍스트
z_i: 학습 가능한 벡터 (dim = hidden_size × m tokens)
θ: LLM 파라미터 (학습 여부 선택 가능)
```

### 구현 상세

```python
class WritePhase:
    def __init__(self, llm, z_dim=256, m_tokens=8):
        self.llm = llm
        self.z_dim = z_dim
        self.m_tokens = m_tokens
        
        # Document별 learnable vector
        # shape: [num_docs, m_tokens, hidden_size]
        self.doc_vectors = nn.ParameterDict()
    
    def learn_vector(self, doc_id, document_text, epochs=100):
        """단일 문서에 대한 z_i 학습"""
        
        # 1. z_i 초기화 (random 또는 문서 임베딩 기반)
        z_i = nn.Parameter(torch.randn(self.m_tokens, self.llm.config.hidden_size) * 0.02)
        
        # 2. 문서 토큰화
        doc_tokens = self.tokenizer(document_text, return_tensors='pt')
        
        # 3. 최적화
        optimizer = torch.optim.AdamW([z_i], lr=1e-3)
        
        for epoch in range(epochs):
            # z_i를 LLM embedding space로 변환
            z_embed = z_i.unsqueeze(0)  # [1, m, hidden]
            
            # Document token embeddings
            doc_embed = self.llm.get_input_embeddings()(doc_tokens.input_ids)
            
            # Concatenate: [z_i tokens] + [doc tokens]
            # Teacher forcing: z_i가 주어졌을 때 doc이 나오도록
            combined = torch.cat([z_embed, doc_embed[:, :-1, :]], dim=1)
            
            # Forward pass
            outputs = self.llm(inputs_embeds=combined)
            
            # Loss: -log P(doc | z_i)
            # 예측 대상은 z_i 이후의 모든 토큰
            logits = outputs.logits[:, self.m_tokens-1:-1, :]
            targets = doc_tokens.input_ids[:, 1:]
            
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), 
                                   targets.reshape(-1))
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # 4. 학습된 z_i 저장
        self.doc_vectors[doc_id] = z_i.detach()
        
        return z_i
```

### 학습 옵션

| 옵션 | LLM | z_i | 용도 |
|------|-----|-----|------|
| **Option 1** | Freeze | 학습 | Sanity check, 빠른 실험 |
| **Option 2** | LoRA/QLoRA | 학습 | 균형 (추천) |
| **Option 3** | Full FT | 학습 | 최대 성능 (비용 높음) |

## 4.2 Read Phase: Query로 정보 검색

### 목표 함수

```
max log P(answer | query, z_selected; θ)

query: 사용자 질문
z_selected: 선택된 관련 문서 벡터들
answer: 정답
```

### Document Selection 전략

**방법 1: Similarity-based (Simple)**
```python
def select_documents(self, query, k=5):
    # Query 임베딩
    q_tokens = self.tokenizer(query, return_tensors='pt')
    q_embed = self.llm.get_input_embeddings()(q_tokens.input_ids).mean(dim=1)
    
    # 모든 z_i와 similarity 계산
    scores = []
    for doc_id, z_i in self.doc_vectors.items():
        z_mean = z_i.mean(dim=0)  # [hidden_size]
        score = F.cosine_similarity(q_embed, z_mean.unsqueeze(0))
        scores.append((doc_id, score.item()))
    
    # Top-K 선택
    top_k = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
    return [doc_id for doc_id, _ in top_k]
```

**방법 2: Learned Router (Advanced)**
```python
class LearnedRouter(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query_proj = nn.Linear(hidden_size, hidden_size // 4)
        self.doc_proj = nn.Linear(hidden_size, hidden_size // 4)
        self.score_head = nn.Linear(hidden_size // 2, 1)
    
    def forward(self, query_embed, doc_vectors):
        q = self.query_proj(query_embed)  # [1, h/4]
        d = self.doc_proj(doc_vectors)    # [N, h/4]
        
        # Concatenate and score
        combined = torch.cat([q.expand(d.size(0), -1), d], dim=-1)
        scores = self.score_head(combined).squeeze(-1)  # [N]
        
        return F.softmax(scores, dim=-1)
```

### Answer Generation

```python
def generate_answer(self, query, selected_z):
    """선택된 z들로 답변 생성"""
    
    # 1. z vectors를 embedding으로 변환
    z_embeds = torch.stack([self.doc_vectors[z_id] for z_id in selected_z])
    z_embeds = z_embeds.view(1, -1, self.llm.config.hidden_size)  # flatten
    
    # 2. Query tokens
    q_tokens = self.tokenizer(query, return_tensors='pt')
    q_embeds = self.llm.get_input_embeddings()(q_tokens.input_ids)
    
    # 3. Prompt 구성: [z_1, z_2, ..., z_k] + [query tokens]
    combined = torch.cat([z_embeds, q_embeds], dim=1)
    
    # 4. Generate
    outputs = self.llm.generate(
        inputs_embeds=combined,
        max_new_tokens=100,
        do_sample=False
    )
    
    return self.tokenizer.decode(outputs[0])
```

## 4.3 Training Pipeline

### Stage 1: Write Phase 학습

```python
# 각 문서에 대해 z_i 학습
for doc_id, doc_text in corpus.items():
    z_i = write_phase.learn_vector(doc_id, doc_text)
    print(f"Doc {doc_id}: loss = {loss:.4f}")
```

**평가:**
- Reconstruction Loss: z_i → D_i 생성 시 perplexity
- Generation Quality: z_i만으로 문서 재생성 품질 (ROUGE, BLEU)

### Stage 2: Read Phase 학습

```python
# QA 데이터로 전체 시스템 fine-tuning
for question, answer, gold_doc_ids in qa_dataset:
    # Document selection
    selected_ids = read_phase.select_documents(question, k=5)
    
    # Answer generation
    predicted = read_phase.generate_answer(question, selected_ids)
    
    # Losses
    qa_loss = compute_loss(predicted, answer)
    retrieval_loss = compute_retrieval_loss(selected_ids, gold_doc_ids)
    
    total_loss = qa_loss + 0.1 * retrieval_loss
    total_loss.backward()
```

**평가:**
- QA Accuracy: EM, F1
- Retrieval Accuracy: Recall@K

---

# 5. 구현 계획

## 5.1 단계별 구현

### Step 1: Sanity Check (3일)

**목표:** Token Learning 자체가 작동하는지 확인

```python
# 최소 설정
model = "Qwen2.5-1.5B"  # 또는 Phi-3-mini
documents = 100개 (SQuAD context)
z_dim = hidden_size
m_tokens = 8
학습 = LLM freeze, z_i만 학습

# 확인 사항
1. z_i 학습 후 D_i 재생성 가능한가?
2. Perplexity가 수렴하는가?
3. 다른 문서와 혼동 없이 해당 문서만 생성하는가?
```

### Step 2: QA Integration (4일)

**목표:** z_i selection + answer generation 파이프라인 구축

```python
# 설정
documents = 1,000개
QA pairs = NQ dev set 1,000개

# 구현
1. Similarity-based selection 구현
2. Answer generation 구현
3. End-to-end evaluation

# 평가
- Selection Recall@5
- QA EM, F1
```

### Step 3: Scaling & Optimization (5일)

**목표:** 실제 규모로 확장 및 성능 최적화

```python
# 설정
model = "Llama-3.1-8B" + QLoRA
documents = 50,000개
QA = NQ full test set

# 최적화
1. z_dim ablation: {64, 128, 256, 512}
2. m_tokens ablation: {4, 8, 16, 32}
3. Selection method: similarity vs learned router
4. LLM training: freeze vs LoRA vs full FT

# 목표
- RAG와 comparable한 성능
- Latency 3-5x 개선
- Storage 50-100x 압축
```

## 5.2 코드 구조

```
parametric-qa/
├── configs/
│   ├── step1_sanity.yaml
│   ├── step2_qa.yaml
│   └── step3_scale.yaml
├── data/
│   ├── download.py          # NQ, HotpotQA 다운로드
│   ├── preprocess.py        # 전처리
│   └── dataloader.py        # DataLoader
├── models/
│   ├── write_phase.py       # z_i 학습
│   ├── read_phase.py        # Selection + Generation
│   ├── router.py            # Learned router
│   └── parametric_qa.py     # 통합 모델
├── training/
│   ├── train_write.py       # Stage 1
│   ├── train_read.py        # Stage 2
│   └── train_e2e.py         # End-to-end
├── evaluation/
│   ├── metrics.py           # EM, F1, ROUGE
│   ├── evaluate_write.py    # Reconstruction 평가
│   └── evaluate_qa.py       # QA 평가
├── baselines/
│   ├── standard_rag.py      # RAG baseline
│   ├── no_retrieval.py      # No retrieval
│   └── run_baselines.py
└── scripts/
    ├── run_step1.sh
    ├── run_step2.sh
    └── run_full.sh
```

---

# 6. 실험 설계

## 6.1 데이터셋

### Primary: Natural Questions (NQ)

| Split | Size | 특징 |
|-------|------|------|
| Train | 79,168 | Wikipedia 기반 |
| Dev | 8,757 | 개발/튜닝용 |
| Test | 3,610 | 최종 평가 |

**선택 이유:**
- Single-hop QA로 시작하기 적합
- 충분한 데이터 크기
- FlashRAG 등 표준 벤치마크

### Secondary: HotpotQA (Multi-hop)

| Split | Size | 특징 |
|-------|------|------|
| Train | 90,447 | 2-hop reasoning |
| Dev | 7,405 | Supporting facts 제공 |

**선택 이유:**
- Multi-hop reasoning 능력 평가
- Bridge entity 기반 추론 테스트

## 6.2 Baseline 선정

### 7개 Baseline (Foundational 3 + Competitive 4)

**Foundational (하한선):**

| # | Method | 설명 |
|---|--------|------|
| 1 | No Retrieval | LLM만 사용, 검색 없음 |
| 2 | BM25 + LLM | Sparse retrieval |
| 3 | Standard RAG | E5 + Llama (FlashRAG 표준) |

**Competitive (상한선):**

| # | Method | Venue | NQ EM |
|---|--------|-------|-------|
| 4 | Self-RAG | ICLR 2024 | ~35% |
| 5 | IRCoT | ACL 2023 | ~38% |
| 6 | Adaptive-RAG | NAACL 2024 | ~36% |
| 7 | CoRAG | NeurIPS 2025 | ~42% |

### Baseline 선정 근거

```
1. No Retrieval: 검색의 필요성 증명 (ablation)
2. BM25: 50년 역사의 sparse retrieval 기준점
3. Standard RAG: 2024년 기준 vanilla RAG
4. Self-RAG: Reflection token 기반 적응적 검색
5. IRCoT: CoT + iterative retrieval 표준
6. Adaptive-RAG: Query complexity routing
7. CoRAG: 2025년 1월 기준 SOTA
```

## 6.3 평가 지표

### Primary Metrics

| Metric | 정의 | 사용처 |
|--------|------|--------|
| **Exact Match (EM)** | 정답과 정확히 일치 | NQ, TriviaQA |
| **Token F1** | Token-level precision/recall | HotpotQA |

### Retrieval Metrics

| Metric | 정의 |
|--------|------|
| **Recall@K** | Top-K에 gold doc 포함 비율 |
| **Selection Precision** | 선택된 문서 중 relevant 비율 |

### Efficiency Metrics

| Metric | 측정 |
|--------|------|
| **Latency** | ms/query |
| **Memory** | GPU VRAM (GB) |
| **Storage** | MB/1K docs |

## 6.4 실험 단계

### Phase 1: Proof of Concept (3일)

```
목표: z_i learning 작동 확인
설정:
- Dataset: NQ dev 1,000 samples
- Documents: 5,000개
- Model: Llama-3.2-3B + QLoRA
- z_dim: 256, m_tokens: 8

실험:
1-1. Write Phase: z_i → D_i reconstruction
1-2. Read Phase: Query → z_i selection → Answer

Success Criteria:
- Reconstruction perplexity < 10
- Selection Recall@5 > 50%
```

### Phase 2: Ablation Studies (4일)

```
목표: 최적 설정 탐색

Ablation 1: z_dim
- z_dim ∈ {64, 128, 256, 512, 1024}

Ablation 2: m_tokens (z vector 길이)
- m_tokens ∈ {1, 4, 8, 16, 32}

Ablation 3: Selection 방법
- Cosine similarity
- Learned router
- Attention-based

Ablation 4: LLM training
- Freeze
- LoRA (rank 8, 32, 64)
- QLoRA
```

### Phase 3: Full Evaluation (5일)

```
목표: Baseline과 공정 비교

설정:
- Model: Llama-3.1-8B + QLoRA
- z_dim: Phase 2 최적값
- Documents: 50,000
- Datasets: NQ, TriviaQA, HotpotQA

비교:
- 동일 LLM backbone
- 동일 corpus
- 동일 test set
- 3 seeds, mean ± std 보고
```

### Phase 4: Analysis (3일)

```
목표: 깊이 있는 분석

분석 1: Error Analysis
- Selection 실패 case 분석
- Generation 실패 case 분석
- Multi-hop 실패 패턴

분석 2: z_i 해석
- t-SNE/UMAP 시각화
- 유사 문서 z_i clustering 확인
- z_i dimension별 역할 분석

분석 3: Scalability
- Document 수 증가에 따른 성능 변화
- z_dim 증가에 따른 storage-accuracy trade-off
```

---

# 7. 기대 결과 및 의의

## 7.1 예상 결과

### Best Case

| Metric | Target | 의미 |
|--------|--------|------|
| NQ EM | 33-35% | RAG와 동등 |
| HotpotQA F1 | 35-40% | Multi-hop도 가능 |
| Latency | 10-20ms | RAG 대비 5-10x 빠름 |
| Storage | 0.5MB/1K docs | 100x 압축 |

### Realistic Case

| Metric | Target | 의미 |
|--------|--------|------|
| NQ EM | 28-32% | RAG보다 5-10% 낮음 |
| HotpotQA F1 | 30-35% | Multi-hop 제한적 |
| Latency | 30-50ms | RAG 대비 3-5x 빠름 |
| Storage | 1MB/1K docs | 50x 압축 |

### Worst Case (실패 시나리오)

| 문제 | 원인 | 대응 |
|------|------|------|
| z_i 정보 손실 | 압축 한계 | z_dim 증가, m_tokens 증가 |
| Selection 실패 | Similarity 부정확 | Learned router 도입 |
| Multi-hop 불가 | 단일 z_i 한계 | Iterative selection |

## 7.2 연구 의의

### 학술적 기여

1. **새로운 패러다임 검증:** "RAG 없이 LLM 파라미터만으로 QA" 가능성 실증
2. **Compression-Accuracy Trade-off 분석:** 문서 압축의 한계와 가능성 정량화
3. **Selection Mechanism 연구:** z space에서의 효과적인 문서 선택 방법론

### 실용적 기여

1. **Latency 개선:** Retrieval 단계 제거로 실시간 응답 가능
2. **Storage 절약:** Vector DB 불필요, 컴팩트한 z만 저장
3. **Deployment 단순화:** LLM만 배포하면 됨

### 교수님 연구 비전과의 연결

```
Short-term Memory: z_i learning (LLM freeze, 빠른 저장)
     ↓
Long-term Memory: Full LLM fine-tuning (중요 정보 영구 저장)
     ↓
Unified Memory System: Parametric QA가 첫 걸음
```

## 7.3 한계 및 향후 연구

### 현재 연구의 한계

1. **Static Knowledge:** z_i 학습 후 업데이트 어려움 (retraining 필요)
2. **Scalability:** 문서 수 증가 시 selection 비용 증가
3. **Interpretability:** z_i가 무엇을 encode하는지 해석 어려움
4. **Multi-hop:** 복잡한 추론은 여전히 도전적

### 향후 연구 방향

| 방향 | 설명 |
|------|------|
| **Incremental Learning** | 새 문서 추가 시 z_i만 학습, LLM 고정 |
| **Encoder 기반** | f(D) = z로 즉시 z 생성, 학습 없이 |
| **Hierarchical z** | Document → Section → z 계층 구조 |
| **Memory Consolidation** | 자주 쓰이는 정보는 LLM weight로 통합 |

---

# 8. 일정 및 리소스

## 8.1 전체 일정 (3주)

```
Week 1 (Day 1-5): 환경 구축 + POC
├── Day 1-2: GCP 환경 설정, 데이터 준비
├── Day 3-4: Step 1 (Sanity Check) 실험
└── Day 5: 결과 분석, 방향 조정

Week 2 (Day 6-10): Ablation + 본실험
├── Day 6-7: Step 2 (QA Integration)
├── Day 8-9: Phase 2 (Ablation Studies)
└── Day 10: Phase 3 시작

Week 3 (Day 11-15): 본실험 + 분석
├── Day 11-12: Phase 3 (Full Evaluation) 완료
├── Day 13-14: Phase 4 (Analysis)
└── Day 15: 결과 정리, 보고서 작성
```

## 8.2 하드웨어 리소스

### GCP 설정

```bash
Machine: g2-standard-8
- vCPU: 8
- Memory: 32GB
- GPU: NVIDIA L4 (24GB VRAM)
- Disk: 200GB SSD

비용: ~$0.83/hour
예상 총 비용: $200-300 (240 GPU-hours)
```

### Memory Budget

```
Llama-3.2-3B (QLoRA 4bit): ~4GB
Llama-3.1-8B (QLoRA 4bit): ~8GB
Document vectors (50K × 256 × fp16): ~25MB
Query encoder: ~440MB
남은 headroom: ~10-14GB

→ Batch size 2, gradient accumulation 8
→ Effective batch: 16
```

## 8.3 소프트웨어 스택

```python
# Core
torch>=2.1.0
transformers>=4.36.0
peft>=0.7.0  # LoRA/QLoRA
bitsandbytes>=0.41.0  # 4-bit quantization

# Optimization
flash-attn>=2.3.0
accelerate>=0.24.0

# Data & Eval
datasets>=2.14.0
sentence-transformers>=2.2.0

# Baselines
flashrag>=0.1.0
```

---

# 9. 참고 문헌 및 관련 연구

## 9.1 핵심 참고 논문

### Parametric Memory

| 논문 | 학회 | 핵심 기여 |
|------|------|----------|
| Geva et al. "FFN as Key-Value Memory" | EMNLP 2021 | 이론적 기반 |
| ROME | NeurIPS 2022 | Causal tracing, rank-one edit |
| MEMIT | ICLR 2023 | 10,000+ 동시 편집 |
| MemoryLLM | ICML 2024 | Self-updating memory pool |
| M+ | ICML 2025 | Long-term memory extension |

### Token/Prompt Compression

| 논문 | 학회 | 핵심 기여 |
|------|------|----------|
| Gist Tokens | NeurIPS 2023 | 26x 압축 |
| ICAE | ICLR 2024 | 4x 압축, LoRA encoder |
| xRAG | NeurIPS 2024 | 178x 압축, retriever 재사용 |
| 500xCompressor | ACL 2025 | 480x 압축, KV values |

### RAG & Retrieval

| 논문 | 학회 | 핵심 기여 |
|------|------|----------|
| Self-RAG | ICLR 2024 | Reflection tokens |
| IRCoT | ACL 2023 | CoT + retrieval interleaving |
| CoRAG | NeurIPS 2025 | Retrieval chain, SOTA |
| HippoRAG | NeurIPS 2024 | Hippocampal indexing |

## 9.2 데이터셋

| 데이터셋 | 크기 | Hop | 용도 |
|----------|------|-----|------|
| Natural Questions | 79K | 1 | Primary |
| TriviaQA | 78K | 1 | Generalization |
| HotpotQA | 90K | 2 | Multi-hop |
| MuSiQue | 25K | 2-4 | Hard multi-hop |

## 9.3 평가 도구

| 도구 | 용도 |
|------|------|
| FlashRAG | RAG 벤치마크 통합 |
| RAGAS | Reference-free 평가 |
| LM-Eval-Harness | 표준 LLM 평가 |

---

# 부록: 핵심 요약

## A. 한 문장 요약

> **RAG의 retriever를 learnable document vector로 대체하여, LLM 파라미터만으로 문서 저장과 검색을 수행하는 Parametric QA 시스템 구축**

## B. 핵심 질문과 답변

| 질문 | 답변 |
|------|------|
| **왜 하는가?** | RAG의 latency, storage 문제 해결 |
| **뭘 하는가?** | 문서를 learnable vector z_i로 압축 |
| **어떻게 하는가?** | Write: z_i → D_i 학습 / Read: z_i 선택 → QA |
| **기대 효과?** | Latency 3-5x↓, Storage 50-100x↓ |
| **성공 기준?** | RAG 대비 EM 5% 이내, Latency 3x 이상 개선 |

## C. 교수님 보고 시 핵심 포인트

1. **방향 이해:** Token learning → RAG 대체 = 맞습니다
2. **단계적 접근:** Sanity check → QA → Scaling 순서로 진행
3. **첫 결과 예상:** 1주일 내 POC 결과 도출 가능
4. **리스크:** z_i 정보 손실 가능, 대안 준비됨 (z_dim 증가 등)

---

**문서 끝**

*이 보고서는 LLM as Memory 연구의 전체 맥락과 Parametric QA 실험 계획을 종합합니다.*
*질문이나 수정 사항이 있으시면 말씀해 주세요.*
