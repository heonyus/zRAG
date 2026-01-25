# Parametric QA 실험 설계서

## 핵심 아이디어

**기존 RAG**: Query → Retriever → Top-K Documents → LLM → Answer
**Parametric QA**: Query → Learned Document Vectors (z_i) Selection → LLM (finetuned) → Answer

문서를 텍스트로 검색하는 대신, 각 문서를 learnable vector로 압축하고 LLM이 직접 해당 벡터를 "읽어서" 답변 생성.

---

## 1. 실험 구성

### 1.1 Primary Experiment: Single-hop QA

| 항목 | 설정 |
|------|------|
| **Dataset** | Natural Questions (NQ) - FlashRAG 전처리 버전 |
| **Train/Dev/Test** | 79,168 / 8,757 / 3,610 |
| **Corpus** | Wikipedia 2018 (FlashRAG 표준) |
| **Document Pool** | 5,000 documents (POC) → 50,000 (확장) |
| **Model** | Llama-3.2-3B (POC) → Llama-3.1-8B (Full) |
| **GPU** | GCP g2-standard-8 (L4 24GB) |

### 1.2 Follow-up Experiment: Multi-hop QA

| 항목 | 설정 |
|------|------|
| **Dataset** | HotpotQA (distractor setting) |
| **Train/Dev/Test** | 90,447 / 7,405 / - |
| **특징** | 2-hop reasoning, supporting facts annotation |

---

## 2. Baseline 선정 (7개)

리서치 결과에 따라 **Foundational (3) + Competitive (4)** 구조:

### Foundational Baselines (하한선)

| # | Method | 설명 | 구현 |
|---|--------|------|------|
| 1 | **No Retrieval** | LLM만 사용, 검색 없음 | Llama-3.1-8B direct |
| 2 | **BM25 + LLM** | Sparse retrieval baseline | FlashRAG 내장 |
| 3 | **Standard RAG** | E5 + Llama-3.1-8B | FlashRAG 표준 설정 |

### Competitive Baselines (상한선)

| # | Method | Venue | HotpotQA F1 | 구현 |
|---|--------|-------|-------------|------|
| 4 | **Self-RAG** | ICLR 2024 | 29.6* | FlashRAG 내장 |
| 5 | **IRCoT** | ACL 2023 | 41.5* | FlashRAG 내장 |
| 6 | **Adaptive-RAG** | NAACL 2024 | 39.1* | FlashRAG 내장 |
| 7 | **CoRAG** | NeurIPS 2025 | 56.6 | HF checkpoint |

*FlashRAG 재현 결과 (Llama3-8B, e5-base-v2, top-5)

### Baseline 선정 근거

```
1. No Retrieval: 검색의 필요성 증명 (ablation)
2. BM25: 50년 역사의 sparse retrieval 기준점
3. Standard RAG: 2024년 기준 vanilla RAG
4. Self-RAG: Reflection token 기반 적응적 검색
5. IRCoT: CoT + iterative retrieval 표준
6. Adaptive-RAG: Query complexity routing
7. CoRAG: 2025년 1월 기준 SOTA (multi-hop)
```

---

## 3. Parametric QA 아키텍처

### 3.1 핵심 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    Parametric QA System                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [Write Phase - Offline]                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Doc_i    │───▶│ Encoder  │───▶│ z_i      │              │
│  │ (text)   │    │ (frozen) │    │ (learned)│              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                                              │
│  [Read Phase - Inference]                                    │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Query    │───▶│ Select   │───▶│ LLM      │───▶ Answer   │
│  │          │    │ z_i      │    │(finetuned)│              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                       │                                      │
│              Similarity or                                   │
│              Learned Routing                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 구현 상세

```python
class ParametricQA(nn.Module):
    def __init__(self, llm_name="meta-llama/Llama-3.2-3B", 
                 num_docs=5000, z_dim=256):
        super().__init__()
        
        # Document vectors (learnable)
        self.doc_vectors = nn.Parameter(
            torch.randn(num_docs, z_dim) * 0.02
        )
        
        # Query encoder for selection
        self.query_encoder = AutoModel.from_pretrained("intfloat/e5-base-v2")
        self.query_proj = nn.Linear(768, z_dim)
        
        # LLM with LoRA
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            load_in_4bit=True,  # QLoRA
            device_map="auto"
        )
        self.llm = get_peft_model(self.llm, lora_config)
        
        # z_i를 LLM input으로 변환
        self.z_to_embedding = nn.Linear(z_dim, self.llm.config.hidden_size)
    
    def select_documents(self, query, k=5):
        """Query와 가장 유사한 k개의 document vector 선택"""
        q_emb = self.query_encoder(query).last_hidden_state[:, 0]
        q_proj = self.query_proj(q_emb)  # [batch, z_dim]
        
        # Cosine similarity
        scores = F.cosine_similarity(
            q_proj.unsqueeze(1),  # [batch, 1, z_dim]
            self.doc_vectors.unsqueeze(0),  # [1, num_docs, z_dim]
            dim=-1
        )  # [batch, num_docs]
        
        top_k = torch.topk(scores, k, dim=-1)
        return top_k.indices, top_k.values
    
    def forward(self, query_ids, doc_indices):
        """선택된 z_i로 답변 생성"""
        # 선택된 document vectors
        selected_z = self.doc_vectors[doc_indices]  # [batch, k, z_dim]
        
        # z를 LLM embedding space로 변환
        z_embeds = self.z_to_embedding(selected_z)  # [batch, k, hidden]
        
        # Query embedding
        query_embeds = self.llm.get_input_embeddings()(query_ids)
        
        # Concatenate: [z_1, z_2, ..., z_k, query tokens]
        combined = torch.cat([z_embeds, query_embeds], dim=1)
        
        # Generate
        outputs = self.llm(inputs_embeds=combined, ...)
        return outputs
```

### 3.3 Training Protocol

**Stage 1: Document Vector Learning (Write Phase)**
```python
# 각 document에 대해 z_i 최적화
# Loss: LLM이 z_i만으로 document content 생성할 수 있도록

for doc_id, doc_text in enumerate(corpus):
    z_i = model.doc_vectors[doc_id]
    z_embed = model.z_to_embedding(z_i)
    
    # Teacher forcing: z_i → doc_text
    loss = -log P(doc_text | z_embed)
    loss.backward()
    
    # z_i만 업데이트 (LLM frozen)
    optimizer.step()  # Only updates doc_vectors
```

**Stage 2: QA Fine-tuning (Read Phase)**
```python
# Question-Answer pair로 전체 시스템 학습

for question, answer, gold_doc_ids in qa_dataset:
    # Document selection
    selected_ids, scores = model.select_documents(question, k=5)
    
    # Answer generation
    output = model(question, selected_ids)
    
    # Losses:
    # 1. Answer generation loss
    qa_loss = cross_entropy(output, answer)
    
    # 2. Retrieval loss (gold docs should be selected)
    retrieval_loss = -log(scores[gold_doc_ids])
    
    total_loss = qa_loss + 0.1 * retrieval_loss
    total_loss.backward()
```

---

## 4. 평가 지표

### 4.1 Answer Quality (Primary)

| Metric | 정의 | 사용처 |
|--------|------|--------|
| **Exact Match (EM)** | 정답과 정확히 일치 비율 | NQ, TriviaQA |
| **Token F1** | Token-level precision/recall | HotpotQA, 2Wiki |
| **Answer Accuracy** | LLM judge (GPT-4) | Open-ended |

### 4.2 Retrieval Quality (Document Selection)

| Metric | 정의 |
|--------|------|
| **Recall@K** | Top-K 선택에 gold doc 포함 비율 |
| **MRR** | Gold doc의 평균 역순위 |
| **Selection Precision** | 선택된 문서 중 relevant 비율 |

### 4.3 Efficiency

| Metric | 측정 |
|--------|------|
| **Latency** | ms/query (inference time) |
| **Memory** | GB (GPU VRAM) |
| **Storage** | MB/1K docs (z_i 저장 크기) |
| **FLOPs** | 연산량 비교 |

### 4.4 RAGAS (Reference-free)

| Metric | 목적 |
|--------|------|
| **Faithfulness** | z_i 정보와 답변 일관성 |
| **Answer Relevancy** | 질문-답변 관련성 |

---

## 5. 실험 단계

### Phase 1: Proof of Concept (3일)

```
목표: 아이디어 작동 확인
- Dataset: NQ dev set 1,000 samples
- Documents: 5,000개 (관련 문서만)
- Model: Llama-3.2-3B + QLoRA
- z_dim: 256

실험:
1-1. z_i learning만 (Stage 1)
     - z_i가 문서 정보를 encode하는지 확인
     - Metric: Reconstruction perplexity
     
1-2. QA fine-tuning (Stage 2)
     - z_i selection + answer generation
     - Metric: EM, F1

비교:
- Parametric QA vs Standard RAG vs No Retrieval
```

### Phase 2: Ablation Studies (4일)

```
목표: 최적 설정 탐색

Ablation 1: z_dim 영향
- z_dim ∈ {64, 128, 256, 512, 1024}
- 측정: EM, storage, latency

Ablation 2: Document pool 크기
- num_docs ∈ {1K, 5K, 10K, 50K}
- 측정: EM, selection accuracy, memory

Ablation 3: Selection 방법
- Cosine similarity (baseline)
- Learned router (MLP)
- Attention-based selection

Ablation 4: k (선택 문서 수)
- k ∈ {1, 3, 5, 10}
- Trade-off: context length vs accuracy
```

### Phase 3: Full Evaluation (5일)

```
목표: Baseline과 공정 비교

설정:
- Model: Llama-3.1-8B + QLoRA
- z_dim: Phase 2 최적값
- Documents: 50,000
- Datasets: NQ, TriviaQA, HotpotQA

비교 대상 (7 baselines):
1. No Retrieval
2. BM25 + LLM
3. Standard RAG (E5 + Llama)
4. Self-RAG
5. IRCoT
6. Adaptive-RAG
7. CoRAG

평가:
- 동일 LLM backbone 사용
- 동일 corpus
- 동일 test set
- Multiple runs (3 seeds)
```

### Phase 4: Multi-hop Extension (3일)

```
목표: Multi-hop QA 성능 확인

Dataset: HotpotQA distractor

Challenge:
- 2개 문서가 필요 (bridge reasoning)
- z_i selection이 연결된 문서를 함께 선택해야 함

방법:
- Multiple z_i concatenation: [z_1, z_2, ..., z_k]
- Iterative selection: z_1 → answer_partial → z_2
- Graph-based linking (write-time)

평가:
- EM, F1
- Supporting Fact F1
- Bridge entity recall
```

---

## 6. 하드웨어 & 구현

### 6.1 GCP 설정

```bash
# Machine type
g2-standard-8
- vCPU: 8
- Memory: 32GB
- GPU: NVIDIA L4 (24GB VRAM)
- Disk: 200GB SSD

# 예상 비용
- $0.83/hour (on-demand)
- ~$200 for 240 GPU-hours
```

### 6.2 Memory Budget (L4 24GB)

```
Llama-3.2-3B (QLoRA 4bit): ~4GB
Llama-3.1-8B (QLoRA 4bit): ~8GB

Document vectors (50K × 256 × fp16): ~25MB
Query encoder (E5-base): ~440MB
z_to_embedding layer: ~8MB

Training headroom: ~10-14GB
→ Batch size 2, gradient accumulation 8
→ Effective batch: 16
```

### 6.3 Software Stack

```python
# requirements.txt
torch>=2.1.0
transformers>=4.36.0
peft>=0.7.0  # LoRA/QLoRA
bitsandbytes>=0.41.0  # 4-bit quantization
flash-attn>=2.3.0  # Flash Attention
sentence-transformers>=2.2.0
faiss-gpu>=1.7.0
datasets>=2.14.0
accelerate>=0.24.0

# FlashRAG for baselines
flashrag>=0.1.0
```

### 6.4 Training Configuration

```python
# QLoRA Config
lora_config = LoraConfig(
    r=64,  # rank
    lora_alpha=128,  # scaling
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training Args
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    warmup_ratio=0.1,
    fp16=True,
    gradient_checkpointing=True,
    logging_steps=50,
    save_strategy="epoch",
)
```

---

## 7. 예상 결과 & 리스크

### 7.1 Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| NQ EM | ≥30% | Standard RAG 35%, 약간 낮아도 OK |
| HotpotQA F1 | ≥35% | Standard RAG 35%, 비슷하면 성공 |
| Latency | <50ms | RAG 대비 3-5x 빠름 |
| Storage | <1MB/1K docs | z_i compression 효과 |

### 7.2 Expected Results

**Best Case:**
- Single-hop: RAG와 competitive (EM 33-35%)
- Multi-hop: 약간 낮음 (F1 30-35%)
- Latency: 5-10x faster (retriever 제거)
- Storage: 100x smaller (text → z_i)

**Realistic Case:**
- Single-hop: RAG보다 5-8% 낮음 (EM 28-32%)
- Multi-hop: 10-15% 낮음 (F1 25-30%)
- Latency: 3-5x faster
- Storage: 50-100x smaller

**Worst Case:**
- z_i가 충분한 정보 encode 실패
- Selection accuracy 낮음 (wrong documents)
- Multi-hop 완전 실패

### 7.3 Risk Mitigation

| Risk | Mitigation |
|------|------------|
| z_i 정보 손실 | z_dim 증가, reconstruction loss 모니터링 |
| Selection 실패 | Learned router, hard negative mining |
| Multi-hop 실패 | Iterative selection, write-time linking |
| Memory overflow | Gradient checkpointing, smaller batch |

---

## 8. Timeline

```
Week 1 (Day 1-5):
├── Day 1-2: 환경 구축, 데이터 준비
├── Day 3-4: Phase 1 (POC)
└── Day 5: 결과 분석, 디버깅

Week 2 (Day 6-10):
├── Day 6-8: Phase 2 (Ablations)
├── Day 9-10: Phase 3 시작
└── 중간 결과 정리

Week 3 (Day 11-15):
├── Day 11-12: Phase 3 완료
├── Day 13-14: Phase 4 (Multi-hop)
└── Day 15: 최종 결과 분석, 보고서
```

---

## 9. 핵심 연구 질문

1. **RQ1**: Per-document learnable vector가 문서 정보를 효과적으로 압축할 수 있는가?
   - Metric: Reconstruction perplexity, QA EM

2. **RQ2**: z_i 기반 selection이 traditional retrieval과 comparable한가?
   - Metric: Recall@K, Selection precision

3. **RQ3**: Parametric QA가 latency-accuracy trade-off에서 우위가 있는가?
   - Metric: EM vs Latency curve

4. **RQ4**: Multi-hop reasoning에서 z_i 접근법의 한계는 무엇인가?
   - Metric: HotpotQA F1, Bridge entity recall

---

## 10. Deliverables

1. **코드**: GitHub repo with reproducible experiments
2. **모델**: Trained checkpoints (z_i vectors, LoRA weights)
3. **결과**: Tables and figures comparing all methods
4. **분석**: Error analysis, ablation insights
5. **논문 초안**: Problem → Method → Experiments → Analysis
