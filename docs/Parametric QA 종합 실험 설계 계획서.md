# Parametric QA 종합 실험 설계 계획서

**연구자:** Honey
**작성일:** 2026년 1월 25일
**버전:** 1.0 (종합)

---

## 목차

1. [연구 개요 및 동기](#1-연구-개요-및-동기)
2. [선행연구 포지셔닝](#2-선행연구-포지셔닝)
3. [시스템 아키텍처](#3-시스템-아키텍처)
4. [데이터셋 및 코퍼스](#4-데이터셋-및-코퍼스)
5. [Baseline 선정](#5-baseline-선정)
6. [평가 체계](#6-평가-체계)
7. [실험 단계 (4 Phases)](#7-실험-단계-4-phases)
8. [구현 상세](#8-구현-상세)
9. [하드웨어 및 소프트웨어 스택](#9-하드웨어-및-소프트웨어-스택)
10. [예상 결과 및 리스크 대응](#10-예상-결과-및-리스크-대응)
11. [후속 연구 로드맵](#11-후속-연구-로드맵)
12. [연구 질문 (RQs) 및 Deliverables](#12-연구-질문-rqs-및-deliverables)

---

# 1. 연구 개요 및 동기

## 1.1 핵심 문제의식

기존 RAG 시스템의 구조적 한계:

| 문제 | 설명 |
|------|------|
| **Latency** | 매 쿼리마다 retrieval 필요 (수십~수백 ms) |
| **Storage** | 전체 문서 + 임베딩 저장 필요 (수십 GB) |
| **Retrieval Error** | 검색 실패 → 답변 품질 저하 (error propagation) |
| **System Complexity** | Retriever + Vector DB + Reranker 다단계 구조 → 오류 추적 어려움 |
| **Integration** | LLM과 retriever 분리 → end-to-end 최적화 불가 |

## 1.2 제안: Parametric QA

> **핵심 아이디어:** Vector DB + Retriever를 "로컬 LLM이 학습한 per-document learnable vector (z_i)"로 대체하여, LLM 파라미터만으로 문서 저장과 검색을 수행한다.

```
[기존 RAG]
Document → Chunking → Embedding → Vector DB → Retriever → Top-K → LLM → Answer

[Parametric QA (제안)]
Document → Learnable Vector z_i (Write Phase)
Query → z_i Selection → LLM(z_i, Query) → Answer (Read Phase)
```

## 1.3 교수님 연구 비전과의 정합

| 구분 | 내용 | 비유 |
|------|------|------|
| **Short-term Memory** | z_i learning (LLM freeze, 빠른 저장) | 워킹 메모리 |
| **Long-term Memory** | Full LLM fine-tuning (중요 정보 영구 저장) | 장기 기억 |
| **Unified System** | Parametric QA = 첫 걸음 | 통합 메모리 OS |

## 1.4 연구 방향 진화 히스토리

```
Phase 1: M+ (ICML 2025) 분석 기반
├── Write-time Retrieval (새 정보 저장 시 기존 메모리와 연결)
└── Iterative Retrieval with Learned Termination

    ↓ 교수님 피드백: "더 근본적인 걸 해봐. Token Learning으로 문서를 압축해"

Phase 2: Token Learning 개념 도입
├── Random z → optimize → z가 넣으면 D가 생성되도록
└── MetaQueries 참고: learnable vector가 모델을 조종하는 인터페이스

    ↓ 교수님 피드백: "Token만 학습은 sanity check. LLM 전체 + z 함께 학습이 진짜 목표"

Phase 3 (최종): Parametric QA
├── Write: max log P(D_i | z_i; θ) - 문서를 z에 압축
├── Read: max log P(answer | q, z_selected; θ) - z로 QA
└── 목표: RAG 완전 대체 (Latency 3-5x↓, Storage 50-100x↓)
```

---

# 2. 선행연구 포지셔닝

## 2.1 SOTA RAG 패러다임 (2024-2025)

| 패러다임 | 대표 연구 | 핵심 혁신 | 성능 |
|----------|-----------|-----------|------|
| **Reasoning-integrated** | CoRAG (NeurIPS 2025) | Rejection sampling for retrieval chains | HotpotQA +10pt EM |
| **RL-based** | Search-R1, R1-Searcher | Pure RL with outcome-based rewards | +41~48% vs baseline |
| **Reflection-based** | Self-RAG (ICLR 2024) | Reflection tokens for adaptive retrieval | Beats ChatGPT |
| **Graph-based** | HippoRAG 2 | Hippocampal indexing + PPR | 96.3 Recall@5, 10-30x cheaper |
| **Parametric** | xRAG (NeurIPS 2024) | 178x compression, modality bridge | RAG-competitive |

## 2.2 가장 가까운 관련 연구

| 계열 | 연구 | 우리와의 차이 |
|------|------|---------------|
| **Soft Prompt** | Prompt Tuning, P-tuning v2 | 태스크 단위 제어 ≠ 문서별 메모리 슬롯 |
| **Generative Retrieval** | DSI, DSI++ | doc ID 생성 ≠ evidence 직접 생성 |
| **Document Compression** | xRAG, ICAE, Gist Tokens | 압축 후에도 retriever 필요 ≠ retriever 제거 |
| **Parametric RAG** | DyPRAG (ICLR 2025) | 문서→파라미터 translator ≈ 가장 유사 |
| **Memory Pool** | MemoryLLM (ICML 2024), M+ | ~1B 파라미터 메모리 풀, cross-attention |

## 2.3 우리 연구의 차별점

```
1. "z_i + query → evidence/answer"를 정면 타깃으로 설정
2. Write/Read Phase 명시적 분리 + end-to-end 최적화
3. 새 문서에 대한 빠른 z initialization (few-step adaptation)
4. RAG 표준 벤치마크(FlashRAG/MIRAGE)에서 정면 비교
5. Efficiency-accuracy trade-off의 정량적 분석
```

## 2.4 이론적 기반

- **Geva et al. (EMNLP 2021):** FFN 레이어가 Key-Value 메모리로 작동
- **MetaQueries (2025):** Learnable queries가 frozen 모델의 인터페이스 역할
- **Token Compression 계보:** Gist (26x) → ICAE (4-16x) → xRAG (178x) → 500xCompressor (480x)

---

# 3. 시스템 아키텍처

## 3.1 전체 구조

```
┌──────────────────────────────────────────────────────────────────┐
│                       Parametric QA System                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ═══════════════════════════════════════════════════════════════   │
│  ║                 WRITE PHASE (Offline)                      ║   │
│  ═══════════════════════════════════════════════════════════════   │
│                                                                    │
│    Document D_i                                                    │
│         │                                                          │
│         ▼                                                          │
│    ┌──────────────┐                                                │
│    │  Initialize  │  z_i = random(m_tokens, hidden_size) * 0.02    │
│    │     z_i      │  또는 z_i = encoder(D_i)                       │
│    └──────────────┘                                                │
│         │                                                          │
│         ▼                                                          │
│    ┌──────────────┐                                                │
│    │   Optimize   │  min -log P(D_i | z_i; θ_LLM)                 │
│    │     z_i      │  Stage 1: θ frozen, z만 학습                   │
│    └──────────────┘  Stage 2: θ + z 함께 학습 (QLoRA)             │
│         │                                                          │
│         ▼                                                          │
│    ┌──────────────┐                                                │
│    │    Store     │  Memory Pool: {z_1, z_2, ..., z_N}             │
│    │     z_i      │  Storage: N × m_tokens × hidden_size × fp16   │
│    └──────────────┘                                                │
│                                                                    │
│  ═══════════════════════════════════════════════════════════════   │
│  ║                 READ PHASE (Inference)                     ║   │
│  ═══════════════════════════════════════════════════════════════   │
│                                                                    │
│    Query q                                                         │
│         │                                                          │
│         ▼                                                          │
│    ┌──────────────┐                                                │
│    │   Select     │  방법 1: cosine_similarity(q_embed, z_mean)    │
│    │    z_i       │  방법 2: Learned Router (MLP)                  │
│    └──────────────┘  방법 3: Attention-based Selection             │
│         │                                                          │
│         ▼                                                          │
│    ┌──────────────┐                                                │
│    │     LLM      │  Input: [z_1, z_2, ..., z_k | query tokens]   │
│    │   Generate   │  Output: answer tokens                         │
│    └──────────────┘                                                │
│         │                                                          │
│         ▼                                                          │
│      Answer                                                        │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

## 3.2 Training Pipeline (2-Stage)

### Stage 1: Write Phase (Document Vector Learning)

```
목표: max log P(D_i | z_i; θ)   [θ frozen]
입력: z_i (learnable) prepended to document tokens
출력: document reconstruction
손실: Cross-entropy (teacher forcing)
학습 대상: z_i만 업데이트
```

### Stage 2: Read Phase (QA Fine-tuning)

```
목표: max log P(answer | q, z_selected; θ)   [θ + z_i 함께 학습]
입력: [z_1, z_2, ..., z_k] + [query tokens]
출력: answer tokens
손실: L_total = L_qa + α × L_retrieval   (α = 0.1)
      L_qa = -log P(answer | z_selected, q)
      L_retrieval = -log P(gold_z in top-k)
학습 대상: LLM (QLoRA) + z_i + query_proj + z_to_embedding
```

## 3.3 Selection Mechanism 비교

| 방법 | 복잡도 | 학습 가능 | 적합 상황 |
|------|--------|-----------|-----------|
| **Cosine Similarity** | O(N) | ❌ | POC, 작은 코퍼스 |
| **Learned Router (MLP)** | O(N) | ✅ | 중간 규모, 정확도 우선 |
| **Attention-based** | O(N × d) | ✅ | 대규모, 복잡한 쿼리 |
| **FAISS + z_mean** | O(log N) | ❌ | 대규모 코퍼스 (50K+) |

---

# 4. 데이터셋 및 코퍼스

## 4.1 Primary: Natural Questions (NQ)

| Split | Size | 특징 |
|-------|------|------|
| Train | 79,168 | Wikipedia 기반, single-hop |
| Dev | 8,757 | 하이퍼파라미터 튜닝 |
| Test | 3,610 | 최종 평가 |

**선택 근거:**
- Single-hop QA로 개념 증명에 적합
- FlashRAG 표준 벤치마크 (E5-base-v2, top-5 통일)
- 충분한 데이터 규모

## 4.2 Secondary: HotpotQA (Multi-hop)

| Split | Size | 특징 |
|-------|------|------|
| Train | 90,447 | 2-hop reasoning |
| Dev | 7,405 | Supporting facts annotation |

**선택 근거:**
- Multi-hop reasoning 능력 평가
- Bridge entity 기반 추론 테스트
- Distractor setting (8 noise + 2 gold)

## 4.3 Optional Extension

| 데이터셋 | Hop | Size | 난이도 |
|----------|-----|------|--------|
| TriviaQA | 1 | 78K | 중 (generalization) |
| MuSiQue | 2-4 | 25K | 상 (hard multi-hop) |
| Bamboogle | 2 | 125 | 최상 (no shortcuts) |

## 4.4 코퍼스 구성

```
Standard: Wikipedia December 2018 dump (~22M passages, 100-word chunks)
FlashRAG 표준: E5-base-v2 pre-built index 사용

실험별 코퍼스 크기:
├── Phase 1 (POC): 5,000 documents (관련 문서만)
├── Phase 2 (Ablation): 10,000 documents
├── Phase 3 (Full): 50,000 documents
└── Phase 4 (Multi-hop): HotpotQA distractor setting
```

---

# 5. Baseline 선정

## 5.1 Baseline 구조 (7개 = Foundational 3 + Competitive 4)

### Foundational Baselines (하한선 설정)

| # | Method | 설명 | 구현 방법 |
|---|--------|------|-----------|
| 1 | **No Retrieval** | LLM만 사용 (parametric knowledge) | Llama-3.1-8B direct |
| 2 | **BM25 + LLM** | Sparse retrieval, 50년 역사의 기준점 | FlashRAG 내장 |
| 3 | **Standard RAG** | Dense retrieval (E5-base-v2) + Llama | FlashRAG 표준 설정 |

### Competitive Baselines (상한선 설정)

| # | Method | Venue | 핵심 혁신 | 구현 방법 |
|---|--------|-------|-----------|-----------|
| 4 | **Self-RAG** | ICLR 2024 | Reflection tokens for adaptive retrieval | FlashRAG 내장 |
| 5 | **IRCoT** | ACL 2023 | CoT + iterative retrieval interleaving | FlashRAG 내장 |
| 6 | **Adaptive-RAG** | NAACL 2024 | Query complexity routing | FlashRAG 내장 |
| 7 | **CoRAG** | NeurIPS 2025 | Retrieval chain, rejection sampling | HF checkpoint |

### 선정 근거 (4원칙)

```
1. Retriever type diversity: Sparse (BM25) + Dense (E5) + Hybrid (learned)
2. Ablation hierarchy: No-RAG → Vanilla → Advanced → SOTA
3. Architectural diversity: Retriever-based / Generator-based / Hybrid
4. Recency: 2023 (IRCoT) ~ 2025 (CoRAG) 포괄
```

## 5.2 공정 비교 조건

모든 baseline과 Parametric QA의 공정 비교를 위한 통제 변인:

| 변인 | 통일 설정 |
|------|-----------|
| LLM Backbone | Llama-3.1-8B-Instruct |
| Corpus | Wikipedia 2018, 동일 passage set |
| Test Set | NQ test 3,610 / HotpotQA dev 7,405 |
| Input Length | 2048 tokens (max) |
| Random Seeds | 3 seeds, mean ± std 보고 |
| Retrieval Count | k=5 (FlashRAG 표준) |

---

# 6. 평가 체계

## 6.1 Answer Quality (Primary)

| Metric | 정의 | 사용처 |
|--------|------|--------|
| **Exact Match (EM)** | 정규화 후 정답과 정확히 일치 비율 | NQ, TriviaQA |
| **Token F1** | Token-level precision/recall harmonic mean | HotpotQA, 긴 답변 |
| **LLM Judge Accuracy** | GPT-4 기반 정답 판정 | Open-ended 평가 |

## 6.2 Retrieval/Selection Quality

| Metric | 정의 | Parametric QA에서의 의미 |
|--------|------|--------------------------|
| **Recall@K** | Top-K에 gold doc 포함 비율 | z selection이 gold z를 포함하는가 |
| **MRR** | Gold doc의 평균 역순위 | z selection 순위 품질 |
| **Selection Precision** | 선택된 중 relevant 비율 | 불필요한 z 선택 비율 |

## 6.3 Efficiency Metrics

| Metric | 측정 | 비교 포인트 |
|--------|------|-------------|
| **Latency** | ms/query (inference) | RAG의 retrieval 시간 vs z selection 시간 |
| **Memory** | GPU VRAM (GB) | z 테이블 vs Vector DB |
| **Storage** | MB/1K docs | z_i 크기 vs passage text + embedding |
| **FLOPs** | 연산량 | xRAG 방식으로 비교 |

## 6.4 RAGAS (Reference-free RAG 평가)

| Metric | 정의 | 계산 방법 |
|--------|------|-----------|
| **Faithfulness** | 답변의 사실적 일관성 | Claim 추출 + entailment 검증 |
| **Answer Relevancy** | 답변-질문 관련성 | Reverse question generation similarity |
| **Context Precision** | 관련 항목 순위 품질 | Position-weighted relevance scoring |
| **Context Recall** | Ground-truth 정보 커버리지 | Sentence-level attribution |

## 6.5 Robustness Metrics (MIRAGE, 선택적)

| Metric | 약어 | 의미 |
|--------|------|------|
| **Noise Vulnerability** | NV | 노이즈 문서에 대한 취약성 |
| **Context Acceptability** | CA | 주어진 컨텍스트 수용도 |
| **Context Insensitivity** | CI | 컨텍스트 변화에 대한 둔감성 |
| **Context Misinterpretation** | CM | 컨텍스트 오독 비율 |

**z-RAG의 기대 우위:** 텍스트를 그대로 넣지 않고 압축된 z를 통해 읽으므로 노이즈 민감도가 다를 가능성 → NV/CM에서 Standard RAG 대비 개선 기대

## 6.6 Multi-hop 전용 지표 (Phase 4)

| Metric | 정의 |
|--------|------|
| **Supporting Fact F1** | Supporting sentence 예측 정확도 |
| **Joint EM/F1** | Answer + Supporting fact 결합 정확도 |
| **Bridge Entity Recall** | 중간 entity 식별 비율 |
| **Hop-wise Recall** | 각 reasoning step에서의 검색 품질 |

## 6.7 Write Phase 전용 지표 (Stage 1 검증)

| Metric | 정의 | 성공 기준 |
|--------|------|-----------|
| **Reconstruction Perplexity** | z_i → D_i 생성 시 perplexity | < 10 |
| **ROUGE-L** | z_i만으로 문서 재생성 품질 | > 0.5 |
| **Compression Ratio** | |D_i| / |z_i| | > 50x |

## 6.8 통계 보고 기준

```
- Multiple runs: 3 seeds (42, 123, 456)
- 보고: mean ± std
- 유의성: Bootstrap confidence interval (필요시)
- 상대 개선율: "+X.X% vs Standard RAG" 형태
```

---

# 7. 실험 단계 (4 Phases)

## Phase 1: Proof of Concept (3일)

### 목표
z_i learning이 작동하는지 확인 (아이디어 검증)

### 설정

| 항목 | 값 |
|------|-----|
| Dataset | NQ dev 1,000 samples |
| Documents | 5,000개 (관련 문서만) |
| Model | Llama-3.2-3B + QLoRA |
| z_dim | 256 (= hidden_size) |
| m_tokens | 8 |
| LLM Training | Frozen (Stage 1만) |

### 실험 내용

```
1-1. Write Phase 검증
     - z_i 학습 후 D_i 재생성 가능한가?
     - Perplexity가 수렴하는가?
     - 다른 문서와 혼동 없이 해당 문서만 생성하는가?
     - Metric: Reconstruction perplexity, ROUGE-L

1-2. Read Phase 검증
     - z_i selection + answer generation 파이프라인 작동
     - Metric: Selection Recall@5, QA EM/F1

비교: Parametric QA vs Standard RAG vs No Retrieval (3개만)
```

### Success Criteria

| 지표 | 기준 | 의미 |
|------|------|------|
| Reconstruction Perplexity | < 10 | z_i가 문서 정보 encode |
| Selection Recall@5 | > 50% | 관련 z_i 찾기 가능 |
| QA EM (NQ) | > 15% | 기본 QA 능력 확인 |

### 실패 시 대응

| 실패 패턴 | 원인 가설 | 대응 |
|-----------|-----------|------|
| Perplexity 미수렴 | z_dim 부족 | z_dim 512, m_tokens 16으로 증가 |
| 문서 혼동 | z_i 공간 겹침 | orthogonal initialization |
| Selection 실패 | similarity 부정확 | E5 encoder 기반 projection |

---

## Phase 2: Ablation Studies (4일)

### 목표
최적 하이퍼파라미터 및 구성 요소 탐색

### Ablation 1: z_dim 영향

| z_dim | 예상 Storage (50K docs) | 비고 |
|-------|-------------------------|------|
| 64 | ~6MB | 최소, 정보 손실 위험 |
| 128 | ~12MB | |
| 256 | ~25MB | 기본값 |
| 512 | ~50MB | |
| 1024 | ~100MB | 최대, 과적합 위험 |

**측정:** EM, Reconstruction PPL, Storage, Latency

### Ablation 2: m_tokens (z vector 토큰 수)

| m_tokens | Input overhead | 비고 |
|----------|---------------|------|
| 1 | 최소 | 극한 압축 |
| 4 | 낮음 | |
| 8 | 중간 | 기본값 |
| 16 | 높음 | 정보 보존 우수 |
| 32 | 매우 높음 | context budget 압박 |

**측정:** EM, ROUGE-L, Context window 잔여량

### Ablation 3: Selection 방법

| 방법 | 구현 복잡도 | 학습 가능 |
|------|------------|-----------|
| Cosine similarity (z_mean vs q_embed) | 낮음 | ❌ |
| Learned Router (MLP) | 중간 | ✅ |
| Attention-based selection | 높음 | ✅ |

**측정:** Recall@K, Selection Precision, Latency

### Ablation 4: LLM Training 방식

| 방법 | 학습 파라미터 | VRAM | 비고 |
|------|-------------|------|------|
| Freeze | z_i만 | ~5GB | Sanity check |
| LoRA (r=8) | z_i + LoRA | ~8GB | 경량 |
| LoRA (r=32) | z_i + LoRA | ~10GB | |
| QLoRA (r=64) | z_i + QLoRA | ~8GB | 추천 |

**측정:** EM, Training speed, VRAM usage

### Ablation 5: k (선택 문서 수)

| k | Context overhead (m=8) | Trade-off |
|---|------------------------|-----------|
| 1 | 8 tokens | 정보 부족 위험 |
| 3 | 24 tokens | |
| 5 | 40 tokens | 기본 (FlashRAG 표준) |
| 10 | 80 tokens | context 효율적 활용 |

---

## Phase 3: Full Evaluation (5일)

### 목표
Phase 2 최적 설정으로 7개 Baseline과 공정 비교

### 설정

| 항목 | 값 |
|------|-----|
| Model | Llama-3.1-8B + QLoRA (r=64) |
| z_dim | Phase 2 최적값 |
| m_tokens | Phase 2 최적값 |
| Selection | Phase 2 최적 방법 |
| Documents | 50,000 |
| Dataset | NQ full test (3,610) |
| Seeds | 3 (mean ± std 보고) |

### 결과 테이블 형식

```
Table 1: Main Results on Natural Questions (NQ)

| Method          | EM (↑)      | F1 (↑)      | Recall@5 (↑) | Latency (ms, ↓) | Storage (MB/1K, ↓) |
|-----------------|-------------|-------------|--------------|-----------------|---------------------|
| No Retrieval    | xx.x ± x.x | xx.x ± x.x | -            | xx.x            | 0                   |
| BM25 + LLM     | xx.x ± x.x | xx.x ± x.x | xx.x         | xx.x            | xx.x                |
| Standard RAG    | xx.x ± x.x | xx.x ± x.x | xx.x         | xx.x            | xx.x                |
| Self-RAG        | xx.x ± x.x | xx.x ± x.x | xx.x         | xx.x            | xx.x                |
| IRCoT           | xx.x ± x.x | xx.x ± x.x | xx.x         | xx.x            | xx.x                |
| Adaptive-RAG    | xx.x ± x.x | xx.x ± x.x | xx.x         | xx.x            | xx.x                |
| CoRAG           | xx.x ± x.x | xx.x ± x.x | xx.x         | xx.x            | xx.x                |
| **Ours (PQA)**  | xx.x ± x.x | xx.x ± x.x | xx.x         | xx.x            | xx.x                |
```

### 추가 분석

```
Table 2: Efficiency Comparison

| Method       | Latency (ms) | Memory (GB) | Storage (MB/1K) | FLOPs |
|--------------|--------------|-------------|-----------------|-------|
| Standard RAG | ~100-200     | ~14         | ~500            | 1.0x  |
| Ours (PQA)   | ~20-50       | ~8-10       | ~1-5            | ~0.3x |

Table 3: RAGAS Scores (Reference-free)

| Method       | Faithfulness | Answer Rel. | Context Prec. | Context Recall |
|--------------|-------------|-------------|---------------|----------------|
| Standard RAG | xx.x        | xx.x        | xx.x          | xx.x           |
| Ours (PQA)   | xx.x        | xx.x        | xx.x          | xx.x           |
```

---

## Phase 4: Multi-hop Extension (3일)

### 목표
Multi-hop QA에서의 Parametric QA 성능 및 한계 분석

### Challenge

```
Multi-hop에서의 핵심 문제:
- 2개 이상의 문서가 필요 (bridge reasoning)
- z_i selection이 연결된 문서를 함께 선택해야 함
- 중간 추론 결과를 다음 selection에 반영해야 함
```

### 접근 방법 (3가지)

| 방법 | 설명 | 복잡도 |
|------|------|--------|
| **Multiple z_i concatenation** | [z_1, z_2, ..., z_k] 한 번에 선택 | 낮음 |
| **Iterative selection** | z_1 → partial answer → z_2 선택 | 중간 |
| **Write-time linking** | z_i 학습 시 관련 z_j와 연결 | 높음 |

### 평가

```
Table 4: Multi-hop Results on HotpotQA

| Method       | EM    | F1    | Sup. Fact F1 | Bridge Recall |
|--------------|-------|-------|-------------|---------------|
| Standard RAG | xx.x  | xx.x  | xx.x        | xx.x          |
| IRCoT        | xx.x  | xx.x  | xx.x        | xx.x          |
| CoRAG        | xx.x  | xx.x  | xx.x        | xx.x          |
| Ours (PQA)   | xx.x  | xx.x  | xx.x        | xx.x          |
```

### Error Analysis 계획

```
Bridge Entity Analysis (HotpotQA):
1. Bridge entity not found → z selection 실패
2. Bridge entity not recognized → z 정보 encoding 실패
3. Bridge propagation error → multi-step reasoning 실패

각 카테고리 비율 분석 + 대표 case study
```

---

# 8. 구현 상세

## 8.1 핵심 모델 클래스

```python
class ParametricQA(nn.Module):
    def __init__(self, llm_name="meta-llama/Llama-3.2-3B",
                 num_docs=5000, z_dim=256, m_tokens=8):
        super().__init__()

        # === Document Vectors (learnable) ===
        # shape: [num_docs, m_tokens, hidden_size]
        self.doc_vectors = nn.Parameter(
            torch.randn(num_docs, m_tokens, self.hidden_size) * 0.02
        )

        # === Query Encoder (for selection) ===
        self.query_encoder = AutoModel.from_pretrained("intfloat/e5-base-v2")
        self.query_proj = nn.Linear(768, z_dim)  # E5 output → z space

        # === LLM with QLoRA ===
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            ),
            device_map="auto"
        )
        self.llm = get_peft_model(self.llm, LoraConfig(
            r=64, lora_alpha=128,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM"
        ))

        # === z → LLM embedding projection ===
        self.z_to_embedding = nn.Linear(z_dim, self.llm.config.hidden_size)

    def select_documents(self, query_tokens, k=5):
        """Query와 가장 유사한 k개의 document vector 선택"""
        # Query encoding
        q_output = self.query_encoder(query_tokens)
        q_embed = q_output.last_hidden_state[:, 0]  # [CLS]
        q_proj = self.query_proj(q_embed)  # [batch, z_dim]

        # z_i의 mean pooling으로 비교
        z_mean = self.doc_vectors.mean(dim=1)  # [num_docs, hidden_size]

        # Cosine similarity
        scores = F.cosine_similarity(
            q_proj.unsqueeze(1),   # [batch, 1, z_dim]
            z_mean.unsqueeze(0),   # [1, num_docs, z_dim]
            dim=-1
        )  # [batch, num_docs]

        top_k = torch.topk(scores, k, dim=-1)
        return top_k.indices, top_k.values

    def forward(self, query_ids, doc_indices):
        """선택된 z_i로 답변 생성"""
        # 선택된 document vectors
        selected_z = self.doc_vectors[doc_indices]  # [batch, k, m_tokens, hidden]
        selected_z = selected_z.view(selected_z.size(0), -1, selected_z.size(-1))
        # → [batch, k*m_tokens, hidden]

        # z를 LLM embedding space로 변환
        z_embeds = self.z_to_embedding(selected_z)  # [batch, k*m, llm_hidden]

        # Query embedding
        query_embeds = self.llm.get_input_embeddings()(query_ids)

        # Concatenate: [z_tokens | query_tokens]
        combined = torch.cat([z_embeds, query_embeds], dim=1)

        # Generate
        outputs = self.llm(inputs_embeds=combined)
        return outputs
```

## 8.2 Write Phase 학습 루프

```python
def train_write_phase(model, corpus, epochs=100, lr=1e-3):
    """Stage 1: z_i만 학습, LLM frozen"""

    # LLM freeze
    for param in model.llm.parameters():
        param.requires_grad = False

    # z_i만 optimizer에 등록
    optimizer = torch.optim.AdamW([model.doc_vectors], lr=lr)

    for epoch in range(epochs):
        for doc_id, doc_text in enumerate(corpus):
            z_i = model.doc_vectors[doc_id].unsqueeze(0)  # [1, m, hidden]

            # Document tokenization
            doc_tokens = tokenizer(doc_text, return_tensors='pt',
                                   max_length=512, truncation=True)
            doc_embeds = model.llm.get_input_embeddings()(doc_tokens.input_ids)

            # Teacher forcing: [z_i] + [doc_tokens[:-1]] → predict doc_tokens[1:]
            combined = torch.cat([z_i, doc_embeds[:, :-1, :]], dim=1)

            outputs = model.llm(inputs_embeds=combined)

            # Loss: predict document tokens (after z positions)
            logits = outputs.logits[:, model.m_tokens-1:-1, :]
            targets = doc_tokens.input_ids[:, 1:]

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Monitoring
            if doc_id % 100 == 0:
                ppl = torch.exp(loss).item()
                print(f"Epoch {epoch}, Doc {doc_id}: PPL = {ppl:.2f}")
```

## 8.3 Read Phase 학습 루프

```python
def train_read_phase(model, qa_dataset, epochs=3, lr=2e-5, alpha=0.1):
    """Stage 2: LLM (QLoRA) + z_i + query encoder 함께 학습"""

    # 모든 학습 가능 파라미터
    optimizer = torch.optim.AdamW([
        {'params': model.doc_vectors, 'lr': 1e-3},
        {'params': model.query_proj.parameters(), 'lr': 1e-4},
        {'params': model.z_to_embedding.parameters(), 'lr': 1e-4},
        {'params': model.llm.parameters(), 'lr': lr},  # QLoRA params만
    ])

    for epoch in range(epochs):
        for question, answer, gold_doc_ids in qa_dataset:
            # 1. Document selection
            q_tokens = tokenizer(question, return_tensors='pt')
            selected_ids, scores = model.select_documents(q_tokens, k=5)

            # 2. Answer generation
            outputs = model(q_tokens.input_ids, selected_ids)

            # 3. QA Loss
            answer_tokens = tokenizer(answer, return_tensors='pt')
            qa_loss = F.cross_entropy(
                outputs.logits[:, -answer_tokens.size(1)-1:-1, :].reshape(-1, vocab_size),
                answer_tokens.input_ids.reshape(-1)
            )

            # 4. Retrieval Loss (gold docs should rank higher)
            gold_mask = torch.zeros(scores.size(-1), dtype=torch.bool)
            gold_mask[gold_doc_ids] = True
            retrieval_loss = -torch.log(scores[:, gold_mask].sum(dim=-1) + 1e-8).mean()

            # 5. Total Loss
            total_loss = qa_loss + alpha * retrieval_loss
            total_loss.backward()

            optimizer.step()
            optimizer.zero_grad()
```

## 8.4 새 문서 적응 (z Initialization)

```python
def adapt_new_document(model, new_doc_text, adaptation_steps=50, lr=5e-3):
    """새 문서에 대한 z_i 빠른 적응 (교수님 요구: initialization 가능한 vector)"""

    # θ frozen, z_new만 학습
    z_new = nn.Parameter(torch.randn(1, model.m_tokens, model.hidden_size) * 0.02)
    optimizer = torch.optim.Adam([z_new], lr=lr)

    doc_tokens = tokenizer(new_doc_text, return_tensors='pt',
                           max_length=512, truncation=True)
    doc_embeds = model.llm.get_input_embeddings()(doc_tokens.input_ids)

    for step in range(adaptation_steps):
        combined = torch.cat([z_new, doc_embeds[:, :-1, :]], dim=1)
        outputs = model.llm(inputs_embeds=combined)

        logits = outputs.logits[:, model.m_tokens-1:-1, :]
        targets = doc_tokens.input_ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return z_new.detach()

# 평가: adaptation_steps vs QA 성능 곡선 제시
# → "n step만에 RAG-competitive한 z를 만들 수 있는가?"
```

## 8.5 코드 디렉토리 구조

```
parametric-qa/
├── configs/
│   ├── phase1_poc.yaml          # POC 설정
│   ├── phase2_ablation.yaml     # Ablation 설정
│   ├── phase3_full.yaml         # Full evaluation 설정
│   └── phase4_multihop.yaml     # Multi-hop 설정
├── data/
│   ├── download.py              # NQ, HotpotQA 다운로드
│   ├── preprocess.py            # FlashRAG 형식 전처리
│   └── dataloader.py            # PyTorch DataLoader
├── models/
│   ├── parametric_qa.py         # 통합 모델 (ParametricQA class)
│   ├── write_phase.py           # z_i 학습 모듈
│   ├── read_phase.py            # Selection + Generation
│   ├── router.py                # Learned Router
│   └── adaptation.py            # 새 문서 z initialization
├── training/
│   ├── train_write.py           # Stage 1: Write Phase
│   ├── train_read.py            # Stage 2: Read Phase
│   └── train_e2e.py             # End-to-end (optional)
├── evaluation/
│   ├── metrics.py               # EM, F1, ROUGE, Perplexity
│   ├── evaluate_write.py        # Reconstruction 평가
│   ├── evaluate_qa.py           # QA 평가 (NQ, HotpotQA)
│   ├── evaluate_efficiency.py   # Latency, Memory, Storage
│   └── evaluate_ragas.py        # RAGAS scores
├── baselines/
│   ├── standard_rag.py          # RAG baseline
│   ├── no_retrieval.py          # No retrieval
│   ├── run_flashrag.py          # FlashRAG 통합 실행
│   └── run_corag.py             # CoRAG 실행
├── analysis/
│   ├── error_analysis.py        # 오류 분석
│   ├── visualization.py         # t-SNE, ablation plots
│   └── bridge_entity.py         # Multi-hop 분석
├── scripts/
│   ├── run_phase1.sh
│   ├── run_phase2.sh
│   ├── run_phase3.sh
│   └── run_phase4.sh
├── requirements.txt
└── README.md
```

---

# 9. 하드웨어 및 소프트웨어 스택

## 9.1 하드웨어

### GCP 설정

```
Machine: g2-standard-8
├── vCPU: 8
├── Memory: 32GB RAM
├── GPU: NVIDIA L4 (24GB VRAM)
├── Disk: 200GB SSD
└── Cost: ~$0.83/hour (on-demand)

예상 총 비용: $200-300 (240 GPU-hours, 3주)
```

### Memory Budget (L4 24GB)

```
╔══════════════════════════════════════════════════╗
║  Component                    │  VRAM Usage      ║
╠══════════════════════════════════════════════════╣
║  Llama-3.2-3B (QLoRA 4bit)   │  ~4GB            ║
║  Llama-3.1-8B (QLoRA 4bit)   │  ~8GB            ║
║  Document vectors (50K×8×h)  │  ~25-50MB        ║
║  Query encoder (E5-base)     │  ~440MB          ║
║  z_to_embedding layer        │  ~8MB            ║
║  Gradient + Activations      │  ~8-12GB         ║
╠══════════════════════════════════════════════════╣
║  Total (8B model)            │  ~20-22GB        ║
║  Headroom                    │  ~2-4GB          ║
╚══════════════════════════════════════════════════╝

→ Batch size 2, Gradient accumulation 8
→ Effective batch: 16
→ Gradient checkpointing: 필수
→ Flash Attention 2: 필수
```

## 9.2 소프트웨어 스택

```python
# requirements.txt

# Core
torch>=2.1.0
transformers>=4.36.0
peft>=0.7.0              # LoRA/QLoRA
bitsandbytes>=0.41.0     # 4-bit quantization

# Optimization
flash-attn>=2.3.0        # Flash Attention 2
accelerate>=0.24.0       # Multi-GPU, mixed precision

# Data
datasets>=2.14.0         # HuggingFace datasets
sentence-transformers>=2.2.0  # E5 encoder

# Evaluation
ragas                    # Reference-free RAG evaluation
# flashrag>=0.1.0        # Baselines (별도 설치)

# Utilities
wandb                    # Experiment tracking
tqdm
numpy
pandas
matplotlib
scikit-learn             # t-SNE visualization
```

## 9.3 Training Configuration

```python
# QLoRA Config
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training Args
training_config = {
    "per_device_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "effective_batch_size": 16,
    "learning_rate_llm": 2e-5,
    "learning_rate_z": 1e-3,
    "learning_rate_projections": 1e-4,
    "num_epochs_write": 100,  # Stage 1 (per doc)
    "num_epochs_read": 3,     # Stage 2
    "warmup_ratio": 0.1,
    "fp16": True,  # or bf16
    "gradient_checkpointing": True,
    "max_input_length": 2048,
    "max_doc_length": 512,
    "logging_steps": 50,
    "save_strategy": "epoch",
    "eval_strategy": "steps",
    "eval_steps": 500,
}
```

---

# 10. 예상 결과 및 리스크 대응

## 10.1 예상 결과 시나리오

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
| NQ EM | 28-32% | RAG보다 5-8% 낮음 |
| HotpotQA F1 | 25-30% | Multi-hop 제한적 |
| Latency | 30-50ms | RAG 대비 3-5x 빠름 |
| Storage | 1MB/1K docs | 50x 압축 |

### Worst Case

| Metric | 예상 | 의미 |
|--------|------|------|
| NQ EM | < 25% | z_i 정보 손실 심각 |
| Selection Recall@5 | < 30% | 관련 문서 못 찾음 |
| Multi-hop | 완전 실패 | bridge reasoning 불가 |

## 10.2 Success Criteria (논문 가능 최소 기준)

```
필수 조건 (모두 충족):
1. Reconstruction PPL < 10 (z_i가 문서 정보를 encode함)
2. Selection Recall@5 > 50% (관련 문서를 찾을 수 있음)
3. NQ EM ≥ 28% (Standard RAG 대비 10% 이내)
4. Latency < 50ms (Standard RAG 대비 3x 이상 개선)

부가 조건 (강한 논문):
5. Storage < 5MB/1K docs (50x 이상 압축)
6. HotpotQA F1 ≥ 25% (Multi-hop에서도 일부 작동)
7. RAGAS Faithfulness > 0.7 (답변 신뢰성)
```

## 10.3 리스크 대응 계획

| Risk | 발생 확률 | Impact | Mitigation |
|------|-----------|--------|------------|
| z_i 정보 손실 | 높음 | 치명적 | z_dim ↑, m_tokens ↑, reconstruction loss 모니터링 |
| Selection 실패 | 중간 | 높음 | Learned router, hard negative mining, E5 projection |
| Multi-hop 실패 | 높음 | 중간 | Iterative selection, write-time linking |
| Memory overflow | 낮음 | 중간 | Gradient checkpointing, batch ↓, accumulation ↑ |
| 학습 불안정 | 중간 | 중간 | LR warmup, gradient clipping, z orthogonal init |
| Baseline 재현 실패 | 낮음 | 낮음 | FlashRAG 표준 설정 사용 |

## 10.4 Pivot 전략

```
Phase 1 완전 실패 시 (PPL > 50, Recall < 10%):
├── Pivot A: z_dim을 hidden_size 전체로 확장 (m_tokens=1, z_dim=4096)
├── Pivot B: Encoder 기반 initialization (E5로 초기값 생성)
└── Pivot C: 연구 방향 재정의 (compression ratio-accuracy curve만 분석)

Phase 3에서 RAG 대비 20%+ 성능 저하 시:
├── Efficiency 관점으로 framing 전환 (accuracy-latency Pareto front)
├── Robustness 관점 추가 (MIRAGE NV/CM에서 우위)
└── Scalability 분석 추가 (document 수 증가에 따른 degradation)
```

---

# 11. 후속 연구 로드맵

## 11.1 Phase 5: z Initializer (이번 연구 이후 즉시)

```
문제: 현재 방식은 새 문서마다 z를 SGD로 최적화 (느림)
목표: z = g(D) 형태로 문서에서 z를 한 번에 생성하는 encoder 학습

방법:
- Encoder f: Document → z_i (one forward pass)
- 학습: 이미 최적화된 {z_i*}를 target으로 MSE loss
- 참고: DyPRAG (ICLR 2025) - document → parameter translator

평가:
- Adaptation time: SGD vs Encoder (speed-up ratio)
- QA 성능: Encoder z vs Optimized z (quality gap)
```

## 11.2 Phase 6: Generative Retrieval 통합

```
문제: 현재 selection은 별도 query encoder에 의존
목표: LLM이 직접 어떤 z를 사용할지 결정 (retriever 완전 제거)

방법:
- DSI 계열: query → z_id 생성 (generative retrieval)
- 또는: attention weight로 자동 z selection (implicit routing)

참고: DSI (Transformer Memory as Differentiable Search Index)
```

## 11.3 Phase 7: Robustness & Safety

```
문제: z_i가 poisoning attack에 취약할 수 있음
목표: 노이즈/공격에 대한 강건성 검증

방법:
- MIRAGE 설정으로 노이즈 주입 실험
- Adversarial z injection (poisoned z_i → wrong answers)
- Detection & Recovery mechanism

연결: 메모리 안전성 연구 (Memory-centric LLM Safety)
```

## 11.4 Phase 8: Scaling & Production

```
문제: N=50K에서 실험 → N=1M+로 확장 시?
목표: 대규모 z 테이블 관리 방법론

방법:
- Hierarchical z (document → section → sentence 계층)
- Codebook quantization (z 벡터 양자화)
- Memory consolidation (자주 쓰이는 정보 → LLM weight로 통합)
- Incremental learning (새 문서 추가 시 기존 z 영향 없이)
```

---

# 12. 연구 질문 (RQs) 및 Deliverables

## 12.1 Research Questions

| RQ | 질문 | 검증 실험 | 핵심 지표 |
|----|------|-----------|-----------|
| **RQ1** | Per-document learnable vector가 문서 정보를 효과적으로 압축할 수 있는가? | Phase 1 (Write) | Reconstruction PPL, ROUGE-L |
| **RQ2** | z_i 기반 selection이 traditional retrieval과 comparable한가? | Phase 2-3 | Recall@K, Selection Precision |
| **RQ3** | Parametric QA가 latency-accuracy trade-off에서 우위가 있는가? | Phase 3 | EM vs Latency curve (Pareto front) |
| **RQ4** | Multi-hop reasoning에서 z_i 접근법의 한계는 무엇인가? | Phase 4 | HotpotQA F1, Bridge Entity Recall |
| **RQ5** | 새 문서에 대한 z adaptation은 얼마나 빠르게 가능한가? | Phase 1-2 | Adaptation steps vs QA accuracy |

## 12.2 Deliverables

| 산출물 | 내용 | 형태 |
|--------|------|------|
| **코드** | 재현 가능한 실험 코드 | GitHub repo |
| **모델** | 학습된 z_i vectors + LoRA weights | HuggingFace checkpoint |
| **결과** | Main results + Ablation tables + Figures | LaTeX tables, matplotlib |
| **분석** | Error analysis + z visualization + Scalability | Jupyter notebooks |
| **논문** | Problem → Method → Experiments → Analysis | LaTeX draft |

## 12.3 교수님 보고용 요약 (4장 슬라이드)

```
Slide 1: 문제정의
- "Vector DB → (Local LLM + z) 치환으로 RAG를 대체"
- 기존 RAG 한계 + 우리 제안 (1장 다이어그램)

Slide 2: 방법론
- Write Phase: z_i → D_i 학습 (수식 1줄)
- Read Phase: z_i selection → LLM generation (수식 1줄)
- 아키텍처 다이어그램

Slide 3: 실험 설계
- Corpus: NQ 50K, HotpotQA
- Baselines: 7개 (No-RAG ~ CoRAG)
- Metrics: EM/F1 + Recall@K + Latency + RAGAS

Slide 4: 결과 (예상/실제)
- Main results table
- Efficiency comparison
- Key insights
```

---

# 부록

## A. 전체 일정표

```
Week 1 (Day 1-5): 환경 구축 + POC
├── Day 1-2: GCP 환경, 데이터 준비, FlashRAG 설치
├── Day 3-4: Phase 1 (Sanity Check) 실험
└── Day 5: 결과 분석, Phase 2 방향 결정

Week 2 (Day 6-10): Ablation + 본실험 시작
├── Day 6-8: Phase 2 (Ablation Studies)
├── Day 9-10: Phase 3 시작 (Full Evaluation)
└── 중간 결과 교수님 보고

Week 3 (Day 11-15): 본실험 완료 + 분석
├── Day 11-12: Phase 3 완료
├── Day 13-14: Phase 4 (Multi-hop)
└── Day 15: 최종 결과 정리, 보고서 작성
```

## B. 핵심 참고 논문 (정렬)

### Parametric Memory & Model Editing

| 논문 | 학회/년도 | 핵심 기여 |
|------|-----------|-----------|
| Geva et al. "FFN as Key-Value Memory" | EMNLP 2021 | 이론적 기반 |
| ROME | NeurIPS 2022 | Causal tracing, rank-one edit |
| MEMIT | ICLR 2023 | 10,000+ 동시 편집 |
| MemoryLLM | ICML 2024 | Self-updating 1B memory pool |
| M+ | ICML 2025 | Long-term memory extension |

### Token/Document Compression

| 논문 | 학회/년도 | 압축률 | 핵심 |
|------|-----------|--------|------|
| Gist Tokens | NeurIPS 2023 | 26x | Virtual token caching |
| ICAE | ICLR 2024 | 4-16x | LoRA encoder, autoencoding |
| xRAG | NeurIPS 2024 | 178x | Modality bridge, retriever 재사용 |
| 500xCompressor | ACL 2025 | 480x | KV value compression |
| DyPRAG | ICLR 2025 | - | Document→parameter translator |

### RAG & Retrieval

| 논문 | 학회/년도 | 핵심 혁신 |
|------|-----------|-----------|
| IRCoT | ACL 2023 | CoT + iterative retrieval |
| Self-RAG | ICLR 2024 | Reflection tokens |
| HippoRAG | NeurIPS 2024 | Hippocampal indexing |
| Adaptive-RAG | NAACL 2024 | Query complexity routing |
| CoRAG | NeurIPS 2025 | Retrieval chain SOTA |
| Search-R1 | arXiv 2025 | Pure RL, +41% |
| R1-Searcher | arXiv 2025 | Two-stage RL, +48% |

### Generative Retrieval

| 논문 | 핵심 |
|------|------|
| DSI | Query → doc ID 생성 (parametric index) |
| DSI++ | 증분 인덱싱 |

### 벤치마크 & 평가

| 도구 | 용도 |
|------|------|
| FlashRAG (WWW 2025) | 36 datasets, 23 methods, 통합 벤치마크 |
| RAGAS | Reference-free RAG 평가 (4 metrics) |
| ARES (NAACL 2024) | 통계적 자동 평가 (confidence interval) |
| RAGChecker | Claim-level retrieval/generation 진단 |
| MIRAGE | RAG 강건성 평가 (NV/CA/CI/CM) |
| RAGBench (NeurIPS 2024) | 100K industry examples |

## C. Reproducibility Checklist

- [ ] Exact model version/checkpoint 명시
- [ ] Wikipedia dump date, passage count 문서화
- [ ] Chunk size, overlap, embedding model 기록
- [ ] 모든 하이퍼파라미터 (LR, batch, epochs, LoRA rank/alpha)
- [ ] Hardware spec, training time 보고
- [ ] Random seeds (3개), mean ± std 보고
- [ ] Code + evaluation scripts 공개 (FlashRAG compatibility)
- [ ] z_i 학습 step 수, convergence criteria 명시

---

**문서 끝**

*이 계획서는 docs/ 내 5개 문서 (종합 보고서 ×2, 실험 설계 ×2, RAG 방법론 가이드)를 종합하여 작성되었습니다.*
*모든 실험 설계는 FlashRAG 표준 + RAGAS 평가 + 교수님 연구 비전에 정합하도록 구성되었습니다.*
