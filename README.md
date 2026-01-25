# zRAG: LLM-as-Memory

> **Vector DB를 Local LLM의 내부 Memory로 대체하는 연구**

기존 RAG의 "Retriever + Vector DB" 파이프라인을 제거하고, LLM 내부의 **learnable memory vectors**로 대체하여 evidence를 생성하는 시스템입니다.

## 핵심 아이디어

```
기존 RAG:
Query → [Retriever] → Vector DB → Retrieved Text → LLM → Answer
              ↑
         외부 모듈

LLM-as-Memory (Ours):
Query → [Local LLM + Internal Memory Z] → Evidence → ChatGPT → Answer
                    ↑
              외부 모듈 없음
              내부 routing으로 z 선택
```

### 핵심 차별점

| 기존 RAG | LLM-as-Memory |
|----------|---------------|
| 외부 Retriever 필요 | **Retriever 제거** |
| Vector DB에 텍스트 저장 | **z_i (벡터)로 압축 저장** |
| Embedding similarity 검색 | **LLM 내부 attention routing** |
| Retrieved text → LLM → Answer | **Evidence 생성 → ChatGPT → Answer** |

## 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    Local LLM (Qwen3-8B)                     │
│                                                             │
│   Memory Pool: Z = {z₁, z₂, ..., zₙ}                       │
│   (N docs × 4 tokens × 256 dim = learnable vectors)        │
│                                                             │
│   Query ──→ [내부 Attention over Z] ──→ Evidence (텍스트)   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              Query + Evidence ──→ ChatGPT ──→ Answer
```

### 학습 목표

```
L = -log P(evidence | query, Z; θ)
```

- **θ**: Local LLM 파라미터 (QLoRA)
- **Z**: 문서별 learnable memory vectors
- **evidence**: 생성할 근거 텍스트

## Quick Start

### 설치

```bash
# uv 사용 (권장)
curl -LsSf https://astral.sh/uv/install.sh | sh
cd zrag
uv sync

# 또는 pip 사용
pip install -e .
```

### 테스트

```bash
# 최소 테스트 (의존성 없이)
uv run python scripts/test_integration.py --minimal

# 전체 테스트 (GPU 필요)
uv run python scripts/test_integration.py --full
```

### 학습

```bash
uv run python training/train_evidence.py --config configs/evidence_poc.yaml
```

## 프로젝트 구조

```
zrag/
├── models/                      # 모델 구현
│   ├── parametric_memory_llm.py # 핵심 모델 (Z prefix + Evidence 생성)
│   ├── evidence_trainer.py      # 학습기
│   └── legacy/                  # 레거시 코드
├── data/                        # 데이터 처리
│   └── evidence_dataloader.py   # Evidence 학습용 DataLoader
├── training/                    # 학습 스크립트
│   └── train_evidence.py        # 메인 학습 스크립트
├── evaluation/                  # 평가
│   ├── evidence_metrics.py      # Evidence 품질 메트릭 (ROUGE-L, Answer Coverage)
│   └── evaluate_qa.py           # QA 및 Evidence 평가
├── baselines/                   # Baseline 구현
│   └── standard_rag.py          # Dense/BM25 RAG Baseline
├── configs/                     # 설정 파일
│   └── evidence_poc.yaml        # POC 설정
├── docs/                        # 문서
│   └── LLM-as-Memory 실험 설계서.md
└── scripts/                     # 유틸리티
    └── test_integration.py      # 통합 테스트
```

## 설정 (POC)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `num_docs` | 2,000 | 저장 가능한 문서 수 |
| `m_tokens` | 4 | 문서당 memory token 수 |
| `z_dim` | 256 | Memory vector 차원 |
| `llm` | Qwen3-8B | Local LLM (QLoRA 4-bit) |

## 의존성

- Python 3.10+
- PyTorch 2.1+
- Transformers, PEFT, bitsandbytes
- CUDA GPU (학습/추론용)

## 참고 문헌

- Soft Prompt: Prompt Tuning, P-tuning v2, Prefix Tuning
- Generative Retrieval: DSI, DSI++
- Document Compression: Gist Tokens, ICAE, xRAG
- Parametric RAG: DyPRAG

## License

MIT
