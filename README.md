# zRAG: LLM-as-Memory

> **Replacing Vector DB with Internal LLM Memory**

This project replaces the traditional "Retriever + Vector DB" pipeline in RAG systems with **learnable memory vectors** inside a Local LLM to generate evidence directly.

## Core Idea

```
Traditional RAG:
Query → [Retriever] → Vector DB → Retrieved Text → LLM → Answer
              ↑
        External Module

LLM-as-Memory (Ours):
Query → [Local LLM + Internal Memory Z] → Evidence → ChatGPT → Answer
                    ↑
              No External Module
              Internal routing selects z
```

### Key Differences

| Traditional RAG | LLM-as-Memory |
|-----------------|---------------|
| Requires external retriever | **No retriever** |
| Stores text in Vector DB | **Compressed into z vectors** |
| Embedding similarity search | **LLM internal attention routing** |
| Retrieved text → LLM → Answer | **Evidence generation → ChatGPT → Answer** |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Local LLM (Qwen3-8B)                     │
│                                                             │
│   Memory Pool: Z = {z₁, z₂, ..., zₙ}                       │
│   (N docs × 4 tokens × 256 dim = learnable vectors)        │
│                                                             │
│   Query ──→ [Internal Attention over Z] ──→ Evidence (text) │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              Query + Evidence ──→ ChatGPT ──→ Answer
```

### Training Objective

```
L = -log P(evidence | query, Z; θ)
```

- **θ**: Local LLM parameters (QLoRA)
- **Z**: Per-document learnable memory vectors
- **evidence**: Target text to generate

## Quick Start

### Installation

```bash
# Using uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
cd zrag
uv sync

# Or using pip
pip install -e .
```

### Test

```bash
# Minimal test (no heavy dependencies)
uv run python scripts/test_integration.py --minimal

# Full test (requires GPU)
uv run python scripts/test_integration.py --full
```

### Training

```bash
uv run python training/train_evidence.py --config configs/evidence_poc.yaml
```

## Project Structure

```
zrag/
├── models/                      # Model implementations
│   ├── parametric_memory_llm.py # Core model (Z prefix + Evidence generation)
│   ├── evidence_trainer.py      # Trainer
│   └── legacy/                  # Legacy code
├── data/                        # Data processing
│   └── evidence_dataloader.py   # Evidence training DataLoader
├── training/                    # Training scripts
│   └── train_evidence.py        # Main training script
├── evaluation/                  # Evaluation
│   ├── evidence_metrics.py      # Evidence quality metrics (ROUGE-L, Answer Coverage)
│   └── evaluate_qa.py           # QA and Evidence evaluation
├── baselines/                   # Baseline implementations
│   └── standard_rag.py          # Dense/BM25 RAG Baseline
├── configs/                     # Configuration files
│   └── evidence_poc.yaml        # POC configuration
└── scripts/                     # Utilities
    └── test_integration.py      # Integration test
```

## Configuration (POC)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_docs` | 2,000 | Number of documents to store |
| `m_tokens` | 4 | Memory tokens per document |
| `z_dim` | 256 | Memory vector dimension |
| `llm` | Qwen3-8B | Local LLM (QLoRA 4-bit) |

## Requirements

- Python 3.10+
- PyTorch 2.1+
- Transformers, PEFT, bitsandbytes
- CUDA GPU (for training/inference)

## References

- Soft Prompt: Prompt Tuning, P-tuning v2, Prefix Tuning
- Generative Retrieval: DSI, DSI++
- Document Compression: Gist Tokens, ICAE, xRAG
- Parametric RAG: DyPRAG

## License

MIT
