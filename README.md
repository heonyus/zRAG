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

## Phased Approach

| Phase | Name | Goal | Status |
|-------|------|------|--------|
| **1** | Write (Token Learning) | z_i → D_i generation | ✅ Implemented |
| **2** | Read - Option A | [Z_all] + query → evidence | 🔄 Ready |
| **3** | Read - Option B-1 | KV Injection (scale to N=2000+) | 📋 Planned |
| **4** | Read - Option B-2 | Resampler (scale to N=10000+) | 📋 Planned |

## Architecture

### Phase 1: Write (Token-as-Document)

```
┌─────────────────────────────────────────────────────────────┐
│                    Local LLM (Qwen3-8B) - FROZEN            │
│                                                             │
│   z_i (learnable tokens)                                    │
│        │                                                    │
│        ▼                                                    │
│   [Projection] → LLM → Document D_i                         │
│                                                             │
│   Loss: -log P(D_i | z_i)                                   │
│   Learn: z_i only (LLM frozen)                              │
└─────────────────────────────────────────────────────────────┘
```

### Phase 3: Read (Z_all Concat)

```
┌─────────────────────────────────────────────────────────────┐
│                    Local LLM (Qwen3-8B)                     │
│                                                             │
│   Memory Pool: Z = {z₁, z₂, ..., zₙ}                       │
│   (N docs × 4 tokens × 256 dim = learnable vectors)        │
│                                                             │
│   [Z_all prefix] + Query → LLM → Evidence                   │
│                                                             │
│   Internal attention routes to relevant z                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              Query + Evidence → ChatGPT → Answer
```

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

### Phase 1: Write Phase Training

```bash
# Full training (200 docs, 100 epochs per doc)
python training/train_write_phase.py --config configs/phase1_write.yaml

# Quick test (10 docs, 20 epochs per doc)
python training/train_write_phase.py --config configs/phase1_write.yaml --test
```

### Phase 3: Read Phase Training

```bash
# After Phase 1 completes, load z_pool and train evidence generation
python training/train_evidence.py --config configs/evidence_poc.yaml
```

## Project Structure

```
zrag/
├── models/
│   ├── write_phase_model.py     # Phase 1: z_i → D_i (NEW)
│   ├── parametric_memory_llm.py # Phase 3: Z_all concat → evidence
│   └── evidence_trainer.py      # Trainer for evidence generation
├── training/
│   ├── train_write_phase.py     # Phase 1 training (NEW)
│   └── train_evidence.py        # Phase 3 training
├── data/
│   ├── dataloader.py            # WritePhaseDataset, ReadPhaseDataset
│   └── download.py              # Dataset download (NQ, HotpotQA)
├── configs/
│   ├── phase1_write.yaml        # Phase 1 config (NEW)
│   └── evidence_poc.yaml        # Phase 3 config
├── evaluation/
│   └── evidence_metrics.py      # ROUGE-L, Answer Coverage
├── baselines/
│   └── standard_rag.py          # BM25/Dense RAG baseline
└── docs/
    └── phases/                  # Detailed phase documentation
```

## Configuration

### Phase 1 (Write)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_docs` | 200 | Documents to learn |
| `m_tokens` | 4 | Memory tokens per document |
| `z_dim` | 256 | Memory vector dimension |
| `epochs_per_doc` | 100 | Training epochs per document |
| `LLM` | Qwen3-8B | Frozen (no LoRA) |

### Phase 3 (Read)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_docs` | 200 | Load from Phase 1 |
| `m_tokens` | 4 | Memory tokens per document |
| `z_dim` | 256 | Memory vector dimension |
| `LLM` | Qwen3-8B | QLoRA fine-tuning |

## Hardware Requirements

| Phase | GPU Memory | Notes |
|-------|------------|-------|
| Phase 1 | ~8GB | Single doc at a time |
| Phase 3 (N=200) | ~12GB | Z prefix = 800 tokens |
| Phase 3 (N=500) | ~16GB | Z prefix = 2000 tokens |
| Phase 3 (N=2000) | ~24GB | Z prefix = 8000 tokens |

**Recommended**: NVIDIA L4 24GB (GCP g2-standard-4)

## Phase 1 → Phase 3 Workflow

```python
# Phase 1: Train z_i for each document
# Output: checkpoints/phase1_write/z_pool.pt

# Phase 3: Load trained z_pool
from models import ParametricMemoryLLM

model = ParametricMemoryLLM(num_docs=200, m_tokens=4, z_dim=256, ...)
model.load_from_phase1(
    z_pool_path="checkpoints/phase1_write/z_pool.pt",
    projection_path="checkpoints/phase1_write/projection.pt"
)

# Continue training for evidence generation
```

## References

- Soft Prompt: Prompt Tuning, P-tuning v2, Prefix Tuning
- Generative Retrieval: DSI, DSI++
- Document Compression: Gist Tokens, ICAE, xRAG
- Parametric RAG: DyPRAG

## License

MIT
