# mz-RAG: LLM-as-Memory

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
| **1** | Write (Token Learning) | z_i → D_i generation | ✅ Completed |
| **1.5** | Evidence Generation | z_i → Evidence (answer-containing) | ✅ Completed |
| **2** | Read - Option A | [Z_all] + query → evidence | 🔄 In Progress |
| **3** | Read - Option B-1 | KV Injection (scale to N=2000+) | 📋 Planned |
| **4** | Read - Option B-2 | Resampler (scale to N=10000+) | 📋 Planned |

---

## Project Structure (Full)

```
zRAG/
├── README.md                 # This file
├── pyproject.toml            # Project dependencies (uv/pip)
├── uv.lock                   # Locked dependencies
│
├── models/                   # Model architectures
│   ├── write_phase_model.py      # Phase 1: z_i → D_i
│   ├── parametric_memory_llm.py  # Phase 2: Z_all concat → evidence
│   └── evidence_trainer.py       # Trainer for evidence generation
│
├── training/                 # Training scripts
│   ├── train_write_phase.py      # Phase 1 training
│   └── train_evidence.py         # Phase 2 training
│
├── data/                     # Data processing
│   ├── dataloader.py             # WritePhaseDataset, ReadPhaseDataset
│   ├── download.py               # Dataset download (NQ, HotpotQA)
│   ├── corpus_builder.py         # Build corpus from HotpotQA
│   ├── build_phase1_5_evidence_dataset.py  # Phase 1.5 dataset builder
│   └── raw/                      # [GITIGNORED] Raw downloaded data
│
├── configs/                  # Configuration files
│   ├── phase1_write.yaml         # Phase 1 config
│   ├── phase1_5.yaml             # Phase 1.5 config
│   └── evidence_poc.yaml         # Phase 2 config
│
├── evaluation/               # Evaluation scripts
│   └── evidence_metrics.py       # ROUGE-L, Answer Coverage
│
├── experiments/              # Experiment runners
│   ├── phase1_analysis/          # Phase 1 ablation study
│   └── phase1_5/                 # Phase 1.5 experiment runner
│
├── baselines/                # Baseline comparisons
│   └── standard_rag.py           # BM25/Dense RAG baseline
│
├── docs/                     # Documentation
│   └── phases/                   # Detailed phase documentation
│
├── checkpoints/              # [GITIGNORED] Model checkpoints
├── results/                  # [GITIGNORED] Experiment results
└── logs/                     # [GITIGNORED] Training logs
```

---

## Gitignored Data Structure (Backup Required)

These folders are NOT in GitHub and must be downloaded separately.

### Download Link
**Backup file**: `zRAG_backup.tar.gz` (1.7GB)

### Extraction
```bash
# Extract to project root
cd /path/to/zRAG
tar -xzvf zRAG_backup.tar.gz
```

### Folder Structure After Extraction

```
zRAG/
├── checkpoints/                          # 401MB - Model weights
│   │
│   ├── phase1_final/                     # Phase 1 initial experiment
│   │   ├── z_pool.pt                     # Final z vectors (50 docs × 4 tokens × 3584 dim)
│   │   ├── z_pool_epoch{10,20,30,40,50}.pt  # Checkpoints per epoch
│   │   ├── projection.pt                 # Projection layer weights
│   │   ├── results.pt                    # Training results
│   │   ├── corpus_manifest.json          # Corpus metadata
│   │   ├── config.yaml                   # Training config
│   │   └── logs/                         # Training logs
│   │
│   ├── phase1_v2/                        # Phase 1 v2 (main experiment)
│   │   ├── z_pool.pt                     # Final z vectors
│   │   ├── z_pool_epoch{10,20,30,40,50}.pt
│   │   ├── projection.pt
│   │   ├── results.pt
│   │   ├── corpus_manifest.json
│   │   └── logs/
│   │
│   ├── phase2_corpus/                    # Corpus for Phase 2
│   │   ├── corpus.json                   # 200 documents
│   │   ├── qa_pairs.json                 # All QA pairs
│   │   ├── qa_train.json                 # Training split
│   │   ├── qa_val.json                   # Validation split
│   │   └── stats.json                    # Corpus statistics
│   │
│   ├── phase2_read/                      # Phase 2 Read experiment
│   │   ├── best.pt                       # Best checkpoint
│   │   ├── best.pt_lora/                 # LoRA adapter
│   │   │   ├── adapter_model.safetensors
│   │   │   └── adapter_config.json
│   │   ├── final.pt
│   │   ├── final.pt_lora/
│   │   ├── results.pt
│   │   └── samples.json
│   │
│   └── phase2_cache/                     # Baseline comparison cache
│       ├── answer_bm25.jsonl
│       ├── answer_contriever.jsonl
│       ├── answer_dense_e5.jsonl
│       ├── answer_zRAG.jsonl
│       ├── evidence_*.jsonl
│       └── ...
│
├── results/                              # 1.4GB - Experiment results
│   │
│   ├── phase1_analysis/                  # Phase 1 ablation study
│   │   └── 20260128_123048/
│   │       ├── 00_meta/                  # Run metadata
│   │       │   ├── effective_config.json
│   │       │   └── run_manifest.json
│   │       ├── 00_logs/                  # Logs
│   │       │   ├── run.log
│   │       │   └── debug.log
│   │       ├── 01_verification/          # A1, A3 tests
│   │       │   ├── A1_confusion/         # Confusion matrix analysis
│   │       │   │   ├── confusion_metrics.json
│   │       │   │   ├── nll_matrix.npy
│   │       │   │   └── artifacts/        # Visualizations
│   │       │   └── A3_zshuffle/          # Z shuffle test
│   │       │       ├── z_shuffle_metrics.json
│   │       │       └── artifacts/
│   │       ├── 02_ablations/             # Ablation studies
│   │       │   └── B2_projection/
│   │       │       ├── proj_normal/
│   │       │       ├── proj_frozen_ckpt/
│   │       │       └── proj_random_frozen/
│   │       └── 03_summary/               # Summary dashboard
│   │           ├── dashboard.md
│   │           └── ablation_summary.json
│   │
│   └── phase1_5/                         # Phase 1.5 experiments
│       └── 20260128_184412/              # Latest run (MAIN RESULT)
│           ├── 00_meta/
│           │   ├── effective_config.json # Experiment config
│           │   └── run_manifest.json
│           ├── 00_logs/
│           │   ├── run.log
│           │   └── debug.log
│           ├── 01_data/
│           │   ├── dataset.jsonl         # Training data
│           │   ├── dataset_manifest.json # Data statistics
│           │   └── samples_preview.md    # Sample preview
│           ├── 02_train/
│           │   ├── train_summary.json    # Training summary
│           │   ├── train_metrics.jsonl   # Per-step metrics
│           │   ├── frozen_params_verification.json
│           │   ├── checkpoints/
│           │   │   ├── best.pt_lora/     # Best LoRA checkpoint
│           │   │   └── last.pt_lora/     # Final LoRA checkpoint
│           │   └── artifacts/
│           │       ├── loss_curve.png
│           │       └── loss_curve.pdf
│           ├── 03_eval/
│           │   ├── evidence_metrics.json # Main evaluation metrics
│           │   ├── evidence_eval_table.csv
│           │   ├── samples/
│           │   │   ├── eyeball_20_best.md    # Best samples
│           │   │   ├── eyeball_20_worst.md   # Worst samples
│           │   │   └── failure_cases.md      # Failures (coverage=0)
│           │   └── artifacts/
│           ├── 04_regression_phase1/
│           │   ├── baseline_metrics.json     # Phase 1 baseline
│           │   ├── post_phase15_metrics.json # After Phase 1.5
│           │   ├── delta.json                # Regression test result
│           │   └── artifacts/
│           └── 05_cache/
│               └── eval_results.jsonl
│
├── logs/                                 # 32KB - Additional logs
│   └── phase2_read/
│       └── train_read_v1.log
│
└── data/
    └── raw/                              # Raw HotpotQA data
        └── hotpot_dev_distractor_v1.json
```

---

## Key Results (Phase 1.5 - 2026-01-28)

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Base Model | Qwen3-8B (frozen) |
| Fine-tuning | LoRA (r=32, alpha=64) |
| Training Samples | 58 |
| Epochs | 10 |
| Final Loss | 0.348 |

### Evaluation Metrics
| Metric | Value | Description |
|--------|-------|-------------|
| **Answer Coverage** | 81.0% | Generated evidence contains gold answer |
| **Source Overlap** | 43.2% | Overlap with original document |
| **ROUGE-L** | 38.3% | Text similarity |
| **4-gram Overlap** | 19.6% | Exact phrase match |

### Regression Test (Phase 1 Retrieval)
| Test | Result | Threshold |
|------|--------|-----------|
| A1 (Top-1 Drop) | 0.0% | < 2% |
| A3 (Shuffle Delta) | 0.98 | > 0.5 |
| **Overall** | ✅ PASS | - |

### Known Issues
- 11 failure cases (Answer Coverage = 0)
- Hallucination: Model generates factually incorrect details
- Paraphrasing: Low source overlap indicates heavy paraphrasing

---

## Quick Start

### Installation

```bash
# Using uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
cd zRAG
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
```

### Download Checkpoints

```bash
# Download and extract backup
# (Get zRAG_backup.tar.gz from shared storage)
tar -xzvf zRAG_backup.tar.gz
```

### Run Phase 1 Training

```bash
python training/train_write_phase.py --config configs/phase1_write.yaml
```

### Run Phase 1.5 Experiment

```bash
python experiments/phase1_5/run_phase1_5.py --config configs/phase1_5.yaml
```

---

## Hardware Requirements

| Phase | GPU Memory | Notes |
|-------|------------|-------|
| Phase 1 | ~8GB | Single doc at a time |
| Phase 1.5 | ~16GB | LoRA fine-tuning |
| Phase 2 (N=200) | ~12GB | Z prefix = 800 tokens |
| Phase 2 (N=2000) | ~24GB | Z prefix = 8000 tokens |

**Recommended**: NVIDIA L4 24GB or A100 40GB

---

## References

- Soft Prompt: Prompt Tuning, P-tuning v2, Prefix Tuning
- Generative Retrieval: DSI, DSI++
- Document Compression: Gist Tokens, ICAE, xRAG
- Parametric RAG: DyPRAG

## License

MIT
