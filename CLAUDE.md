# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

zRAG is a research project implementing a **Parametric QA** system that replaces traditional RAG retrieval with per-document learnable vectors (z_i). The core idea: compress documents into learnable embeddings, then use similarity-based selection + LLM generation instead of vector DB retrieval.

**Current Stage:** Research planning (documentation complete, code implementation not yet started).

## Research Architecture

```
Write Phase (Offline):  Document D_i → Encoder → z_i (learnable vector)
Read Phase (Inference):  Query q → Select z_i (top-k similarity) → LLM → Answer
```

**Goal:** Achieve RAG-competitive accuracy (within 5% EM) with 3-5x latency improvement and 50-100x storage compression.

## Planned Tech Stack

- **Core:** PyTorch, Transformers, PEFT (QLoRA), bitsandbytes (4-bit quantization)
- **Models:** Llama-3.2-3B (POC) → Llama-3.1-8B (full experiments)
- **Evaluation:** FlashRAG, RAGAS, lm-eval-harness
- **Data:** sentence-transformers, faiss-gpu, datasets (HuggingFace)
- **Hardware:** GCP g2-standard-8 (L4 24GB, bf16 mixed precision)

## Planned Code Structure

```
parametric-qa/
├── configs/         # YAML configs for each experiment phase
├── data/            # Download, preprocess, dataloader
├── models/          # write_phase.py, read_phase.py, router.py, parametric_qa.py
├── training/        # train_write.py (Stage 1), train_read.py (Stage 2), train_e2e.py
├── evaluation/      # metrics.py (EM, F1, ROUGE), evaluate_write.py, evaluate_qa.py
├── baselines/       # standard_rag.py, no_retrieval.py, run_baselines.py
└── scripts/         # Shell scripts for running experiments
```

## Training Pipeline

- **Stage 1 (Write Phase):** Learn z_i by maximizing log P(D_i | z_i; LLM). Only z_i is trained; LLM is frozen. Loss: cross-entropy with teacher forcing.
- **Stage 2 (Read Phase):** QA fine-tuning with LLM + z_i + query encoder jointly trained. Loss: QA loss + 0.1 × Retrieval loss.

## Experiment Design

- **Datasets:** Natural Questions (NQ) primary; HotpotQA for multi-hop extension
- **Baselines (7):** No Retrieval, BM25+LLM, Standard RAG, Self-RAG, IRCoT, Adaptive-RAG, CoRAG
- **Metrics:** EM, Token F1, LLM Judge accuracy, Recall@K, MRR, Latency (ms), Storage (MB/1K docs), RAGAS scores
- **Phases:** POC (3d) → Ablation (4d) → Full Eval (5d) → Multi-hop (3d)

## Key Documents

- `docs/LLM as Memory 연구 종합 보고서(claude).md` — Final research direction (v3.0), Parametric QA architecture, evolution from M+ to token learning
- `docs/실험 설계 예시(claude).md` — Detailed experiment plan with baselines, metrics, hardware budget, success criteria
- `docs/RAG Experiment Design Methodology...md` — Meta-analysis of 2024-2025 RAG papers, baseline selection principles, evaluation frameworks

## Research Context

The project evolved through advisor feedback:
1. Phase 1: M+ analysis → Write-time/Iterative Retrieval
2. Phase 2: Advisor directed pivot to Token Learning (compress documents into learnable tokens)
3. Phase 3 (Final): Parametric QA — learnable document vectors + full LLM fine-tuning

Key references: Gist Tokens (NeurIPS 2023), ICAE (ICLR 2024), xRAG (NeurIPS 2024), 500xCompressor (ACL 2025), M+ (ICML 2025), CoRAG (NeurIPS 2025).
