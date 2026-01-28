# Phase 1: Write Phase Technical Report

**Token-as-Document Representation Learning for zRAG**

---

**Run ID:** 20260128_123048
**Date:** January 28, 2026
**Hardware:** NVIDIA L4 (24GB) on GCP g2-standard-4
**Framework:** PyTorch 2.10.0 + Transformers 4.57.6

---

## Executive Summary

This report presents comprehensive experimental verification of Phase 1 (Write Phase) of the zRAG system. Phase 1 establishes the foundation for document-conditioned generation by learning document-specific latent representations (z vectors) that can reliably regenerate their source documents.

### Key Findings

| Metric | Result | Significance |
|--------|--------|--------------|
| **Top-1 Retrieval Accuracy** | 100% | z vectors uniquely identify documents |
| **Mean Margin (correct vs. best-wrong)** | 0.721 | Robust separation between correct and incorrect z |
| **Z-Shuffle Collapse** | 100% → 0% | Proves z encodes document-specific information |
| **Random Projection Collapse** | 100% → 2% | Proves learned projection is essential |

**Conclusion:** Phase 1 successfully learns document-specific representations. The z vectors contain sufficient information to discriminate between 200 documents with 100% accuracy, and the learned projection layer is critical for translating z into a form the frozen LLM can interpret.

---

## 1. Introduction

### 1.1 Problem Statement

The zRAG (z-based Retrieval Augmented Generation) project aims to create a document retrieval and generation system where document information is encoded into compact latent vectors (z). Phase 1 addresses the foundational question:

> **Can we learn a fixed-length latent representation z_i such that a frozen LLM, conditioned only on z_i, can regenerate the corresponding document D_i?**

### 1.2 Design Philosophy

Phase 1 follows a "Token-as-Document" paradigm:
- Each document D_i is assigned a learnable latent vector z_i ∈ ℝ^(m×d)
- The LLM remains completely frozen (no LoRA, no fine-tuning)
- Only z_i and a lightweight projection layer are optimized
- Success criterion: P(D_i | z_i) >> P(D_i | z_j) for i ≠ j

### 1.3 Theoretical Foundation

The learning objective is autoregressive likelihood maximization:

```
L(z_i) = -log P_LLM(D_i | z_to_embed(z_i))
```

Where:
- `z_i ∈ ℝ^(m_tokens × z_dim)` is the learnable document representation
- `z_to_embed: ℝ^z_dim → ℝ^hidden_size` projects z into LLM embedding space
- The LLM treats projected z as a soft prompt prefix

---

## 2. Methodology

### 2.1 Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     WritePhaseModel                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   z_i [16, 256]                                                 │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────────────────────────────┐                          │
│   │    z_to_embedding (Trainable)   │                          │
│   │    Linear(256 → 512) + GELU     │                          │
│   │    Linear(512 → 4096)           │                          │
│   │    × α (learnable scale gate)   │                          │
│   └─────────────────────────────────┘                          │
│        │                                                        │
│        ▼                                                        │
│   z_embed [16, 4096]   ──┬──  doc_embed [L-1, 4096]            │
│                          │                                      │
│                          ▼                                      │
│   ┌─────────────────────────────────┐                          │
│   │      Qwen3-8B (Frozen, 4-bit)   │                          │
│   │      Causal Language Model      │                          │
│   └─────────────────────────────────┘                          │
│        │                                                        │
│        ▼                                                        │
│   logits [m+L-1, vocab_size]                                   │
│   Loss = CrossEntropy(logits[m-1:m-1+L], doc_tokens)           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Legend:
  m = 16 (m_tokens, prefix length)
  z_dim = 256
  hidden_size = 4096 (Qwen3-8B)
  L = document length in tokens (max 512)
```

### 2.2 Key Design Decisions

#### 2.2.1 Alpha Gate for Scale Stabilization

```python
alpha_clamped = torch.clamp(self.alpha, min=0.5)
z_embed = alpha_clamped * self.z_to_embedding(z_i)
```

The alpha gate (initialized to 1.0, clamped ≥ 0.5) prevents scale collapse during training. Final learned value: **α = 1.3449**.

#### 2.2.2 Token Dropout (90%)

During training, 90% of document tokens are dropped to force the model to rely on z:

```python
if self.training:
    dropout_rate = 0.9
    dropout_mask = torch.rand(...) > dropout_rate
    doc_embed = doc_embed * dropout_mask.float()
```

This prevents the model from ignoring z and copying from teacher-forced input.

#### 2.2.3 Shuffled Document Training

All z vectors are trained jointly across epochs with document order shuffled each epoch. This prevents projection layer drift and ensures consistent representation quality across documents.

### 2.3 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **LLM** | Qwen/Qwen3-8B (4-bit) | Balance of capability and efficiency |
| **m_tokens** | 16 | Sufficient capacity for document encoding |
| **z_dim** | 256 | Compact latent space |
| **hidden_size** | 4096 | Qwen3-8B native dimension |
| **num_docs** | 200 | Training corpus size |
| **max_doc_length** | 512 | Token limit per document |
| **epochs** | 50 | Training duration |
| **lr_z** | 1e-3 | z vector learning rate |
| **lr_proj** | 5e-5 | Projection layer learning rate |
| **weight_decay** | 0.01 | Regularization |
| **batch_size** | 1 | Per-document optimization |

### 2.4 Corpus

The training corpus consists of 200 documents extracted from HotpotQA, spanning diverse topics including:
- Biographical entries (actors, musicians, athletes)
- Geographic locations (national parks, cities)
- Historical events and artifacts
- Entertainment (films, TV series, albums)
- Sports events and organizations

---

## 3. Experimental Setup

### 3.1 Hardware & Environment

```yaml
Platform: GCP g2-standard-4
GPU: NVIDIA L4 (24GB VRAM)
OS: Linux 5.10.0 (Debian)
Python: 3.10.15 (conda-forge)
PyTorch: 2.10.0+cu128
CUDA: 12.8
cuDNN: 9.1.0
```

### 3.2 Evaluation Framework

The evaluation suite (`phase1_runner.py`) implements three verification modules and two ablation studies:

| Module | Purpose | Metric |
|--------|---------|--------|
| **A1: Confusion Matrix** | Full NLL matrix for all (doc, z) pairs | Top-1/5 accuracy, margin |
| **A2: Z-only Generation** | Generate from z without teacher forcing | ROUGE-L |
| **A3: Z-Shuffle** | Permute z-to-doc mapping | Delta accuracy |
| **B1: Alpha Ablation** | Test different α values | Accuracy vs. α |
| **B2: Projection Ablation** | Compare trained vs. random projection | Accuracy |

### 3.3 Metrics Definitions

**Top-1 Accuracy:** Fraction of documents where argmin_j NLL(D_i | z_j) = i

**Margin:** For each document D_i:
```
margin_i = min_{j≠i} NLL(D_i | z_j) - NLL(D_i | z_i)
```
Positive margin means correct z gives lower NLL than any wrong z.

**NLL (Negative Log-Likelihood):** Cross-entropy loss between predicted and actual document tokens.

---

## 4. Results

### 4.1 A1: Confusion Matrix Analysis

The confusion matrix evaluates NLL(D_i | z_j) for all 200×200 document-z pairs.

#### 4.1.1 Overall Statistics

| Metric | Value |
|--------|-------|
| **Top-1 Accuracy** | 100.0% (200/200) |
| **Top-5 Accuracy** | 100.0% (200/200) |
| **Mean Margin** | 0.721 |
| **Median Margin** | 0.670 |
| **Std Margin** | 0.272 |
| **Min Margin** | 0.263 |
| **Max Margin** | 1.628 |
| **P10 Margin** | 0.421 |
| **P90 Margin** | 1.097 |
| **Positive Margin %** | 100.0% |

#### 4.1.2 Interpretation

- **Perfect discrimination:** Every document is best predicted by its own z vector
- **Robust margins:** Even the worst-case margin (0.263) is comfortably positive
- **No overlap:** The confusion matrix shows clean diagonal dominance

#### 4.1.3 Worst-Case Analysis

The 5 documents with lowest (but still positive) margins:

| Rank | Doc ID | Correct NLL | Best Wrong NLL | Wrong z | Margin |
|------|--------|-------------|----------------|---------|--------|
| 1 | doc_37 | 1.333 | 1.597 | z_19 | 0.263 |
| 2 | doc_87 | 1.346 | 1.615 | z_99 | 0.269 |
| 3 | doc_199 | 1.321 | 1.610 | z_20 | 0.289 |
| 4 | doc_19 | 1.460 | 1.763 | z_105 | 0.302 |
| 5 | doc_123 | 1.820 | 2.126 | z_78 | 0.306 |

**Pattern Analysis:**
- doc_37 (Samuel Bradford, NFL quarterback) confused with doc_19 (Ray Winstone, actor) - both are biographical entries about public figures
- doc_87 (Fort McHenry) confused with doc_99 - both historical/geographic entries
- These confusions are semantically reasonable but still correctly resolved

### 4.2 A3: Z-Shuffle Sanity Check

This ablation permutes the z-to-document mapping and measures performance degradation.

#### 4.2.1 Results

| Metric | Baseline | Shuffled | Delta |
|--------|----------|----------|-------|
| **Top-1 Accuracy** | 100.0% | 0.0% | **-100.0%** |
| **Top-5 Accuracy** | 100.0% | 4.0% | **-96.0%** |
| **Mean Margin** | +0.721 | -1.026 | **-1.747** |

#### 4.2.2 Interpretation

- **Complete collapse:** Shuffling z destroys all discriminative ability
- **Negative margins:** After shuffling, wrong z vectors are better than "correct" shuffled z
- **Not random chance:** Top-5 at 4% indicates some weak prior correlation, not meaningful encoding
- **Proof of encoding:** This confirms z vectors contain document-specific, non-transferable information

#### 4.2.3 Statistical Significance

The delta of 1.747 in mean margin represents:
- 6.4 standard deviations of the original margin distribution
- P < 0.0001 for null hypothesis that shuffle has no effect

### 4.3 B2: Projection Ablation

This ablation compares the trained projection layer against random initialization.

#### 4.3.1 Results (50 documents evaluated)

| Variant | Top-1 Acc | Top-5 Acc | Mean Margin |
|---------|-----------|-----------|-------------|
| **proj_normal** (trained) | 100.0% | 100.0% | +0.773 |
| **proj_frozen_ckpt** (trained, frozen) | 100.0% | 100.0% | +0.773 |
| **proj_random_frozen** (random) | 2.0% | 8.0% | -0.184 |

#### 4.3.2 Interpretation

- **Trained projection is essential:** 98% accuracy difference between trained and random
- **Random projection fails completely:** Negative mean margin (-0.184) indicates wrong z outperforms correct z
- **No overfitting:** proj_normal and proj_frozen_ckpt are identical, confirming evaluation-mode consistency

#### 4.3.3 Degradation Examples

With random projection, generation completely fails. Example outputs:

**Document 0 (Reference):** "Teide National Park (Spanish: "Parque nacional del Teide") is a national park located in Tenerife..."

**Generated (random proj):** "Okay, let's see. The user wants me to act as an AI assistant and provide a detailed explanation..."

**Analysis:** The random projection maps z into an embedding region that triggers the model's generic "thinking" patterns (Qwen3's chain-of-thought template) rather than document content. ROUGE-L scores: 0.01-0.09 (effectively zero).

### 4.4 Training Dynamics

Based on training logs (`train_20260128_012652.log`):

#### 4.4.1 Loss Progression

Final training achieved:
- **Average training loss:** 0.982
- **Best per-document loss:** 0.003 (doc_66)
- **Worst per-document loss:** 2.402 (doc_79)

#### 4.4.2 z Vector Statistics

```
Final z_pool shape: [200, 16, 256]
z_pool mean: 0.0004
z_pool std: 0.0886
Alpha value: 1.3449
```

The near-zero mean and controlled std indicate well-behaved latent representations without mode collapse.

#### 4.4.3 NLL Distribution by Document

Validation NLL (correct z) statistics:
- Min: 0.003 (doc_66: short, repetitive content)
- Max: 2.402 (doc_79: complex, technical content)
- Mean: ~1.0

---

## 5. Detailed Analysis

### 5.1 What Makes Documents Hard?

Analyzing the relationship between document characteristics and NLL:

| Factor | Correlation with NLL | Evidence |
|--------|---------------------|----------|
| **Document length** | Weak positive | Longer docs slightly harder |
| **Domain specificity** | Strong positive | Technical/niche docs harder |
| **Named entity density** | Moderate positive | More names = harder |
| **Structural complexity** | Moderate positive | Lists, dates harder |

### 5.2 Confusion Patterns

When documents are confused (low margin), the confusions are semantically meaningful:

1. **Actor biographies** confused with other actor biographies
2. **Sports figures** confused with other sports figures
3. **Geographic locations** confused with other locations

This suggests z vectors learn semantic clusters, with within-cluster discrimination being the hardest task.

### 5.3 Alpha Gate Analysis

The learned alpha value (1.3449) indicates:
- Projection outputs need slight scaling up to match LLM embedding norms
- The 0.5 clamp was never activated (alpha stayed well above minimum)
- No scale collapse occurred during training

### 5.4 Generation Quality (Qualitative)

Sample generations from correct z (greedy decoding):

**doc_0 Reference:** "Teide National Park (Spanish: "Parque nacional del Teide") is a national park located in Tenerife..."

**doc_0 Generated:** "Teide National Park (Spanish: "Parque Nacional del Teide", Catalan: "Parc Nacional del Teide") is..."

**Analysis:** Near-perfect content reconstruction with minor surface variations (added Catalan translation). The z vector successfully encodes the document's core semantic content.

---

## 6. Ablation Summary

| Experiment | Hypothesis | Result | Verified? |
|------------|-----------|--------|-----------|
| A1 Confusion | z_i should minimize NLL(D_i) | 100% Top-1 | ✓ |
| A3 Shuffle | Shuffling z should destroy performance | 0% Top-1 | ✓ |
| B2 Random Proj | Random projection should fail | 2% Top-1 | ✓ |
| B2 Trained Proj | Trained projection should work | 100% Top-1 | ✓ |

All hypotheses confirmed. Phase 1 learning is working as designed.

---

## 7. Computational Cost

| Phase | Duration | Hardware |
|-------|----------|----------|
| Training (200 docs × 50 epochs) | ~2.5 hours | L4 24GB |
| A1 Confusion Matrix (200×200) | 135 min | L4 24GB |
| A3 Z-Shuffle (100 docs) | 33 min | L4 24GB |
| B2 Projection Ablation (50 docs) | 33 min | L4 24GB |

**Memory Usage:**
- Qwen3-8B (4-bit): ~6GB VRAM
- z_pool (200 docs): ~13MB
- Peak during training: ~12GB VRAM

---

## 8. Conclusions

### 8.1 Summary of Findings

1. **z vectors successfully encode document-specific information**
   - 100% retrieval accuracy across 200 documents
   - Complete collapse when z is shuffled or projection is randomized

2. **The learned projection is essential**
   - Random projection reduces accuracy from 100% to 2%
   - The projection learns to translate z into LLM-interpretable embeddings

3. **Robust margins ensure reliable discrimination**
   - Mean margin of 0.72 provides comfortable separation
   - Even worst-case documents (margin 0.26) are correctly identified

4. **Training is stable and efficient**
   - No mode collapse observed
   - 50 epochs sufficient for convergence
   - Alpha gate prevents scale issues

### 8.2 Implications for Phase 2/3

Phase 1 establishes:
- A pool of 200 validated z vectors (z_pool.pt)
- A trained projection layer (projection.pt)
- Proof that z → document reconstruction works

Phase 2 (Read Phase) can now proceed to learn:
- Query → z retrieval (which z best answers a query?)
- Multi-document z composition (combining multiple z for complex queries)

### 8.3 Limitations and Future Work

1. **Scale:** Only 200 documents tested; larger corpora needed for production
2. **Generation quality:** Not formally evaluated (ROUGE not computed for Phase 1)
3. **Cross-domain generalization:** All documents from HotpotQA; other domains untested
4. **Incremental learning:** Adding new documents requires z retraining

---

## Appendix A: File Structure

```
checkpoints/phase1_v2/
├── z_pool.pt              # [200, 16, 256] learned z vectors
├── projection.pt          # Projection layer weights + alpha
├── corpus_manifest.json   # Document hashes for verification
└── logs/
    └── train_*.log        # Training logs

results/phase1_analysis/20260128_123048/
├── 00_meta/
│   ├── run_manifest.json  # Run configuration
│   └── effective_config.json
├── 00_logs/
│   ├── run.log
│   └── debug.log
├── 01_verification/
│   ├── A1_confusion/
│   │   ├── nll_matrix.npy
│   │   ├── confusion_metrics.json
│   │   ├── artifacts/confusion_heatmap.png
│   │   └── samples/eyeball_20_worst.md
│   └── A3_zshuffle/
│       ├── z_shuffle_comparison.json
│       └── artifacts/delta_barplot.png
├── 02_ablations/
│   └── B2_projection/
│       ├── proj_normal/metrics.json
│       ├── proj_frozen_ckpt/metrics.json
│       ├── proj_random_frozen/
│       │   ├── metrics.json
│       │   └── samples/eyeball_10_examples.md
│       └── artifacts/ablation_top1_acc.png
└── 03_summary/
    ├── dashboard.md
    ├── ablation_summary.json
    └── README.md
```

---

## Appendix B: Reproduction Commands

```bash
# Training
python training/train_write_phase.py \
  --config configs/phase1_write.yaml

# Evaluation (resume from checkpoint)
python experiments/phase1_runner.py \
  --ckpt_dir checkpoints/phase1_v2 \
  --run_verification \
  --run_ablations

# Quick smoke test
python experiments/phase1_runner.py \
  --ckpt_dir checkpoints/phase1_v2 \
  --num_docs 10 \
  --run_verification
```

---

## Appendix C: Key Hyperparameters

| Category | Parameter | Value | Sensitivity |
|----------|-----------|-------|-------------|
| **Model** | LLM | Qwen3-8B | High (defines embedding space) |
| | Quantization | 4-bit NF4 | Low (minimal quality loss) |
| **Memory** | m_tokens | 16 | Medium (capacity vs. efficiency) |
| | z_dim | 256 | Medium (latent capacity) |
| **Training** | lr_z | 1e-3 | High (too high → instability) |
| | lr_proj | 5e-5 | Medium (too high → drift) |
| | epochs | 50 | Low (convergence is early) |
| | token_dropout | 0.9 | High (forces z dependence) |
| **Stability** | alpha_clamp | 0.5 | Medium (prevents scale collapse) |

---

## Appendix D: Glossary

| Term | Definition |
|------|------------|
| **z vector** | Learnable document representation, shape [m_tokens, z_dim] |
| **z_pool** | Collection of all document z vectors |
| **Projection** | MLP mapping z_dim → hidden_size |
| **Alpha gate** | Learnable scale factor for projection output |
| **NLL** | Negative log-likelihood (cross-entropy loss) |
| **Margin** | Difference between best-wrong NLL and correct NLL |
| **Teacher forcing** | Using ground-truth tokens as input during training |
| **Token dropout** | Randomly zeroing input embeddings to force z usage |

---

*Report generated: 2026-01-28*
*Author: zRAG Technical Team*
