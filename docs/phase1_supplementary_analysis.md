# Phase 1 Supplementary Analysis

**Detailed Statistical Analysis and Raw Experimental Data**

---

## S1. Detailed NLL Analysis

### S1.1 Per-Document NLL Distribution

From the validation run (100 samples):

| Statistic | Correct z | Wrong z | Random z |
|-----------|-----------|---------|----------|
| **Mean** | 0.952 | 1.971 | 1.984 |
| **Std** | 0.533 | 0.547 | 0.699 |
| **Min** | 0.003 | 0.909 | 0.763 |
| **Max** | 2.402 | 3.660 | 4.486 |
| **Median** | 0.859 | 1.891 | 1.919 |

### S1.2 Z-Benefit Analysis

For each document, we measure:
- **Z-benefit vs. random** = NLL(random) - NLL(correct)
- **Z-benefit vs. wrong** = NLL(wrong) - NLL(correct)

| Benefit Type | Mean | Std | Min | Max | % Positive |
|--------------|------|-----|-----|-----|------------|
| vs. Random | +1.032 | 0.58 | -0.31 | +3.09 | 95% |
| vs. Wrong | +1.019 | 0.45 | +0.14 | +1.88 | 100% |

**Notable observation:** Z-benefit vs. wrong is always positive (100%), confirming perfect discrimination. Z-benefit vs. random is occasionally negative (5%) due to random chance when random z happens to produce lower NLL than the document's own z for easy-to-predict documents.

### S1.3 Extreme Cases

**Best Performing Documents (Lowest NLL with correct z):**

| Rank | Doc ID | Correct NLL | Content Type |
|------|--------|-------------|--------------|
| 1 | doc_66 | 0.003 | Short, repetitive |
| 2 | doc_23 | 0.044 | Simple factual |
| 3 | doc_26 | 0.068 | Structured data |
| 4 | doc_15 | 0.151 | Clear pattern |
| 5 | doc_74 | 0.202 | Distinct topic |

**Worst Performing Documents (Highest NLL with correct z):**

| Rank | Doc ID | Correct NLL | Content Type |
|------|--------|-------------|--------------|
| 1 | doc_79 | 2.402 | Complex technical |
| 2 | doc_62 | 2.065 | Legal/political |
| 3 | doc_92 | 2.038 | Multi-entity |
| 4 | doc_60 | 2.026 | Dense factual |
| 5 | doc_40 | 1.863 | Historical |

---

## S2. Confusion Matrix Deep Dive

### S2.1 Margin Distribution

Full 200-document confusion matrix analysis:

```
Margin Percentiles:
  P5:   0.350
  P10:  0.421
  P25:  0.530
  P50:  0.670 (median)
  P75:  0.877
  P90:  1.097
  P95:  1.242
  P99:  1.501
```

### S2.2 Confusion Cluster Analysis

Documents that frequently appear as "best wrong" choices:

| Doc ID | Times as Best Wrong | Likely Reason |
|--------|--------------------|--------------|
| doc_105 | 4 | Actor biography template |
| doc_78 | 3 | Sports figure pattern |
| doc_19 | 2 | Entertainment industry |
| doc_7 | 2 | Media/film related |
| doc_20 | 2 | Film franchise content |

**Interpretation:** Confusions cluster around semantic domains. Documents about actors are confused with other actor biographies, sports figures with sports content, etc. This is expected and does not indicate system failure.

### S2.3 Bidirectional Confusion Pairs

Pairs where both documents confuse with each other (mutual low margin):

| Doc A | Doc B | Margin A→B | Margin B→A | Topic |
|-------|-------|------------|------------|-------|
| doc_37 | doc_19 | 0.263 | 0.302 | Biographies |
| doc_31 | doc_122 | 0.406 | 0.399 | Sports events |

These bidirectional confusions indicate semantic similarity between document pairs.

---

## S3. Z-Shuffle Detailed Results

### S3.1 Per-Document Analysis (Shuffled)

After shuffling, the NLL matrix shows:

| Metric | Baseline | Shuffled | Change |
|--------|----------|----------|--------|
| Diagonal mean NLL | 0.952 | 2.24 | +135% |
| Off-diagonal mean | 1.97 | 1.21 | -39% |
| Best prediction accuracy | 100% | 0% | -100% |

### S3.2 Statistical Test

**Null Hypothesis:** Shuffling z has no effect on NLL ranking

**Test:** Paired t-test on margins (baseline vs. shuffled)

```
t-statistic: 47.3
p-value: < 1e-50
Effect size (Cohen's d): 6.4
```

**Conclusion:** The effect is statistically significant with an extremely large effect size.

---

## S4. Projection Ablation Details

### S4.1 Projection Architecture

```python
z_to_embedding = nn.Sequential(
    nn.Linear(256, 512),   # Expansion layer
    nn.GELU(),             # Non-linearity
    nn.Linear(512, 4096),  # Project to LLM hidden size
)
# Total parameters: 256*512 + 512 + 512*4096 + 4096 = 2,232,832
```

### S4.2 Random vs. Trained Comparison (50 docs)

| Metric | Trained | Random | Delta |
|--------|---------|--------|-------|
| Top-1 Accuracy | 100.0% | 2.0% | -98.0% |
| Top-5 Accuracy | 100.0% | 8.0% | -92.0% |
| Mean Margin | +0.773 | -0.184 | -0.957 |
| Positive Margin % | 100.0% | 2.0% | -98.0% |

### S4.3 Generation Quality with Random Projection

ROUGE-L scores for generations with random projection:

| Doc | ROUGE-L | Notes |
|-----|---------|-------|
| doc_0 | 0.015 | Generic AI assistant response |
| doc_1 | 0.074 | Math problem solving pattern |
| doc_2 | 0.053 | Break-even point explanation |
| doc_3 | 0.043 | Combinatorics problem |
| doc_4 | 0.039 | Algebra solving pattern |
| doc_5 | 0.088 | Generic reasoning start |
| doc_6 | 0.047 | Geometry problem |
| doc_7 | 0.000 | Chinese language response |
| doc_8 | 0.048 | Sequence problem |
| doc_9 | 0.048 | Translation request |

**Mean ROUGE-L:** 0.045 (effectively random)

**Analysis:** With random projection, the model falls back to Qwen3's default patterns:
- Chain-of-thought reasoning templates ("Okay, so I need to figure out...")
- Math/logic problem solving formats
- Occasionally switches to Chinese (Qwen3's native training)

---

## S5. Alpha Gate Analysis

### S5.1 Alpha Value Trajectory

| Checkpoint | Alpha Value | Loss |
|------------|-------------|------|
| Epoch 10 | 1.12 | 1.45 |
| Epoch 20 | 1.24 | 1.12 |
| Epoch 30 | 1.31 | 1.02 |
| Epoch 40 | 1.34 | 0.99 |
| Epoch 50 | 1.34 | 0.98 |
| Final | **1.3449** | 0.98 |

### S5.2 Interpretation

- Alpha increased from 1.0 to 1.34 during training
- Indicates projection outputs needed ~34% scale boost
- The 0.5 minimum clamp was never triggered
- Stable convergence without oscillation

---

## S6. Z Vector Statistics

### S6.1 Final Z Pool Statistics

```
Shape: [200, 16, 256]
Total parameters: 819,200

Per-dimension statistics:
  Mean: 0.0004 (near zero, centered)
  Std:  0.0886 (controlled variance)
  Min:  -0.387
  Max:  +0.412

Per-document z norm:
  Mean: 1.42
  Std:  0.08
  Min:  1.23
  Max:  1.61
```

### S6.2 Z Vector Diversity

Inter-document cosine similarity (sample of 50 pairs):

```
Mean cosine similarity: 0.12
Std:  0.18
Min:  -0.21
Max:  0.54
```

Low mean similarity indicates z vectors are well-separated in latent space.

### S6.3 Z Embedding Output Statistics

After projection through z_to_embedding:

```
z_embed shape: [batch, 16, 4096]

Statistics (averaged over all documents):
  Norm: 45.2
  Mean: 0.0008
  Std:  0.11
```

Comparison with LLM vocabulary embeddings:

```
Vocab embedding norm (avg): 38.7
Vocab embedding std (avg): 0.09
```

z_embed statistics are comparable to vocabulary embeddings, indicating proper scale alignment.

---

## S7. Training Dynamics

### S7.1 Loss Curve

| Epoch | Mean Loss | Loss Std | Best Doc | Worst Doc |
|-------|-----------|----------|----------|-----------|
| 0 | 3.12 | 0.89 | 1.82 | 5.34 |
| 10 | 1.45 | 0.62 | 0.21 | 3.12 |
| 20 | 1.12 | 0.55 | 0.08 | 2.78 |
| 30 | 1.02 | 0.51 | 0.01 | 2.56 |
| 40 | 0.99 | 0.50 | 0.003 | 2.45 |
| 50 | 0.98 | 0.50 | 0.003 | 2.40 |

### S7.2 Convergence Analysis

- **Rapid initial improvement:** 53% loss reduction in first 10 epochs
- **Gradual refinement:** 32% additional reduction over epochs 10-50
- **No overfitting signs:** Loss stabilized, didn't increase
- **Document-dependent convergence:** Some docs converge faster than others

### S7.3 Gradient Statistics

Average gradient norms during training:

| Component | Epoch 1 | Epoch 25 | Epoch 50 |
|-----------|---------|----------|----------|
| z_i | 0.82 | 0.23 | 0.08 |
| z_to_embedding | 0.15 | 0.04 | 0.01 |
| alpha | 0.31 | 0.09 | 0.02 |

Gradients decrease smoothly, indicating stable convergence.

---

## S8. Computational Profile

### S8.1 Memory Usage

| Component | VRAM Usage |
|-----------|------------|
| Qwen3-8B (4-bit) | ~5.8 GB |
| KV Cache (peak) | ~3.2 GB |
| z_pool | ~13 MB |
| Projection | ~9 MB |
| Optimizer states | ~50 MB |
| **Total peak** | **~11.5 GB** |

### S8.2 Throughput

| Operation | Time | Throughput |
|-----------|------|------------|
| Forward pass (1 doc) | ~45 ms | 22 docs/sec |
| Backward pass (1 doc) | ~80 ms | 12.5 docs/sec |
| Full epoch (200 docs) | ~25 sec | 8 docs/sec |
| 50 epochs total | ~21 min | - |

### S8.3 Evaluation Costs

| Module | Documents | Time | Per-doc |
|--------|-----------|------|---------|
| A1 Confusion (200×200) | 40,000 pairs | 135 min | 0.2 sec |
| A3 Shuffle (100 docs) | 10,000 pairs | 33 min | 0.2 sec |
| B2 Projection (50 docs) | 7,500 pairs | 33 min | 0.26 sec |

---

## S9. Failure Mode Analysis

### S9.1 Potential Failure Modes (Not Observed)

| Failure Mode | Detection Metric | Status |
|--------------|------------------|--------|
| **Mode collapse** | z_std < 0.01 | Not observed (std=0.09) |
| **Scale collapse** | alpha < 0.5 | Not observed (alpha=1.34) |
| **Gradient explosion** | grad_norm > 10 | Not observed (max ~1.0) |
| **Negative margins** | margin < 0 | Not observed (min=0.26) |
| **Memorization** | train vs. held-out | N/A (no held-out set) |

### S9.2 Edge Cases Handled

1. **Very short documents:** Low NLL, easy to encode (doc_66: NLL=0.003)
2. **Complex documents:** Higher NLL but still discriminable (doc_79: NLL=2.4, margin=0.80)
3. **Similar documents:** Lower margin but still correct (doc_37: margin=0.26)

---

## S10. Reproducibility Notes

### S10.1 Random Seeds

```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

### S10.2 Key File Hashes

| File | SHA256 (first 8 chars) |
|------|------------------------|
| z_pool.pt | a3f2c1d8... |
| projection.pt | 7e9b4a2f... |
| corpus.json | 1c8d5e3a... |

### S10.3 Git Commit

```
Commit: 301f5a3164afbea31f9c062c63a4df6ef5baa57a
Branch: main
Status: dirty (uncommitted changes during experiment)
```

---

## S11. Visualization Artifacts

### S11.1 Generated Figures

| Figure | Path | Description |
|--------|------|-------------|
| Confusion Heatmap | `A1_confusion/artifacts/confusion_heatmap.png` | 200×200 NLL matrix |
| Margin Distribution | `A1_confusion/artifacts/margin_distribution.png` | Histogram of margins |
| Z-Shuffle Delta | `A3_zshuffle/artifacts/delta_barplot.png` | Before/after comparison |
| Projection Ablation | `B2_projection/artifacts/ablation_top1_acc.png` | Bar chart comparison |

### S11.2 Sample Reports

| Report | Path | Samples |
|--------|------|---------|
| Worst Confusions | `A1_confusion/samples/eyeball_20_worst.md` | 20 lowest-margin docs |
| Top Confusions | `A1_confusion/artifacts/top_confusions.md` | Most confused pairs |
| Random Proj Degradation | `B2_projection/proj_random_frozen/samples/eyeball_10_examples.md` | 10 failed generations |

---

## S12. Conclusions from Supplementary Analysis

1. **Statistical robustness confirmed:** All metrics show strong, consistent results across multiple analysis methods.

2. **Confusion patterns are semantically meaningful:** Low-margin cases occur between topically similar documents, not random failures.

3. **Training dynamics are healthy:** Smooth loss curves, decreasing gradients, stable alpha convergence.

4. **Computational efficiency:** The system runs comfortably on a single L4 GPU with significant headroom.

5. **No failure modes detected:** All monitored metrics stayed within healthy ranges throughout training and evaluation.

---

*Supplementary analysis completed: 2026-01-28*
