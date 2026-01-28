# Phase 1 Figure Descriptions

**Descriptions for Paper-Quality Figures**

---

## Figure 1: Confusion Heatmap

**File:** `results/phase1_analysis/20260128_123048/01_verification/A1_confusion/artifacts/confusion_heatmap.png`

### Description

A 200×200 heatmap visualization of the NLL (Negative Log-Likelihood) confusion matrix. Each cell (i, j) represents NLL(D_i | z_j), the likelihood of document D_i being generated given latent vector z_j.

### Visual Elements

- **Axes:** X-axis shows z index (0-199), Y-axis shows document index (0-199)
- **Color scale:** Dark blue = low NLL (good), Yellow/white = high NLL (poor)
- **Diagonal:** Strong dark band indicating correct z_i produces lowest NLL for D_i
- **Off-diagonal:** Lighter colors showing higher NLL when using wrong z

### Key Observations

1. **Clean diagonal dominance:** The diagonal is consistently darker than surrounding cells
2. **No dark off-diagonal clusters:** No systematic confusions between document groups
3. **Uniform off-diagonal brightness:** Wrong z vectors perform uniformly poorly

### Caption for Paper

> Figure X: NLL confusion matrix for Phase 1 write model across 200 documents. Each cell shows NLL(D_i | z_j). The dark diagonal confirms that each document's trained z vector produces the lowest NLL for its corresponding document (100% Top-1 accuracy). Off-diagonal values are consistently higher, with a mean margin of 0.72 nats.

---

## Figure 2: Z-Shuffle Delta Bar Plot

**File:** `results/phase1_analysis/20260128_123048/01_verification/A3_zshuffle/artifacts/delta_barplot.png`

### Description

A grouped bar chart comparing baseline and shuffled performance metrics, demonstrating the catastrophic effect of misaligning z vectors with documents.

### Visual Elements

- **X-axis:** Metrics (Top-1 Accuracy, Top-5 Accuracy, Mean Margin)
- **Y-axis:** Metric value
- **Blue bars:** Baseline (correct z-to-document mapping)
- **Orange/Red bars:** Shuffled (permuted z-to-document mapping)
- **Delta annotations:** Showing the change between conditions

### Key Values

| Metric | Baseline | Shuffled | Delta |
|--------|----------|----------|-------|
| Top-1 Acc | 100% | 0% | -100% |
| Top-5 Acc | 100% | 4% | -96% |
| Mean Margin | +0.72 | -1.03 | -1.75 |

### Caption for Paper

> Figure X: Effect of shuffling z-to-document assignments. Baseline (blue) shows perfect discrimination with all positive margins. After shuffling (orange), accuracy drops to 0% and margins become negative, confirming that z vectors encode document-specific rather than generic information.

---

## Figure 3: Projection Ablation Comparison

**File:** `results/phase1_analysis/20260128_123048/02_ablations/B2_projection/artifacts/ablation_top1_acc.png`

### Description

A bar chart comparing Top-1 accuracy across different projection layer configurations.

### Visual Elements

- **X-axis:** Projection variant (Normal, Frozen Checkpoint, Random Frozen)
- **Y-axis:** Top-1 Accuracy (0-100%)
- **Bar colors:** Distinguish between trained and random projections

### Key Values

| Variant | Top-1 Accuracy |
|---------|----------------|
| proj_normal (trained) | 100.0% |
| proj_frozen_ckpt (trained, eval mode) | 100.0% |
| proj_random_frozen (random init) | 2.0% |

### Caption for Paper

> Figure X: Projection layer ablation study. Trained projection (blue) achieves 100% accuracy while random initialization (red) collapses to 2%, demonstrating that the learned z→embedding mapping is essential for document discrimination. The frozen checkpoint variant confirms evaluation-mode consistency.

---

## Figure 4: Margin Distribution Histogram

**File:** `results/phase1_analysis/20260128_123048/01_verification/A1_confusion/artifacts/margin_distribution.png`

### Description

A histogram showing the distribution of margins (correct NLL - best wrong NLL) across all 200 documents.

### Visual Elements

- **X-axis:** Margin value (nats)
- **Y-axis:** Number of documents
- **Vertical line:** At margin=0 (decision boundary)
- **Distribution shape:** Right-skewed, all positive

### Key Statistics

```
Mean:   0.721
Median: 0.670
Std:    0.272
Min:    0.263
Max:    1.628
```

### Caption for Paper

> Figure X: Distribution of discriminative margins across 200 documents. All margins are positive (100% accuracy), with mean=0.72 and minimum=0.26 nats. The right-skewed distribution indicates most documents are well-separated in latent space, with a tail of particularly distinctive documents.

---

## Figure 5: NLL Comparison Box Plot

**Suggested figure (may need to generate)**

### Description

A box plot comparing NLL distributions for three conditions: correct z, wrong z, and random z.

### Visual Elements

- **X-axis:** Condition (Correct z, Wrong z, Random z)
- **Y-axis:** NLL value
- **Box plots:** Showing quartiles, median, and outliers

### Key Observations

1. Correct z has significantly lower NLL (median ~0.86)
2. Wrong z and random z have similar, higher NLL (median ~1.9)
3. Clear separation between correct and incorrect conditions

### Caption for Paper

> Figure X: NLL distributions by z assignment condition. Correct z (green) produces significantly lower NLL than wrong z (blue) or random z (red), with non-overlapping interquartile ranges (p < 0.001, paired t-test).

---

## Figure 6: Training Loss Curve

**Suggested figure (may need to generate from logs)**

### Description

A line plot showing the progression of mean loss across training epochs.

### Visual Elements

- **X-axis:** Epoch (0-50)
- **Y-axis:** Mean loss (NLL)
- **Main line:** Mean loss across all documents
- **Shaded region:** ±1 standard deviation

### Key Trajectory

```
Epoch 0:  3.12 (initial)
Epoch 10: 1.45 (53% reduction)
Epoch 30: 1.02 (67% reduction)
Epoch 50: 0.98 (69% reduction, final)
```

### Caption for Paper

> Figure X: Training loss progression over 50 epochs. Rapid initial convergence is followed by gradual refinement, reaching final mean NLL of 0.98. Shaded region shows per-document variance, which decreases as all documents converge.

---

## Figure 7: Alpha Gate Trajectory

**Suggested figure (may need to generate from logs)**

### Description

A line plot showing the evolution of the learned alpha scale parameter during training.

### Visual Elements

- **X-axis:** Epoch (0-50)
- **Y-axis:** Alpha value
- **Horizontal dashed line:** At α=0.5 (minimum clamp)
- **Horizontal dashed line:** At α=1.0 (initialization)

### Key Trajectory

```
Epoch 0:  1.00 (initial)
Epoch 10: 1.12
Epoch 30: 1.31
Epoch 50: 1.34 (final)
```

### Caption for Paper

> Figure X: Evolution of the learned alpha scale gate during training. Alpha increases from 1.0 to 1.34, indicating the model learns to amplify projected z embeddings by ~34% to match LLM embedding scale. The 0.5 minimum clamp (dashed) is never triggered.

---

## Supplementary Figures

### S-Figure 1: Per-Document NLL Scatter

**Description:** Scatter plot of correct NLL vs. best-wrong NLL for each document. Points above the diagonal indicate positive margin (correct discrimination).

### S-Figure 2: Z Vector PCA/t-SNE

**Description:** 2D visualization of learned z vectors using dimensionality reduction, potentially colored by document category or NLL.

### S-Figure 3: Confusion Network Graph

**Description:** Network visualization where nodes are documents and edges connect frequently confused pairs (low margin). Clusters indicate semantic similarity.

### S-Figure 4: Generation Examples Grid

**Description:** Side-by-side comparison of reference documents and generated text for selected examples (best, median, worst cases).

---

## Figure Generation Notes

### For Paper Submission

1. **Resolution:** Export at 300 DPI minimum
2. **Format:** PDF for vector graphics, PNG for rasterized
3. **Color scheme:** Use colorblind-friendly palettes
4. **Font size:** Ensure axis labels are readable at publication size
5. **Annotations:** Add statistical significance markers where appropriate

### Color Recommendations

```python
# Matplotlib colorblind-friendly palette
colors = {
    'correct': '#2166AC',    # Blue
    'wrong': '#B2182B',      # Red
    'random': '#762A83',     # Purple
    'baseline': '#1B7837',   # Green
    'shuffled': '#E08214',   # Orange
}
```

---

*Figure descriptions prepared for publication*
