# Phase 1 Analysis Dashboard

Generated: 2026-01-29T17:15:03.075569
Run directory: `20260129_170900`

---

## Top 5 Artifacts to Check First

### 1. Confusion Heatmap
**Path:** `01_verification/A1_confusion/artifacts/confusion_heatmap.png`

**What to look for:**
- Diagonal should be dark (low NLL = correct z works well)
- Off-diagonal should be bright (high NLL = wrong z produces poor predictions)
- Any dark off-diagonal clusters indicate potentially confused document pairs

### 2. Worst Cases Report
**Path:** `01_verification/A1_confusion/samples/eyeball_20_worst.md`

**What to look for:**
- Review 20 docs with lowest margin between correct and best-wrong z
- Are the confusions reasonable (similar documents)?
- Are there any negative margins (failure cases)?

### 3. Z-only Best Examples
**Path:** `01_verification/A2_zonly/samples/eyeball_20_best.md`

**What to look for:**
- Do high-ROUGE generations actually capture the document content?
- Is the generated text coherent and factually similar to reference?

### 4. Z-only Worst Examples
**Path:** `01_verification/A2_zonly/samples/eyeball_20_worst.md`

**What to look for:**
- What goes wrong in low-ROUGE cases?
- Is it hallucination, wrong topic, or just different wording?

### 5. Random Projection Degradation
**Path:** `02_ablations/B2_projection/proj_random_frozen/samples/eyeball_10_examples.md`

**What to look for:**
- Confirm outputs are garbage when projection is random
- This validates that the trained projection is essential

---

## Manual Sample Count

| Category | Count | File |
|----------|-------|------|
| Confusion worst cases | 20 | `A1_confusion/samples/eyeball_20_worst.md` |
| Z-only best | 20 | `A2_zonly/samples/eyeball_20_best.md` |
| Z-only worst | 20 | `A2_zonly/samples/eyeball_20_worst.md` |
| Random projection degradation | 10 | `B2_projection/proj_random_frozen/samples/eyeball_10_examples.md` |
| **Total** | **~70** | |

---

## Quick Metrics Summary

| Module | Key Metric | Value | Status |
|--------|------------|-------|--------|
| A2 Z-only | Mean ROUGE-L | 0.238 | WARN |

---

## File Structure

```
20260129_170900/
├── 00_meta/              # Run metadata and config
├── 00_logs/              # Log files (run.log, debug.log, warnings.log)
├── 01_verification/      # Verification modules (A1, A2, A3)
│   ├── A1_confusion/     # NLL confusion matrix
│   ├── A2_zonly/         # Z-only generation
│   └── A3_zshuffle/      # Z-shuffle sanity check
├── 02_ablations/         # Ablation studies (B1, B2)
│   ├── B1_alpha/         # Alpha ablation
│   └── B2_projection/    # Projection ablation
├── 03_summary/           # This dashboard and summary files
└── 04_cache/             # Partial computation cache
```
