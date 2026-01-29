# Phase 1 Verification & Ablation Results

This directory contains the results of Phase 1 (Write Phase) verification and ablation experiments.

## How to Run

```bash
# Full run with verification + ablations
python experiments/phase1_runner.py \
  --ckpt_dir checkpoints/phase1_v2 \
  --out_root results/phase1_analysis \
  --run_verification \
  --run_ablations

# Quick smoke test (10 docs)
python experiments/phase1_runner.py \
  --ckpt_dir checkpoints/phase1_v2 \
  --out_root results/phase1_analysis \
  --num_docs 10 \
  --run_verification
```

## Interpretation Guide

### A1: Confusion Matrix

- **Top-1 Accuracy**: Percentage of documents where the correct z produces the lowest NLL
  - Target: >95% (100% is ideal)
- **Mean Margin**: Average difference between best-wrong NLL and correct NLL
  - Target: >0.5 (higher is better)

### A2: Z-only Generation

- **Mean ROUGE-L**: Average ROUGE-L score between generated and reference text
  - Target: >0.3 for meaningful generation
- Check eyeball samples to verify generations capture document semantics

### A3: Z-shuffle

- **Delta Top-1**: Drop in top-1 accuracy after shuffling z-to-doc mapping
  - Target: Large drop (>50%) proves z encodes document-specific information

### B1: Alpha Ablation

- **alpha=0**: Should have near-random performance (proves alpha scaling is needed)
- **alpha=1**: Baseline without learned scaling
- **alpha=trained**: Should be best

### B2: Projection Ablation

- **Random projection**: Should have near-random performance (proves projection is learned)

## Key Files

- `dashboard.md`: Quick overview with links to top artifacts
- `ablation_summary.csv`: One row per variant with all metrics
- `ablation_summary.json`: Same data in JSON format

## Eyeball Samples

Manual inspection of ~70 samples is recommended:

1. `A1_confusion/samples/eyeball_20_worst.md` - Documents with lowest confidence
2. `A2_zonly/samples/eyeball_20_best.md` - Best z-only generations
3. `A2_zonly/samples/eyeball_20_worst.md` - Worst z-only generations
4. `B2_projection/proj_random_frozen/samples/eyeball_10_examples.md` - Random projection failures

## Run Configuration

```json
{'ckpt_dir': 'checkpoints/phase1_v2', 'out_root': 'results/phase1_analysis', 'resume_dir': None, 'num_docs': 20, 'max_eval_tokens': 256, 'max_new_tokens': 128, 'device': 'cuda', 'seed': 42, 'no_amp': False, 'run_verification': True, 'run_ablations': False, 'skip_a1': True, 'skip_a2': False, 'skip_a3': True, 'skip_b1': False, 'skip_b2': False, 'run_proj_only_baseline': False, 'sweep_dropout': False, 'sweep_mtokens': False, 'corpus_path': None}
```
