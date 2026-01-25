#!/bin/bash
# Phase 3: Full Evaluation
# 목표: 7개 Baseline과 공정 비교 (5일)

set -e

echo "============================================"
echo "Phase 3: Full Evaluation"
echo "============================================"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG="configs/phase3_full.yaml"
LOG_DIR="logs/phase3"
mkdir -p "$LOG_DIR"

# Step 1: Write Phase (large corpus)
echo "[Step 1] Write Phase (50K documents)..."
python training/train_write.py --config "$CONFIG" 2>&1 | tee "$LOG_DIR/write.log"

# Step 2: Read Phase
echo "[Step 2] Read Phase..."
python training/train_read.py \
    --config "$CONFIG" \
    --write_checkpoint "checkpoints/phase3/write_final.pt" \
    2>&1 | tee "$LOG_DIR/read.log"

# Step 3: All Baselines
echo "[Step 3] Running all baselines..."
python baselines/run_baselines.py --config "$CONFIG" 2>&1 | tee "$LOG_DIR/baselines.log"

# Step 4: Multi-seed evaluation (3 seeds)
echo "[Step 4] Multi-seed evaluation..."
for seed in 42 123 456; do
    echo "  Seed=$seed"
    python training/train_e2e.py \
        --config "$CONFIG" \
        --override "logging.run_name=phase3-seed-$seed" \
        --seed "$seed" \
        2>&1 | tee "$LOG_DIR/seed_$seed.log"
done

echo ""
echo "Phase 3 Complete!"
echo "Results: $LOG_DIR/"
echo "Checkpoints: checkpoints/phase3/"
