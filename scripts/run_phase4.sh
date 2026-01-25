#!/bin/bash
# Phase 4: Multi-hop Extension
# 목표: HotpotQA에서의 성능 및 한계 분석 (3일)

set -e

echo "============================================"
echo "Phase 4: Multi-hop Extension (HotpotQA)"
echo "============================================"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG="configs/phase4_multihop.yaml"
LOG_DIR="logs/phase4"
mkdir -p "$LOG_DIR"

# Step 1: Train on HotpotQA
echo "[Step 1] Training on HotpotQA..."
python training/train_e2e.py --config "$CONFIG" 2>&1 | tee "$LOG_DIR/train.log"

# Step 2: Multi-hop strategies comparison
echo "[Step 2] Multi-hop strategies..."
for strategy in concat iterative; do
    echo "  Strategy: $strategy"
    python training/train_e2e.py \
        --config "$CONFIG" \
        --override "multihop.active_strategy=$strategy" \
        --override "logging.run_name=phase4-$strategy" \
        2>&1 | tee "$LOG_DIR/strategy_$strategy.log"
done

# Step 3: Baselines on HotpotQA
echo "[Step 3] Baselines on HotpotQA..."
python baselines/run_baselines.py --config "$CONFIG" 2>&1 | tee "$LOG_DIR/baselines.log"

echo ""
echo "Phase 4 Complete!"
echo "Results: $LOG_DIR/"
