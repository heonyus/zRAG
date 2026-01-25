#!/bin/bash
# Phase 1: Proof of Concept
# 목표: z_i learning 작동 확인 (3일)

set -e

echo "============================================"
echo "Phase 1: Proof of Concept"
echo "============================================"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG="configs/phase1_poc.yaml"
LOG_DIR="logs/phase1"
mkdir -p "$LOG_DIR"

# Step 1: Data Download
echo "[Step 1] Downloading dataset..."
python data/download.py --dataset natural_questions --save_dir data/raw

# Step 2: End-to-End Training (Write + Read)
echo "[Step 2] Running end-to-end training..."
python training/train_e2e.py --config "$CONFIG" 2>&1 | tee "$LOG_DIR/train.log"

# Step 3: Baselines (minimal: No Retrieval + Standard RAG)
echo "[Step 3] Running baselines..."
python baselines/run_baselines.py --config "$CONFIG" --max_samples 500 2>&1 | tee "$LOG_DIR/baselines.log"

echo ""
echo "Phase 1 Complete!"
echo "Results: $LOG_DIR/"
echo "Checkpoints: checkpoints/phase1/"
