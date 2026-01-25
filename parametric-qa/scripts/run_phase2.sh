#!/bin/bash
# Phase 2: Ablation Studies
# 목표: 최적 하이퍼파라미터 탐색 (4일)

set -e

echo "============================================"
echo "Phase 2: Ablation Studies"
echo "============================================"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG="configs/phase2_ablation.yaml"
LOG_DIR="logs/phase2"
mkdir -p "$LOG_DIR"

# Ablation 1: z_dim
echo "[Ablation 1] z_dim sweep: {64, 128, 256, 512, 1024}"
for z_dim in 64 128 256 512 1024; do
    echo "  z_dim=$z_dim"
    python training/train_e2e.py \
        --config "$CONFIG" \
        --override "parametric_qa.z_dim=$z_dim" \
        --override "logging.run_name=ablation-zdim-$z_dim" \
        2>&1 | tee "$LOG_DIR/zdim_$z_dim.log"
done

# Ablation 2: m_tokens
echo "[Ablation 2] m_tokens sweep: {1, 4, 8, 16, 32}"
for m_tokens in 1 4 8 16 32; do
    echo "  m_tokens=$m_tokens"
    python training/train_e2e.py \
        --config "$CONFIG" \
        --override "parametric_qa.m_tokens=$m_tokens" \
        --override "logging.run_name=ablation-mtokens-$m_tokens" \
        2>&1 | tee "$LOG_DIR/mtokens_$m_tokens.log"
done

# Ablation 3: Selection method
echo "[Ablation 3] Selection method: {cosine, learned_router, attention}"
for method in cosine learned_router attention; do
    echo "  method=$method"
    python training/train_e2e.py \
        --config "$CONFIG" \
        --override "parametric_qa.selection_method=$method" \
        --override "logging.run_name=ablation-selection-$method" \
        2>&1 | tee "$LOG_DIR/selection_$method.log"
done

# Ablation 4: LoRA rank
echo "[Ablation 4] LoRA rank: {frozen, 8, 32, 64}"
# Frozen (no LoRA, z only)
python training/train_e2e.py \
    --config "$CONFIG" \
    --override "write_phase.llm_frozen=true" \
    --override "logging.run_name=ablation-frozen" \
    2>&1 | tee "$LOG_DIR/lora_frozen.log"

for rank in 8 32 64; do
    echo "  LoRA rank=$rank"
    python training/train_e2e.py \
        --config "$CONFIG" \
        --override "model.lora.r=$rank" \
        --override "model.lora.alpha=$((rank*2))" \
        --override "logging.run_name=ablation-lora-r$rank" \
        2>&1 | tee "$LOG_DIR/lora_r$rank.log"
done

echo ""
echo "Phase 2 Complete!"
echo "Results: $LOG_DIR/"
