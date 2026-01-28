#!/bin/bash
# Phase 1 정리 스크립트
# 실행: bash scripts/cleanup_phase1.sh

set -e

cd /home/lhe339/data/zRAG

echo "=== Phase 1 Cleanup Script ==="

# 1. Stage 2 백업 삭제 (23MB 확보)
if [ -d "checkpoints/phase1_write_stage2_backup_0127_0821" ]; then
    echo "[1/4] Removing old Stage 2 backup..."
    rm -rf checkpoints/phase1_write_stage2_backup_0127_0821/
    echo "      Deleted: checkpoints/phase1_write_stage2_backup_0127_0821/"
else
    echo "[1/4] Stage 2 backup not found, skipping..."
fi

# 2. Phase 1 이름 변경
if [ -d "checkpoints/phase1_write" ]; then
    echo "[2/4] Renaming phase1_write → phase1_final..."
    mv checkpoints/phase1_write checkpoints/phase1_final
    echo "      Done: checkpoints/phase1_final/"
else
    echo "[2/4] phase1_write not found (maybe already renamed?)"
fi

# 3. Config 복사
if [ -d "checkpoints/phase1_final" ]; then
    echo "[3/4] Copying config to archive..."
    cp configs/phase1_write.yaml checkpoints/phase1_final/config.yaml
    echo "      Done: checkpoints/phase1_final/config.yaml"
fi

# 4. Phase 2 디렉토리 생성
echo "[4/4] Creating Phase 2 directories..."
mkdir -p experiments
mkdir -p checkpoints/phase2/zrag_evidence
mkdir -p checkpoints/phase2/baselines
mkdir -p results/phase2_track_a
echo "      Done: experiments/, checkpoints/phase2/, results/phase2_track_a/"

echo ""
echo "=== Cleanup Complete ==="
echo ""
echo "Phase 1 archive: checkpoints/phase1_final/"
echo "Phase 2 ready:   checkpoints/phase2/"
echo ""

# 검증: Phase 1 checkpoint 확인
echo "=== Verifying Phase 1 checkpoint ==="
python -c "
import torch
ckpt = torch.load('checkpoints/phase1_final/z_pool_epoch50.pt', map_location='cpu')
print(f'z_vectors: {len(ckpt[\"z_vectors\"])} docs')
print(f'alpha: {ckpt[\"alpha\"]:.4f}')
print(f'projection: {\"z_to_embedding\" in ckpt}')
print(f'z shape: {list(ckpt[\"z_vectors\"].values())[0].shape}')
"
