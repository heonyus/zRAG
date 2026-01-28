"""
Phase 1: Write Phase Training Script

교수님 의도 (2=A):
- z_i만 넣으면 해당 문서 D_i가 생성되도록 학습
- LLM freeze, z_i + projection만 학습
- 문서별로 z_i를 최적화하고, 전체 z_pool로 저장

사용법:
    python training/train_write_phase.py --config configs/phase1_write.yaml
    python training/train_write_phase.py --config configs/phase1_write.yaml --test       # 빠른 테스트
    python training/train_write_phase.py --config configs/phase1_write.yaml --eval_only  # 평가만 실행
"""

import sys
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
import yaml
from omegaconf import OmegaConf

# Path setup
sys.path.append(str(Path(__file__).parent.parent))

from models.write_phase_model import WritePhaseModel, ZPoolManager
from data.download import download_dataset
from data.dataloader import WritePhaseDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 기본 콘솔 핸들러 (기존 방식)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(console_handler)


def setup_file_logging(log_dir: Path, run_name: str = None):
    """파일 로깅 설정"""
    from datetime import datetime

    log_dir.mkdir(parents=True, exist_ok=True)

    # 실행 이름 생성 (없으면 타임스탬프)
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = log_dir / f"train_{run_name}.log"

    # 파일 핸들러 추가
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    logger.info(f"📁 Log file: {log_file}")
    return log_file


def save_log_snapshot(log_file: Path, checkpoint_dir: Path, epoch: int):
    """체크포인트 저장 시 로그 스냅샷도 저장"""
    import shutil
    if log_file and log_file.exists():
        snapshot_path = checkpoint_dir / f"log_epoch{epoch}.txt"
        shutil.copy(log_file, snapshot_path)
        logger.info(f"  📝 Log snapshot: {snapshot_path}")


def prepare_corpus(dataset, max_docs: int = 200, dataset_name: str = "hotpot_qa", corpus_path: str = None) -> dict:
    """
    데이터셋에서 corpus 추출 또는 pre-built corpus 로드

    Args:
        dataset: HuggingFace dataset (corpus_path가 없을 때 사용)
        max_docs: 최대 문서 수
        dataset_name: 데이터셋 이름 (hotpot_qa, natural_questions 등)
        corpus_path: pre-built corpus JSON 파일 경로 (corpus_builder.py로 생성)

    Returns:
        corpus: {doc_id: doc_text} dict
    """
    # Pre-built corpus 로드 (있으면)
    if corpus_path is not None:
        import json
        from pathlib import Path
        corpus_file = Path(corpus_path)
        if corpus_file.exists():
            logger.info(f"Loading pre-built corpus from: {corpus_path}")
            with open(corpus_file, "r", encoding="utf-8") as f:
                corpus = json.load(f)
            logger.info(f"  Loaded {len(corpus)} documents from pre-built corpus")
            # max_docs 제한 적용
            if len(corpus) > max_docs:
                doc_ids = list(corpus.keys())[:max_docs]
                corpus = {k: corpus[k] for k in doc_ids}
                logger.info(f"  Trimmed to {len(corpus)} documents")
            return corpus
        else:
            logger.warning(f"corpus_path specified but not found: {corpus_path}")
            logger.warning("  Falling back to dataset extraction")

    corpus = {}

    # train split 처리
    if hasattr(dataset, "keys") and "train" in dataset.keys():
        data = dataset["train"]
    else:
        data = dataset

    for i, item in enumerate(data):
        if len(corpus) >= max_docs:
            break

        # HotpotQA format: context = {'title': [...], 'sentences': [[...], ...]}
        if dataset_name == "hotpot_qa" and "context" in item:
            ctx = item["context"]
            titles = ctx.get("title", [])
            sentences_list = ctx.get("sentences", [])

            # 각 문서(title + sentences)를 별도 문서로 추출
            for title, sentences in zip(titles, sentences_list):
                if len(corpus) >= max_docs:
                    break
                doc_text = f"{title}\n" + " ".join(sentences)
                if len(doc_text) > 50:  # 너무 짧은 문서 제외
                    doc_id = f"doc_{len(corpus)}"
                    corpus[doc_id] = doc_text

        # FlashRAG NQ format: retrieval_result가 있으면 그것이 문서
        elif "retrieval_result" in item and item["retrieval_result"]:
            for j, doc in enumerate(item["retrieval_result"][:1]):  # 첫 번째 문서만
                doc_id = f"doc_{len(corpus)}"
                doc_text = doc.get("contents", doc.get("text", ""))
                if doc_text and len(doc_text) > 50:
                    corpus[doc_id] = doc_text

        # 일반 context (문자열)
        elif "context" in item and isinstance(item["context"], str):
            doc_id = f"doc_{len(corpus)}"
            doc_text = item["context"]
            if len(doc_text) > 50:
                corpus[doc_id] = doc_text

    logger.info(f"Extracted {len(corpus)} documents from dataset")
    return corpus


def train_single_document(
    model: WritePhaseModel,
    doc_id: str,
    doc_ids: torch.Tensor,
    doc_attention_mask: torch.Tensor,
    config: dict,
    scaler: GradScaler = None,
    enable_diagnostics: bool = True,
) -> tuple:
    """
    단일 문서에 대해 z_i를 학습

    Args:
        model: WritePhaseModel (LLM frozen)
        doc_id: 문서 ID
        doc_ids: [1, doc_len] 토큰화된 문서
        doc_attention_mask: [1, doc_len]
        config: 학습 설정
        scaler: GradScaler for mixed precision
        enable_diagnostics: 중간 샘플 생성 및 통계 출력 여부

    Returns:
        z_i: 학습된 z_i tensor
        final_loss: 최종 loss
    """
    # 새 z_i 생성
    z_i = model.create_z_for_doc()
    z_i_init = z_i.clone().detach()  # 초기값 저장 (변화량 측정용)

    # Learning rates from config
    lr_z = float(config.get("lr_z", 1e-2))
    lr_proj = float(config.get("lr_proj", 0))

    # Optimizer (z_i + projection if lr_proj > 0)
    optimizer = AdamW(
        model.get_trainable_params(z_i, lr_z=lr_z, lr_proj=lr_proj),
        weight_decay=config.get("weight_decay", 0.01),
    )

    # 학습 설정
    epochs = config.get("epochs_per_doc", 100)
    log_every = config.get("log_every", 20)
    use_amp = config.get("use_amp", True)
    early_stop_loss = config.get("early_stop_loss", 0.5)

    best_loss = float("inf")
    best_z = z_i.clone().detach()

    # 진단용: 중간 샘플 생성할 epoch들
    diagnostic_epochs = {0, 1, 5, 10, 20, 50, epochs - 1} if enable_diagnostics else set()

    # 첫 문서의 첫 epoch에서 초기 상태 로깅
    if doc_id == "doc_0" and enable_diagnostics:
        stats = model.get_z_embed_stats(z_i)
        logger.info(f"  [{doc_id}] INIT: z_norm={stats['z_i_norm']:.4f}, "
                   f"z_embed_norm={stats['z_embed_norm']:.4f}, z_embed_std={stats['z_embed_std']:.4f}")

    for epoch in range(epochs):
        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast('cuda', dtype=torch.bfloat16):
                outputs = model(z_i, doc_ids, doc_attention_mask)
                loss = outputs["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Gradient norm 계산 (z_i만)
            z_grad_norm = z_i.grad.norm().item() if z_i.grad is not None else 0.0

            nn.utils.clip_grad_norm_([z_i], 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(z_i, doc_ids, doc_attention_mask)
            loss = outputs["loss"]

            loss.backward()
            z_grad_norm = z_i.grad.norm().item() if z_i.grad is not None else 0.0
            nn.utils.clip_grad_norm_([z_i], 1.0)
            optimizer.step()

        loss_val = loss.item()

        # Best 저장
        if loss_val < best_loss:
            best_loss = loss_val
            best_z = z_i.clone().detach()

        # Early stopping
        if loss_val < early_stop_loss:
            logger.info(f"  [{doc_id}] Early stop at epoch {epoch}, loss={loss_val:.4f}")
            break

        # Logging (every log_every epochs)
        if epoch % log_every == 0 or epoch == epochs - 1:
            z_change = (z_i - z_i_init).norm().item()
            logger.debug(f"  [{doc_id}] Epoch {epoch}/{epochs}: loss={loss_val:.4f}, "
                        f"z_grad={z_grad_norm:.4f}, z_change={z_change:.4f}")

        # 진단: 중간 샘플 생성 및 통계
        if epoch in diagnostic_epochs and enable_diagnostics:
            # z_embed 통계 출력
            stats = model.get_z_embed_stats(z_i)
            z_change = (z_i - z_i_init).norm().item()
            logger.info(f"  [{doc_id}] Epoch {epoch}: loss={loss_val:.4f} | "
                       f"z_norm={stats['z_i_norm']:.4f}, z_change={z_change:.4f}, "
                       f"z_grad={z_grad_norm:.4f} | "
                       f"z_embed_norm={stats['z_embed_norm']:.4f}, z_embed_std={stats['z_embed_std']:.4f}")

            # 첫 문서만 중간 생성 테스트 (시간 절약)
            if doc_id == "doc_0":
                try:
                    sample = model.generate_from_z(z_i.detach(), max_new_tokens=50, do_sample=True)
                    logger.info(f"  [{doc_id}] Epoch {epoch} sample: {sample[:100]}...")
                except Exception as e:
                    logger.warning(f"  [{doc_id}] Epoch {epoch} generate failed: {e}")

    # 최종 상태 로깅
    final_z_change = (best_z - z_i_init).norm().item()
    logger.debug(f"  [{doc_id}] FINAL: best_loss={best_loss:.4f}, total_z_change={final_z_change:.4f}")

    return best_z, best_loss


def train_shuffled_documents(
    model: WritePhaseModel,
    tokenized_docs: dict,
    z_vectors: dict,
    config: dict,
    scaler: GradScaler = None,
    start_epoch: int = 0,
    log_file: Path = None,
) -> dict:
    """
    Shuffled doc training: projection drift 방지를 위해 문서들을 섞어서 학습

    Args:
        start_epoch: resume할 경우 시작 epoch (0이면 처음부터)
        log_file: 로그 파일 경로 (체크포인트마다 스냅샷 저장)
    """
    import random
    import time
    import statistics

    # === Config 로드 ===
    lr_z = float(config.get("lr_z", 1e-2))
    lr_proj = float(config.get("lr_proj", 1e-5))
    epochs = config.get("epochs_per_doc", 100)
    log_every = config.get("log_every", 20)
    use_amp = config.get("use_amp", True)
    early_stop_loss = config.get("early_stop_loss", 0.5)
    collapse_threshold = config.get("collapse_threshold", 0.01)
    stagnation_patience = config.get("stagnation_patience", 5)
    checkpoint_every = config.get("checkpoint_every", 10)

    doc_ids = list(tokenized_docs.keys())
    num_docs = len(doc_ids)
    remaining_epochs = epochs - start_epoch
    total_iters = num_docs * remaining_epochs

    # === 학습 설정 출력 ===
    config_msg = f"""
{'=' * 70}
📋 TRAINING CONFIGURATION
{'=' * 70}
  Documents:     {num_docs}
  Epochs:        {epochs} (start={start_epoch}, remaining={remaining_epochs})
  Total iters:   {total_iters:,}
  lr_z:          {lr_z}
  lr_proj:       {lr_proj}
  use_amp:       {use_amp}
  log_every:     {log_every} epochs
  checkpoint:    every {checkpoint_every} epochs
{'=' * 70}
"""
    print(config_msg)
    logger.info(config_msg)

    # === Optimizer 설정 ===
    z_params = [z_vectors[doc_id] for doc_id in doc_ids]
    param_groups = [
        {"params": z_params, "lr": lr_z, "weight_decay": config.get("weight_decay", 0.01), "name": "z_vectors"},
        {"params": [model.alpha], "lr": lr_z, "weight_decay": 0.0, "name": "alpha"},
    ]

    if lr_proj > 0:
        param_groups.append({
            "params": model.z_to_embedding.parameters(),
            "lr": lr_proj,
            "weight_decay": 0.0,
            "name": "z_to_embedding"
        })
        print(f"🔧 Optimizer: z_lr={lr_z}, alpha_lr={lr_z}, proj_lr={lr_proj}")
    else:
        for param in model.z_to_embedding.parameters():
            param.requires_grad = False
        print(f"🔧 Optimizer: z_lr={lr_z}, alpha_lr={lr_z}, proj=FROZEN")

    optimizer = AdamW(param_groups, weight_decay=0.0)

    # === 상태 추적 변수 ===
    best_losses = {doc_id: float("inf") for doc_id in doc_ids}
    current_losses = {doc_id: float("inf") for doc_id in doc_ids}
    z_init = {doc_id: z_vectors[doc_id].clone().detach() for doc_id in doc_ids}

    loss_history = []
    best_avg_loss = float("inf")
    stagnation_counter = 0
    collapse_warned = False

    # === 초기 z 통계 ===
    init_z_norms = [z_vectors[d].norm().item() for d in doc_ids]
    init_z_stds = [z_vectors[d].std().item() for d in doc_ids]
    init_stats_msg = f"""
📊 Initial z stats:
   z_norm: mean={statistics.mean(init_z_norms):.4f}, std={statistics.stdev(init_z_norms) if len(init_z_norms) > 1 else 0:.4f}
   z_std:  mean={statistics.mean(init_z_stds):.4f}
   alpha:  {model.alpha.item():.4f}"""
    print(init_stats_msg)
    logger.info(init_stats_msg)

    # === 타이밍 ===
    start_time = time.time()
    epoch_times = []

    # === 메인 학습 루프 ===
    print("\n" + "=" * 70)
    print("🚀 TRAINING START")
    print("=" * 70)

    # 전체 진행률 바
    total_pbar = tqdm(
        total=total_iters,
        desc="Total",
        position=0,
        leave=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    global_iter = 0

    if start_epoch > 0:
        print(f"\n🔄 RESUMING from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        random.shuffle(doc_ids)

        epoch_losses = []
        epoch_grad_norms = []

        # Epoch 진행률 바
        epoch_pbar = tqdm(
            doc_ids,
            desc=f"Ep {epoch:03d}",
            position=1,
            leave=False,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
        )

        for doc_idx, doc_id in enumerate(epoch_pbar):
            optimizer.zero_grad()

            doc_data = tokenized_docs[doc_id]
            z_i = z_vectors[doc_id]

            # Forward + Backward
            if use_amp and scaler is not None:
                with autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(z_i, doc_data["input_ids"], doc_data["attention_mask"])
                    loss = outputs["loss"]

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(z_params + list(model.z_to_embedding.parameters()), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(z_i, doc_data["input_ids"], doc_data["attention_mask"])
                loss = outputs["loss"]

                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(z_params + list(model.z_to_embedding.parameters()), 1.0)
                optimizer.step()

            # 통계 수집
            loss_val = loss.item()
            grad_norm_val = grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm
            epoch_losses.append(loss_val)
            epoch_grad_norms.append(grad_norm_val)
            current_losses[doc_id] = loss_val

            if loss_val < best_losses[doc_id]:
                best_losses[doc_id] = loss_val

            # Epoch 진행률 바 업데이트
            running_avg = sum(epoch_losses) / len(epoch_losses)
            epoch_pbar.set_postfix({
                'loss': f'{loss_val:.3f}',
                'avg': f'{running_avg:.3f}',
                'α': f'{model.alpha.item():.2f}'
            })

            # 전체 진행률 바 업데이트
            global_iter += 1
            total_pbar.update(1)
            total_pbar.set_postfix({
                'ep': f'{epoch}/{epochs}',
                'loss': f'{running_avg:.3f}',
                'α': f'{model.alpha.item():.2f}'
            })

        epoch_pbar.close()

        # === Epoch 통계 계산 ===
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        avg_loss = statistics.mean(epoch_losses)
        loss_std = statistics.stdev(epoch_losses) if len(epoch_losses) > 1 else 0
        loss_min = min(epoch_losses)
        loss_max = max(epoch_losses)

        avg_grad = statistics.mean(epoch_grad_norms)

        # z 통계
        z_norms = [z_vectors[d].norm().item() for d in doc_ids]
        z_stds = [z_vectors[d].std().item() for d in doc_ids]
        z_changes = [(z_vectors[d] - z_init[d]).norm().item() for d in doc_ids]

        avg_z_norm = statistics.mean(z_norms)
        avg_z_std = statistics.mean(z_stds)
        avg_z_change = statistics.mean(z_changes)

        loss_history.append(avg_loss)

        # === Epoch 로그 출력 ===
        if epoch % log_every == 0 or epoch == epochs - 1 or epoch == start_epoch:
            elapsed = time.time() - start_time
            epochs_done = epoch - start_epoch + 1
            epochs_remaining = epochs - epoch - 1
            if epochs_done > 1:
                eta = (elapsed / epochs_done) * epochs_remaining
                eta_str = f"{int(eta // 60):02d}:{int(eta % 60):02d}"
            else:
                eta_str = "--:--"
            elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"

            epoch_log = f"""
{'─' * 70}
📈 EPOCH {epoch:03d}/{epochs} | elapsed={elapsed_str} | ETA={eta_str} | {epoch_time:.1f}s/ep
{'─' * 70}
  Loss:  avg={avg_loss:.4f} | std={loss_std:.4f} | min={loss_min:.4f} | max={loss_max:.4f}
  Grad:  avg_norm={avg_grad:.4f}
  Alpha: {model.alpha.item():.4f}
  z:     norm={avg_z_norm:.4f} | std={avg_z_std:.4f} | Δ={avg_z_change:.4f}"""

            # 개선 상태
            if epoch > start_epoch:
                improvement = loss_history[-2] - avg_loss
                arrow = "↓" if improvement > 0 else "↑" if improvement < 0 else "→"
                epoch_log += f"\n  Δloss: {arrow} {abs(improvement):.4f} (prev={loss_history[-2]:.4f})"

            print(epoch_log)
            logger.info(epoch_log)

        # === 안전장치 체크 ===
        # 1. Collapse 감지
        if avg_z_std < collapse_threshold and not collapse_warned:
            print(f"\n⚠️  [COLLAPSE WARNING] z_std={avg_z_std:.6f} < {collapse_threshold}")
            print(f"    z vectors가 너무 비슷해지고 있음!")
            collapse_warned = True

        # 2. Stagnation 감지
        if avg_loss < best_avg_loss - 0.001:
            best_avg_loss = avg_loss
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            if stagnation_counter >= stagnation_patience and stagnation_counter % stagnation_patience == 0:
                print(f"\n⚠️  [STAGNATION] {stagnation_counter} epochs without improvement")
                print(f"    best={best_avg_loss:.4f}, current={avg_loss:.4f}")

        # 3. 샘플 생성 (특정 epoch에서)
        if epoch in {start_epoch, start_epoch + 1, 5, 10, epochs // 2, epochs - 1}:
            test_doc_id = doc_ids[0]
            try:
                with torch.no_grad():
                    sample = model.generate_from_z(
                        z_vectors[test_doc_id].detach(),
                        max_new_tokens=40,
                        do_sample=True
                    )
                print(f"  Sample: \"{sample[:80]}...\"")
            except Exception as e:
                print(f"  Sample: [failed: {e}]")

        # 4. 체크포인트 저장
        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            checkpoint_dir = Path(config.get("save_dir", "./checkpoints/phase1_write"))
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = checkpoint_dir / f"z_pool_epoch{epoch+1}.pt"
            checkpoint_data = {
                "epoch": epoch + 1,
                "z_vectors": {doc_id: z_vectors[doc_id].detach().cpu() for doc_id in doc_ids},
                "avg_loss": avg_loss,
                "alpha": model.alpha.item(),
                "loss_history": loss_history,
                # projection layer도 저장 (resume 시 필요)
                "z_to_embedding": model.z_to_embedding.state_dict(),
            }
            torch.save(checkpoint_data, checkpoint_path)
            print(f"  💾 Checkpoint: {checkpoint_path}")

            # 로그 스냅샷 저장
            if log_file is not None:
                save_log_snapshot(log_file, checkpoint_dir, epoch + 1)

        # 5. Early stopping
        if avg_loss < early_stop_loss:
            print(f"\n✅ Early stopping at epoch {epoch}, loss={avg_loss:.4f} < {early_stop_loss}")
            break

    total_pbar.close()

    # === 최종 요약 ===
    total_time = time.time() - start_time
    final_avg_loss = statistics.mean(list(best_losses.values()))

    summary_msg = f"""
{'=' * 70}
🏁 TRAINING COMPLETED
{'=' * 70}
  Total time:    {int(total_time // 60)}m {int(total_time % 60)}s
  Epochs:        {start_epoch} → {len(loss_history) + start_epoch} (ran {len(loss_history)}/{epochs - start_epoch})
  Final loss:    {loss_history[-1]:.4f}
  Best avg loss: {final_avg_loss:.4f}
  Final alpha:   {model.alpha.item():.4f}"""

    print(summary_msg)
    logger.info(summary_msg)

    # z 최종 통계
    final_z_norms = [z_vectors[d].norm().item() for d in doc_ids]
    final_z_stds = [z_vectors[d].std().item() for d in doc_ids]
    final_z_changes = [(z_vectors[d] - z_init[d]).norm().item() for d in doc_ids]

    z_stats_msg = f"""
  z final stats:
    norm:   {statistics.mean(final_z_norms):.4f} (init: {statistics.mean(init_z_norms):.4f})
    std:    {statistics.mean(final_z_stds):.4f} (init: {statistics.mean(init_z_stds):.4f})
    change: {statistics.mean(final_z_changes):.4f}"""
    print(z_stats_msg)
    logger.info(z_stats_msg)

    # Loss 변화
    if len(loss_history) > 1:
        loss_msg = f"""
  Loss trajectory: {loss_history[0]:.3f} → {loss_history[-1]:.3f}
    Reduction: {loss_history[0] - loss_history[-1]:.3f} ({(1 - loss_history[-1]/loss_history[0])*100:.1f}%)"""
        print(loss_msg)
        logger.info(loss_msg)

    print("=" * 70 + "\n")
    logger.info("=" * 70)

    return best_losses


def run_write_phase_training(config_path: str = None, config: dict = None, test_mode: bool = False, eval_only: bool = False, resume: bool = False):
    """
    Phase 1: Write Phase 전체 학습 실행

    Args:
        config_path: YAML config 파일 경로
        config: config dict (직접 전달 시)
        test_mode: True면 소규모로 빠른 테스트
        eval_only: True면 학습 스킵하고 저장된 checkpoint로 평가만 실행
        resume: True면 마지막 체크포인트에서 이어서 학습

    Returns:
        model: WritePhaseModel
        z_pool_manager: 학습된 z_i들
        results: 학습 결과
    """
    # Load config
    if config is None:
        with open(config_path, "r") as f:
            config = OmegaConf.create(yaml.safe_load(f))

    # Test mode 오버라이드
    if test_mode:
        config.data.num_docs = 10
        config.training.epochs_per_doc = 20
        logger.info("=" * 60)
        logger.info("TEST MODE: num_docs=10, epochs_per_doc=20")
        logger.info("=" * 60)

    logger.info("=" * 60)
    logger.info("Phase 1: Write Phase Training (Token-as-Document)")
    logger.info("=" * 60)
    logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # ==========================================
    # 1. Data Preparation
    # ==========================================
    logger.info("\n[Step 1] Data Preparation")

    data_config = config.data
    raw_data = download_dataset(
        dataset_name=data_config.dataset,
        save_dir=data_config.get("save_dir", "./data/raw"),
    )

    # Corpus 추출 (pre-built corpus가 있으면 그것 사용)
    corpus_path = data_config.get("corpus_path", None)
    corpus = prepare_corpus(
        raw_data,
        max_docs=data_config.num_docs,
        dataset_name=data_config.dataset,
        corpus_path=corpus_path,
    )
    logger.info(f"Corpus size: {len(corpus)} documents")

    if len(corpus) == 0:
        raise ValueError("No documents extracted from dataset!")

    # ==========================================
    # 2. Model Initialization
    # ==========================================
    logger.info("\n[Step 2] Model Initialization")

    model_config = config.model
    memory_config = config.memory

    model = WritePhaseModel(
        llm_name=model_config.llm_name,
        m_tokens=memory_config.m_tokens,
        z_dim=memory_config.z_dim,
        quantization=model_config.get("quantization", "4bit"),
    )

    # Z Pool Manager
    z_pool_manager = ZPoolManager(
        m_tokens=memory_config.m_tokens,
        z_dim=memory_config.z_dim,
    )

    # ==========================================
    # 3. Tokenize Documents
    # ==========================================
    logger.info("\n[Step 3] Tokenizing Documents")

    tokenizer = model.tokenizer
    max_doc_length = data_config.get("max_doc_length", 512)

    tokenized_docs = {}
    for doc_id, doc_text in tqdm(corpus.items(), desc="Tokenizing"):
        encoded = tokenizer(
            doc_text,
            max_length=max_doc_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tokenized_docs[doc_id] = {
            "input_ids": encoded["input_ids"].cuda(),
            "attention_mask": encoded["attention_mask"].cuda(),
        }

    logger.info(f"Tokenized {len(tokenized_docs)} documents")

    # ==========================================
    # 4. Training: Shuffled Document Training
    # ==========================================
    train_config = config.training
    use_amp = train_config.get("use_amp", True)

    save_dir = Path(config.logging.get("save_dir", "./checkpoints/phase1_write"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # 로그 디렉토리 설정
    log_dir = save_dir / "logs"
    log_file = setup_file_logging(log_dir)

    doc_ids_list = list(tokenized_docs.keys())

    # Resume 모드: 최신 체크포인트 찾기
    start_epoch = 0
    resume_z_vectors = None
    resume_alpha = None
    resume_projection = None

    if resume:
        logger.info("\n[Step 3.5] RESUME MODE - Finding latest checkpoint")
        logger.info("=" * 60)

        # z_pool_epoch{N}.pt 파일들 찾기
        import re
        checkpoint_files = list(save_dir.glob("z_pool_epoch*.pt"))
        if not checkpoint_files:
            logger.warning("No checkpoint files found, starting from scratch")
            resume = False
        else:
            # 가장 높은 epoch 번호 찾기
            epoch_pattern = re.compile(r"z_pool_epoch(\d+)\.pt")
            epochs_found = []
            for f in checkpoint_files:
                match = epoch_pattern.search(f.name)
                if match:
                    epochs_found.append((int(match.group(1)), f))

            if epochs_found:
                epochs_found.sort(key=lambda x: x[0], reverse=True)
                latest_epoch, latest_ckpt = epochs_found[0]
                logger.info(f"Found {len(epochs_found)} checkpoints, latest: epoch {latest_epoch}")

                # 체크포인트 로드
                ckpt = torch.load(latest_ckpt, map_location="cpu")
                start_epoch = ckpt["epoch"]
                resume_z_vectors = ckpt["z_vectors"]
                resume_alpha = ckpt.get("alpha", None)
                resume_projection = ckpt.get("z_to_embedding", None)

                logger.info(f"Resuming from epoch {start_epoch}")
                logger.info(f"  z_vectors: {len(resume_z_vectors)} documents")

                # z_vectors shape 검증
                sample_z = list(resume_z_vectors.values())[0]
                expected_shape = (memory_config.m_tokens, memory_config.z_dim)
                if tuple(sample_z.shape) != expected_shape:
                    logger.error(f"  z_vectors shape mismatch!")
                    logger.error(f"    checkpoint: {tuple(sample_z.shape)}")
                    logger.error(f"    expected:   {expected_shape}")
                    logger.error(f"  Cannot resume - delete old checkpoints and restart")
                    raise ValueError(f"z_vectors shape mismatch: {tuple(sample_z.shape)} != {expected_shape}")
                logger.info(f"  z_vectors shape: {tuple(sample_z.shape)} ✓")

                if resume_alpha is not None:
                    logger.info(f"  alpha: {resume_alpha:.4f}")
                if resume_projection is not None:
                    logger.info(f"  projection: found in checkpoint ✓")
                else:
                    logger.warning(f"  projection: NOT found in checkpoint!")
                logger.info("=" * 60)
            else:
                logger.warning("No valid checkpoint files found, starting from scratch")
                resume = False

    if eval_only:
        # ==========================================
        # EVAL ONLY MODE: Load from checkpoint
        # ==========================================
        logger.info("\n[Step 4] EVAL ONLY MODE - Loading from checkpoint")
        logger.info("=" * 60)

        z_pool_path = save_dir / "z_pool.pt"
        proj_path = save_dir / "projection.pt"

        if not z_pool_path.exists():
            raise FileNotFoundError(f"z_pool not found: {z_pool_path}")
        if not proj_path.exists():
            raise FileNotFoundError(f"projection not found: {proj_path}")

        # Load z_pool
        logger.info(f"Loading z_pool from: {z_pool_path}")
        z_pool_manager.load(z_pool_path)

        # Load projection
        logger.info(f"Loading projection from: {proj_path}")
        model.load_projection(proj_path)

        # alpha fallback: epoch checkpoint에서 로드 시도
        if model.alpha.item() == 1.0:
            # projection.pt에 alpha 없음 - epoch checkpoint에서 찾기
            epochs = config.training.get("epochs_per_doc", 30)
            epoch_ckpt_path = save_dir / f"z_pool_epoch{epochs}.pt"
            if epoch_ckpt_path.exists():
                epoch_ckpt = torch.load(epoch_ckpt_path, map_location="cpu")
                if "alpha" in epoch_ckpt:
                    loaded_alpha = epoch_ckpt["alpha"]
                    with torch.no_grad():
                        model.alpha.fill_(loaded_alpha)
                    logger.info(f"Loaded alpha={loaded_alpha:.4f} from epoch checkpoint")

                    # projection.pt를 새 포맷(alpha 포함)으로 재저장
                    model.save_projection(proj_path)
                    logger.info(f"Re-saved projection.pt with alpha={loaded_alpha:.4f}")

        z_pool_tensor = z_pool_manager.get_pool_tensor()
        logger.info(f"Loaded z_pool: shape={tuple(z_pool_tensor.shape)}")
        logger.info(f"Alpha after load: {model.alpha.item():.4f}")
        logger.info("=" * 60)

        # 결과는 저장된 것 로드 (있으면)
        results_path = save_dir / "results.pt"
        if results_path.exists():
            results = torch.load(results_path)
            logger.info(f"Loaded previous results: avg_loss={results.get('avg_loss', 'N/A')}")
        else:
            results = {
                "num_docs": len(corpus),
                "config": OmegaConf.to_container(config),
            }
    else:
        # ==========================================
        # TRAINING MODE
        # ==========================================
        logger.info("\n[Step 4] Training z_i with shuffled documents (drift 방지)")

        scaler = GradScaler('cuda') if use_amp else None

        # Training config 로깅
        lr_z = float(train_config.get("lr_z", 1e-2))
        lr_proj = float(train_config.get("lr_proj", 1e-5))
        epochs_per_doc = train_config.get("epochs_per_doc", 100)
        logger.info(f"Training config: lr_z={lr_z}, lr_proj={lr_proj}, epochs={epochs_per_doc}")
        logger.info(f"Projection: {'FROZEN' if lr_proj == 0 else f'learning (lr={lr_proj})'}")
        logger.info(f"Training mode: SHUFFLED (all docs trained together)")

        # 모든 문서에 대해 z_i 생성 (또는 resume에서 로드)
        z_vectors = {}
        if resume and resume_z_vectors is not None:
            logger.info(f"Loading {len(resume_z_vectors)} z_vectors from checkpoint")
            for doc_id in tqdm(doc_ids_list, desc="Loading z_i vectors"):
                if doc_id in resume_z_vectors:
                    z_vectors[doc_id] = resume_z_vectors[doc_id].to(model.device).requires_grad_(True)
                else:
                    logger.warning(f"  {doc_id} not in checkpoint, creating new")
                    z_vectors[doc_id] = model.create_z_for_doc()

            # alpha 복원
            if resume_alpha is not None:
                with torch.no_grad():
                    model.alpha.fill_(resume_alpha)
                logger.info(f"Restored alpha: {model.alpha.item():.4f}")

            # projection layer 복원
            if resume_projection is not None:
                model.z_to_embedding.load_state_dict(resume_projection)
                logger.info(f"Restored projection layer from checkpoint")
            else:
                logger.warning("Projection NOT restored - training from scratch projection!")
        else:
            for doc_id in tqdm(doc_ids_list, desc="Creating z_i vectors"):
                z_vectors[doc_id] = model.create_z_for_doc()
        logger.info(f"Prepared {len(z_vectors)} z_i vectors")

        # train_config에 save_dir 추가 (체크포인트용)
        train_config_dict = dict(train_config)
        train_config_dict["save_dir"] = str(save_dir)

        # Shuffled training 실행
        losses = train_shuffled_documents(
            model=model,
            tokenized_docs=tokenized_docs,
            z_vectors=z_vectors,
            config=train_config_dict,
            scaler=scaler,
            start_epoch=start_epoch,
            log_file=log_file,
        )

        # 결과 저장
        results = {
            "losses": losses,
            "num_docs": len(corpus),
            "config": OmegaConf.to_container(config),
        }

        # z_pool에 추가
        for doc_id in tqdm(doc_ids_list, desc="Saving to z_pool"):
            z_pool_manager.add_z(doc_id, z_vectors[doc_id].detach())

    # ==========================================
    # 5. Final Save (skip if eval_only)
    # ==========================================
    if not eval_only:
        logger.info("\n[Step 5] Saving Results")

        # z_pool 저장 (Phase 3에서 로드할 메인 파일)
        z_pool_path = save_dir / "z_pool.pt"
        z_pool_manager.save(z_pool_path)

        # Projection layer 저장
        proj_path = save_dir / "projection.pt"
        model.save_projection(proj_path)

        # Results 저장
        results["avg_loss"] = sum(results["losses"].values()) / len(results["losses"])
        torch.save(results, save_dir / "results.pt")

        logger.info(f"\nFinal Average Loss: {results['avg_loss']:.4f}")
        logger.info(f"Saved z_pool to: {z_pool_path}")
        logger.info(f"Saved projection to: {proj_path}")

        # ==========================================
        # 5.1 Corpus Manifest 저장 (데이터 동일성 검증용)
        # ==========================================
        import hashlib
        import json

        corpus_manifest = {
            "created_at": str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu",
            "num_docs": len(doc_ids_list),
            "documents": {}
        }

        for doc_id in tqdm(doc_ids_list, desc="Creating manifest"):
            text = corpus[doc_id]
            text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            first_16_tokens = tokenized_docs[doc_id]["input_ids"][0, :16].tolist()

            corpus_manifest["documents"][doc_id] = {
                "text_sha256": text_hash,
                "text_len_chars": len(text),
                "first_16_tokens": first_16_tokens,
            }

        manifest_path = save_dir / "corpus_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(corpus_manifest, f, indent=2)
        logger.info(f"Saved corpus manifest to: {manifest_path}")
    else:
        logger.info("\n[Step 5] Skipping save (eval_only mode)")

    # ==========================================
    # 6. Validation: Document NLL Evaluation (핵심 지표)
    # ==========================================
    logger.info("\n[Step 6] Validation: Document NLL Evaluation")
    logger.info("=" * 60)
    logger.info("목적: z_i가 실제로 문서 content를 인코딩했는지 확인")
    logger.info("방법: NLL(doc_i | z_i) vs NLL(doc_i | z_random) 비교")
    logger.info("=" * 60)

    # Projection 상태 확인
    proj_frozen = not any(p.requires_grad for p in model.z_to_embedding.parameters())
    logger.info(f"\nProjection layer: {'FROZEN' if proj_frozen else 'TRAINABLE'}")
    logger.info(f"Final alpha value: {model.alpha.item():.4f}")

    # z_pool 통계
    z_pool_tensor = z_pool_manager.get_pool_tensor()
    logger.info(f"z_pool shape: {z_pool_tensor.shape}")
    logger.info(f"z_pool stats: mean={z_pool_tensor.mean():.4f}, std={z_pool_tensor.std():.4f}")

    # Document NLL 평가 (샘플 100개로 증가 - 안정적 평균 확보)
    num_eval_samples = min(100, len(doc_ids_list))
    correct_nlls = []
    random_nlls = []
    wrong_nlls = []

    logger.info(f"\n📊 Document NLL Evaluation ({num_eval_samples} samples)")
    logger.info("-" * 60)

    # autocast context for evaluation (same as training)
    from contextlib import nullcontext
    amp_context = autocast('cuda', dtype=torch.bfloat16) if use_amp else nullcontext()

    model.eval()
    with torch.no_grad(), amp_context:
        for i in tqdm(range(num_eval_samples), desc="NLL Evaluation"):
            doc_id = doc_ids_list[i]
            doc_data = tokenized_docs[doc_id]

            # 1. NLL with correct z_i
            z_correct = z_pool_manager.get_z(doc_id).to(model.device)
            outputs_correct = model(z_correct, doc_data["input_ids"], doc_data["attention_mask"])
            nll_correct = outputs_correct["loss"].item()
            correct_nlls.append(nll_correct)

            # 2. NLL with random z (norm-matched for fair comparison)
            z_random_raw = torch.randn_like(z_correct)
            # correct z와 동일한 norm으로 정규화
            z_random = z_random_raw * (z_correct.norm() / z_random_raw.norm())
            outputs_random = model(z_random, doc_data["input_ids"], doc_data["attention_mask"])
            nll_random = outputs_random["loss"].item()
            random_nlls.append(nll_random)

            # 3. NLL with wrong z (다른 문서의 z)
            wrong_idx = (i + num_eval_samples // 2) % len(doc_ids_list)
            wrong_doc_id = doc_ids_list[wrong_idx]
            z_wrong = z_pool_manager.get_z(wrong_doc_id).to(model.device)
            outputs_wrong = model(z_wrong, doc_data["input_ids"], doc_data["attention_mask"])
            nll_wrong = outputs_wrong["loss"].item()
            wrong_nlls.append(nll_wrong)

            logger.info(f"  {doc_id}: correct={nll_correct:.3f}, wrong={nll_wrong:.3f}, random={nll_random:.3f}")

    # 통계 계산
    import statistics
    avg_correct = statistics.mean(correct_nlls)
    avg_random = statistics.mean(random_nlls)
    avg_wrong = statistics.mean(wrong_nlls)

    std_correct = statistics.stdev(correct_nlls) if len(correct_nlls) > 1 else 0
    std_random = statistics.stdev(random_nlls) if len(random_nlls) > 1 else 0
    std_wrong = statistics.stdev(wrong_nlls) if len(wrong_nlls) > 1 else 0

    # z 효과성 지표
    z_benefit_vs_random = avg_random - avg_correct
    z_benefit_vs_wrong = avg_wrong - avg_correct
    z_specificity = (sum(1 for c, w in zip(correct_nlls, wrong_nlls) if c < w) / num_eval_samples) * 100

    print("\n" + "=" * 60)
    print(f"📈 DOCUMENT NLL RESULTS (n={num_eval_samples})")
    print("=" * 60)
    print(f"  avg NLL (correct z):  {avg_correct:.4f} ± {std_correct:.4f}")
    print(f"  avg NLL (wrong z):    {avg_wrong:.4f} ± {std_wrong:.4f}")
    print(f"  avg NLL (random z):   {avg_random:.4f} ± {std_random:.4f}")
    print()
    print(f"  z benefit vs random:  {z_benefit_vs_random:+.4f} ({'✅ GOOD' if z_benefit_vs_random > 0.5 else '⚠️ WEAK' if z_benefit_vs_random > 0 else '❌ BAD'})")
    print(f"  z benefit vs wrong:   {z_benefit_vs_wrong:+.4f} ({'✅ GOOD' if z_benefit_vs_wrong > 0.3 else '⚠️ WEAK' if z_benefit_vs_wrong > 0 else '❌ BAD'})")
    print(f"  z specificity:        {z_specificity:.1f}% correct < wrong ({'✅ GOOD' if z_specificity > 70 else '⚠️ WEAK' if z_specificity > 50 else '❌ BAD'})")
    print("=" * 60)

    # 결과 저장
    results["nll_correct"] = avg_correct
    results["nll_random"] = avg_random
    results["nll_wrong"] = avg_wrong
    results["z_benefit_vs_random"] = z_benefit_vs_random
    results["z_specificity"] = z_specificity

    # 간단 생성 테스트 (참고용)
    logger.info("\n📝 Sample Generation (참고용)")
    with amp_context:
        for i in range(min(2, len(doc_ids_list))):
            doc_id = doc_ids_list[i]
            z_i = z_pool_manager.get_z(doc_id).to(model.device)
            try:
                generated = model.generate_from_z(z_i, max_new_tokens=60, do_sample=False)
                original = corpus[doc_id][:100]
                logger.info(f"\n  [{doc_id}]")
                logger.info(f"    Original: {original}...")
                logger.info(f"    Generated: {generated[:100]}...")
            except Exception as e:
                logger.warning(f"  [{doc_id}] Generation failed: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("Phase 1 Training Complete!")
    logger.info("=" * 60)

    return model, z_pool_manager, results


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Phase 1: Write Phase Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase1_write.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (small scale)",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training, load checkpoint and run evaluation only",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint",
    )
    args = parser.parse_args()

    run_write_phase_training(
        config_path=args.config,
        test_mode=args.test,
        eval_only=args.eval_only,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
