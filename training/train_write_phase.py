"""
Phase 1: Write Phase Training Script

교수님 의도 (2=A):
- z_i만 넣으면 해당 문서 D_i가 생성되도록 학습
- LLM freeze, z_i + projection만 학습
- 문서별로 z_i를 최적화하고, 전체 z_pool로 저장

사용법:
    python training/train_write_phase.py --config configs/phase1_write.yaml
    python training/train_write_phase.py --config configs/phase1_write.yaml --test  # 빠른 테스트
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def prepare_corpus(dataset, max_docs: int = 200, dataset_name: str = "hotpot_qa") -> dict:
    """
    데이터셋에서 corpus 추출

    Args:
        dataset: HuggingFace dataset
        max_docs: 최대 문서 수
        dataset_name: 데이터셋 이름 (hotpot_qa, natural_questions 등)

    Returns:
        corpus: {doc_id: doc_text} dict
    """
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
) -> dict:
    """
    Shuffled doc training: projection drift 방지를 위해 문서들을 섞어서 학습
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
    total_iters = num_docs * epochs

    # === 학습 설정 출력 ===
    print("\n" + "=" * 70)
    print("📋 TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"  Documents:     {num_docs}")
    print(f"  Epochs:        {epochs}")
    print(f"  Total iters:   {total_iters:,}")
    print(f"  lr_z:          {lr_z}")
    print(f"  lr_proj:       {lr_proj}")
    print(f"  use_amp:       {use_amp}")
    print(f"  log_every:     {log_every} epochs")
    print(f"  checkpoint:    every {checkpoint_every} epochs")
    print("=" * 70 + "\n")

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
    print(f"\n📊 Initial z stats:")
    print(f"   z_norm: mean={statistics.mean(init_z_norms):.4f}, std={statistics.stdev(init_z_norms) if len(init_z_norms) > 1 else 0:.4f}")
    print(f"   z_std:  mean={statistics.mean(init_z_stds):.4f}")
    print(f"   alpha:  {model.alpha.item():.4f}")

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

    for epoch in range(epochs):
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
        if epoch % log_every == 0 or epoch == epochs - 1 or epoch < 3:
            elapsed = time.time() - start_time
            if epoch > 0:
                eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)
                eta_str = f"{int(eta // 60):02d}:{int(eta % 60):02d}"
            else:
                eta_str = "--:--"
            elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"

            print(f"\n{'─' * 70}")
            print(f"📈 EPOCH {epoch:03d}/{epochs} | elapsed={elapsed_str} | ETA={eta_str} | {epoch_time:.1f}s/ep")
            print(f"{'─' * 70}")
            print(f"  Loss:  avg={avg_loss:.4f} | std={loss_std:.4f} | min={loss_min:.4f} | max={loss_max:.4f}")
            print(f"  Grad:  avg_norm={avg_grad:.4f}")
            print(f"  Alpha: {model.alpha.item():.4f}")
            print(f"  z:     norm={avg_z_norm:.4f} | std={avg_z_std:.4f} | Δ={avg_z_change:.4f}")

            # 개선 상태
            if epoch > 0:
                improvement = loss_history[-2] - avg_loss
                arrow = "↓" if improvement > 0 else "↑" if improvement < 0 else "→"
                print(f"  Δloss: {arrow} {abs(improvement):.4f} (prev={loss_history[-2]:.4f})")

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
        if epoch in {0, 1, 5, 10, epochs // 2, epochs - 1}:
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
            }
            torch.save(checkpoint_data, checkpoint_path)
            print(f"  💾 Checkpoint: {checkpoint_path}")

        # 5. Early stopping
        if avg_loss < early_stop_loss:
            print(f"\n✅ Early stopping at epoch {epoch}, loss={avg_loss:.4f} < {early_stop_loss}")
            break

    total_pbar.close()

    # === 최종 요약 ===
    total_time = time.time() - start_time
    final_avg_loss = statistics.mean(list(best_losses.values()))

    print("\n" + "=" * 70)
    print("🏁 TRAINING COMPLETED")
    print("=" * 70)
    print(f"  Total time:    {int(total_time // 60)}m {int(total_time % 60)}s")
    print(f"  Epochs:        {len(loss_history)}/{epochs}")
    print(f"  Final loss:    {loss_history[-1]:.4f}")
    print(f"  Best avg loss: {final_avg_loss:.4f}")
    print(f"  Final alpha:   {model.alpha.item():.4f}")

    # z 최종 통계
    final_z_norms = [z_vectors[d].norm().item() for d in doc_ids]
    final_z_stds = [z_vectors[d].std().item() for d in doc_ids]
    final_z_changes = [(z_vectors[d] - z_init[d]).norm().item() for d in doc_ids]

    print(f"\n  z final stats:")
    print(f"    norm:   {statistics.mean(final_z_norms):.4f} (init: {statistics.mean(init_z_norms):.4f})")
    print(f"    std:    {statistics.mean(final_z_stds):.4f} (init: {statistics.mean(init_z_stds):.4f})")
    print(f"    change: {statistics.mean(final_z_changes):.4f}")

    # Loss 변화
    if len(loss_history) > 1:
        print(f"\n  Loss trajectory: {loss_history[0]:.3f} → {loss_history[-1]:.3f}")
        print(f"    Reduction: {loss_history[0] - loss_history[-1]:.3f} ({(1 - loss_history[-1]/loss_history[0])*100:.1f}%)")

    print("=" * 70 + "\n")

    return best_losses


def run_write_phase_training(config_path: str = None, config: dict = None, test_mode: bool = False):
    """
    Phase 1: Write Phase 전체 학습 실행

    Args:
        config_path: YAML config 파일 경로
        config: config dict (직접 전달 시)
        test_mode: True면 소규모로 빠른 테스트

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

    # Corpus 추출
    corpus = prepare_corpus(
        raw_data,
        max_docs=data_config.num_docs,
        dataset_name=data_config.dataset,
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
    logger.info("\n[Step 4] Training z_i with shuffled documents (drift 방지)")

    train_config = config.training
    use_amp = train_config.get("use_amp", True)
    scaler = GradScaler('cuda') if use_amp else None

    # Training config 로깅
    lr_z = float(train_config.get("lr_z", 1e-2))
    lr_proj = float(train_config.get("lr_proj", 1e-5))
    epochs_per_doc = train_config.get("epochs_per_doc", 100)
    logger.info(f"Training config: lr_z={lr_z}, lr_proj={lr_proj}, epochs={epochs_per_doc}")
    logger.info(f"Projection: {'FROZEN' if lr_proj == 0 else f'learning (lr={lr_proj})'}")
    logger.info(f"Training mode: SHUFFLED (all docs trained together)")

    save_dir = Path(config.logging.get("save_dir", "./checkpoints/phase1_write"))
    save_dir.mkdir(parents=True, exist_ok=True)

    doc_ids_list = list(tokenized_docs.keys())

    # 모든 문서에 대해 z_i 생성
    z_vectors = {}
    for doc_id in tqdm(doc_ids_list, desc="Creating z_i vectors"):
        z_vectors[doc_id] = model.create_z_for_doc()
    logger.info(f"Created {len(z_vectors)} z_i vectors")

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
    # 5. Final Save
    # ==========================================
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

    # ==========================================
    # 6. Validation: Generate from z + Keyword Check
    # ==========================================
    logger.info("\n[Step 6] Validation: Generate from learned z")

    # Projection 상태 확인
    proj_frozen = not any(p.requires_grad for p in model.z_to_embedding.parameters())
    logger.info(f"Projection layer: {'FROZEN' if proj_frozen else 'TRAINABLE'}")
    logger.info(f"Final alpha value: {model.alpha.item():.4f}")

    # z_pool 통계
    z_pool_tensor = z_pool_manager.get_pool_tensor()
    logger.info(f"z_pool shape: {z_pool_tensor.shape}")
    logger.info(f"z_pool stats: mean={z_pool_tensor.mean():.4f}, std={z_pool_tensor.std():.4f}, "
               f"min={z_pool_tensor.min():.4f}, max={z_pool_tensor.max():.4f}")

    def extract_keywords(text: str, top_n: int = 10) -> set:
        """문서에서 주요 키워드 추출 (간단 버전: 길이 4+ 단어)"""
        import re
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        # 빈도순 정렬
        from collections import Counter
        word_counts = Counter(words)
        # stopwords 제외
        stopwords = {'this', 'that', 'with', 'from', 'have', 'were', 'been', 'their', 'which', 'would', 'could', 'should', 'there', 'where', 'when', 'what', 'about', 'into', 'more', 'some', 'also', 'than', 'them', 'then', 'only', 'over', 'such', 'just', 'like', 'being', 'other', 'very', 'after', 'most', 'make', 'made', 'well', 'back', 'even', 'want', 'give', 'because', 'these', 'first', 'your', 'said'}
        filtered = [(w, c) for w, c in word_counts.most_common(top_n * 2) if w not in stopwords]
        return set(w for w, c in filtered[:top_n])

    def check_keyword_overlap(original: str, generated: str) -> tuple:
        """원문과 생성문의 키워드 겹침 확인"""
        orig_kw = extract_keywords(original)
        gen_kw = extract_keywords(generated)
        if not orig_kw:
            return 0.0, set(), set()
        overlap = orig_kw & gen_kw
        ratio = len(overlap) / len(orig_kw)
        return ratio, overlap, orig_kw

    # 몇 개 샘플 생성
    num_samples = min(3, len(doc_ids_list))
    total_keyword_score = 0.0

    logger.info(f"Generating {num_samples} validation samples...")
    for i in tqdm(range(num_samples), desc="Validation samples"):
        doc_id = doc_ids_list[i]
        z_i = z_pool_manager.get_z(doc_id).to(model.device)

        # z_i 통계
        logger.info(f"\n--- Sample {i+1}: {doc_id} ---")
        logger.info(f"z_i shape: {z_i.shape}, dtype: {z_i.dtype}")
        logger.info(f"z_i stats: mean={z_i.mean():.4f}, std={z_i.std():.4f}, norm={z_i.norm():.4f}")

        # 샘플링과 greedy 둘 다 테스트
        generated_sample = model.generate_from_z(z_i, max_new_tokens=128, do_sample=True)
        generated_greedy = model.generate_from_z(z_i, max_new_tokens=128, do_sample=False)
        original = corpus[doc_id]

        # z_embed 통계
        stats = model.get_z_embed_stats(z_i)

        logger.info(f"z_embed stats: norm={stats['z_embed_norm']:.4f}, mean={stats['z_embed_mean']:.4f}, std={stats['z_embed_std']:.4f}")
        logger.info(f"Original (first 200 chars): {original[:200]}...")
        logger.info(f"Generated (sampling): {generated_sample[:200]}...")
        logger.info(f"Generated (greedy):   {generated_greedy[:200]}...")

        # Keyword overlap check (objective verification)
        kw_ratio, overlap, orig_kw = check_keyword_overlap(original, generated_sample)
        total_keyword_score += kw_ratio
        logger.info(f"Keyword check: {kw_ratio:.1%} overlap ({len(overlap)}/{len(orig_kw)})")
        logger.info(f"  Original keywords: {list(orig_kw)[:5]}...")
        logger.info(f"  Matched keywords:  {list(overlap)}")

    # 전체 keyword 점수
    avg_keyword_score = total_keyword_score / num_samples if num_samples > 0 else 0
    logger.info(f"\n[Objective] Avg keyword overlap: {avg_keyword_score:.1%}")
    results["avg_keyword_score"] = avg_keyword_score

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
    args = parser.parse_args()

    run_write_phase_training(config_path=args.config, test_mode=args.test)


if __name__ == "__main__":
    main()
