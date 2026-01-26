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
    # 4. Training: Document-wise z_i Optimization
    # ==========================================
    logger.info("\n[Step 4] Training z_i for each document")

    train_config = config.training
    use_amp = train_config.get("use_amp", True)
    scaler = GradScaler('cuda') if use_amp else None

    # Training config 로깅
    lr_z = float(train_config.get("lr_z", 1e-2))
    lr_proj = float(train_config.get("lr_proj", 0))
    epochs_per_doc = train_config.get("epochs_per_doc", 100)
    logger.info(f"Training config: lr_z={lr_z}, lr_proj={lr_proj}, epochs_per_doc={epochs_per_doc}")
    logger.info(f"Projection: {'FROZEN' if lr_proj == 0 else f'learning (lr={lr_proj})'}")

    results = {
        "losses": {},
        "num_docs": len(corpus),
        "config": OmegaConf.to_container(config),
    }

    save_dir = Path(config.logging.get("save_dir", "./checkpoints/phase1_write"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # 문서별 학습
    doc_ids_list = list(tokenized_docs.keys())

    for i, doc_id in enumerate(tqdm(doc_ids_list, desc="Training z_i")):
        doc_data = tokenized_docs[doc_id]

        z_i, loss = train_single_document(
            model=model,
            doc_id=doc_id,
            doc_ids=doc_data["input_ids"],
            doc_attention_mask=doc_data["attention_mask"],
            config=train_config,
            scaler=scaler,
        )

        # z_pool에 추가
        z_pool_manager.add_z(doc_id, z_i)
        results["losses"][doc_id] = loss

        # 진행 상황 로깅
        if (i + 1) % 10 == 0 or (i + 1) == len(doc_ids_list):
            avg_loss = sum(results["losses"].values()) / len(results["losses"])
            logger.info(f"Progress: {i+1}/{len(doc_ids_list)}, Avg Loss: {avg_loss:.4f}")

        # 중간 저장 (매 100개)
        if (i + 1) % 100 == 0:
            z_pool_manager.save(save_dir / f"z_pool_checkpoint_{i+1}.pt")

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
    # 6. Validation: Generate from z
    # ==========================================
    logger.info("\n[Step 6] Validation: Generate from learned z")

    # Projection 상태 확인
    proj_frozen = not any(p.requires_grad for p in model.z_to_embedding.parameters())
    logger.info(f"Projection layer: {'FROZEN' if proj_frozen else 'TRAINABLE'}")

    # z_pool 통계
    z_pool_tensor = z_pool_manager.get_pool_tensor()
    logger.info(f"z_pool shape: {z_pool_tensor.shape}")
    logger.info(f"z_pool stats: mean={z_pool_tensor.mean():.4f}, std={z_pool_tensor.std():.4f}, "
               f"min={z_pool_tensor.min():.4f}, max={z_pool_tensor.max():.4f}")

    # 몇 개 샘플 생성
    num_samples = min(3, len(doc_ids_list))
    for i in range(num_samples):
        doc_id = doc_ids_list[i]
        z_i = z_pool_manager.get_z(doc_id).to(model.device)

        # z_i 통계
        logger.info(f"\n--- Sample {i+1}: {doc_id} ---")
        logger.info(f"z_i shape: {z_i.shape}, dtype: {z_i.dtype}")
        logger.info(f"z_i stats: mean={z_i.mean():.4f}, std={z_i.std():.4f}, norm={z_i.norm():.4f}")

        # 샘플링과 greedy 둘 다 테스트
        generated_sample = model.generate_from_z(z_i, max_new_tokens=128, do_sample=True)
        generated_greedy = model.generate_from_z(z_i, max_new_tokens=128, do_sample=False)
        original = corpus[doc_id][:200]

        # z_embed 통계
        stats = model.get_z_embed_stats(z_i)

        logger.info(f"z_embed stats: norm={stats['z_embed_norm']:.4f}, mean={stats['z_embed_mean']:.4f}, std={stats['z_embed_std']:.4f}")
        logger.info(f"Original (first 200 chars): {original}...")
        logger.info(f"Generated (sampling): {generated_sample[:200]}...")
        logger.info(f"Generated (greedy):   {generated_greedy[:200]}...")

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
