"""
Evidence 생성 학습 메인 스크립트

학습 목표: -log P(evidence | query, Z; θ)

사용법:
    python training/train_evidence.py --config configs/evidence_poc.yaml
"""

import sys
import logging
import argparse
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf

# Path setup
sys.path.append(str(Path(__file__).parent.parent))

from models.parametric_memory_llm import ParametricMemoryLLM
from models.evidence_trainer import EvidenceTrainer
from data.download import download_dataset
from data.evidence_dataloader import (
    create_evidence_dataloader,
    prepare_evidence_pairs_from_nq,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run_evidence_training(config_path: str = None, config: dict = None):
    """
    Evidence 생성 학습 실행

    Args:
        config_path: YAML config 파일 경로
        config: config dict (직접 전달 시)

    Returns:
        model: 학습된 ParametricMemoryLLM
        results: 학습 결과 metrics
    """
    # Load config
    if config is None:
        with open(config_path, "r") as f:
            config = OmegaConf.create(yaml.safe_load(f))

    logger.info("=" * 60)
    logger.info("Evidence Generation Training")
    logger.info("=" * 60)
    logger.info(f"Config: {OmegaConf.to_yaml(config)}")

    # ==========================================
    # 1. Data Preparation
    # ==========================================
    logger.info("\n[Step 1] Data Preparation")

    # Download dataset
    data_config = config.data
    raw_data = download_dataset(
        dataset_name=data_config.dataset,
        save_dir=data_config.get("save_dir", "./data/raw"),
    )

    # Prepare corpus
    corpus = raw_data.get("corpus", {})
    if len(corpus) > config.memory.num_docs:
        # Corpus 크기 제한
        corpus = {k: v for k, v in list(corpus.items())[:config.memory.num_docs]}
        logger.info(f"Corpus truncated to {len(corpus)} documents")

    # Prepare evidence pairs
    qa_pairs = raw_data.get("qa_pairs", raw_data.get("train", []))
    evidence_pairs = prepare_evidence_pairs_from_nq(
        {"qa_pairs": qa_pairs, "corpus": corpus},
        corpus,
        max_samples=data_config.get("max_samples"),
    )

    logger.info(f"Prepared {len(evidence_pairs)} evidence pairs")
    logger.info(f"Corpus size: {len(corpus)} documents")

    # Train/eval split
    split_ratio = data_config.get("train_split", 0.9)
    split_idx = int(len(evidence_pairs) * split_ratio)
    train_pairs = evidence_pairs[:split_idx]
    eval_pairs = evidence_pairs[split_idx:]

    logger.info(f"Train: {len(train_pairs)}, Eval: {len(eval_pairs)}")

    # ==========================================
    # 2. Model Initialization
    # ==========================================
    logger.info("\n[Step 2] Model Initialization")

    model_config = config.model
    memory_config = config.memory

    model = ParametricMemoryLLM(
        llm_name=model_config.llm_name,
        num_docs=memory_config.num_docs,
        z_dim=memory_config.z_dim,
        m_tokens=memory_config.m_tokens,
        quantization=model_config.get("quantization", "4bit"),
        lora_r=model_config.lora.get("r", 32),
        lora_alpha=model_config.lora.get("alpha", 64),
        lora_target_modules=model_config.lora.get("target_modules"),
        lora_dropout=model_config.lora.get("dropout", 0.05),
    )

    # Memory stats
    mem_stats = model.get_memory_stats()
    logger.info(f"Memory Pool: {mem_stats['num_docs']} docs × {mem_stats['m_tokens']} tokens")
    logger.info(f"Memory Pool size: {mem_stats['memory_pool_mb']:.2f} MB")
    logger.info(f"Z embed tokens: {mem_stats['z_embed_tokens']} tokens")

    # ==========================================
    # 3. DataLoader
    # ==========================================
    logger.info("\n[Step 3] DataLoader Creation")

    train_config = config.training

    train_dataloader = create_evidence_dataloader(
        qa_pairs=train_pairs,
        tokenizer=model.tokenizer,
        batch_size=train_config.batch_size,
        max_query_length=data_config.get("max_query_length", 128),
        max_evidence_length=data_config.get("max_evidence_length", 256),
        shuffle=True,
    )

    eval_dataloader = create_evidence_dataloader(
        qa_pairs=eval_pairs,
        tokenizer=model.tokenizer,
        batch_size=train_config.batch_size,
        max_query_length=data_config.get("max_query_length", 128),
        max_evidence_length=data_config.get("max_evidence_length", 256),
        shuffle=False,
    )

    logger.info(f"Train batches: {len(train_dataloader)}")
    logger.info(f"Eval batches: {len(eval_dataloader)}")

    # ==========================================
    # 4. Trainer
    # ==========================================
    logger.info("\n[Step 4] Trainer Setup")

    trainer = EvidenceTrainer(
        model=model,
        lr_llm=train_config.get("lr_llm", 2e-5),
        lr_z=train_config.get("lr_z", 1e-3),
        lr_proj=train_config.get("lr_proj", 1e-4),
        warmup_ratio=train_config.get("warmup_ratio", 0.1),
        gradient_accumulation_steps=train_config.get("gradient_accumulation", 8),
        max_grad_norm=train_config.get("max_grad_norm", 1.0),
        use_amp=train_config.get("use_amp", True),
    )

    # ==========================================
    # 5. Training
    # ==========================================
    logger.info("\n[Step 5] Training")

    save_dir = config.logging.get("save_dir", "./checkpoints/evidence")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    results = trainer.train(
        train_dataloader=train_dataloader,
        epochs=train_config.epochs,
        eval_dataloader=eval_dataloader,
        eval_steps=train_config.get("eval_steps", 500),
        save_path=save_dir,
        log_steps=train_config.get("log_steps", 50),
    )

    # ==========================================
    # 6. Sample Generation
    # ==========================================
    logger.info("\n[Step 6] Sample Generation")

    samples = trainer.generate_samples(
        eval_dataloader,
        num_samples=5,
        max_new_tokens=128,
    )

    logger.info("Generated samples:")
    for i, sample in enumerate(samples):
        logger.info(f"\n--- Sample {i + 1} ---")
        logger.info(f"Query: {sample['query'][:100]}...")
        logger.info(f"Gold: {sample['gold_evidence'][:100]}...")
        logger.info(f"Generated: {sample['generated_evidence'][:100]}...")

    # ==========================================
    # 7. Final Evaluation
    # ==========================================
    logger.info("\n[Step 7] Final Evaluation")

    final_eval_loss = trainer.evaluate(eval_dataloader)
    logger.info(f"Final eval loss: {final_eval_loss:.4f}")

    results["final_eval_loss"] = final_eval_loss
    results["samples"] = samples

    # Save final checkpoint
    final_path = f"{save_dir}/final.pt"
    model.save_checkpoint(final_path)
    logger.info(f"Saved final checkpoint: {final_path}")

    # Save results
    torch.save({
        "results": results,
        "config": OmegaConf.to_container(config),
    }, f"{save_dir}/results.pt")

    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Final train loss: {results['final_train_loss']:.4f}")
    logger.info(f"Final eval loss: {final_eval_loss:.4f}")
    logger.info("=" * 60)

    return model, results


def main():
    """CLI entry point for evidence training."""
    parser = argparse.ArgumentParser(description="Evidence Generation Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/evidence_poc.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    run_evidence_training(config_path=args.config)


if __name__ == "__main__":
    main()
