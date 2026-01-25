"""
Write Phase 학습 스크립트
- Stage 1: z_i 학습 (LLM frozen)
- 목표: max log P(D_i | z_i; θ)
"""

import os
import sys
import logging
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).parent.parent))

from data.download import download_dataset
from data.preprocess import preprocess_nq, preprocess_hotpotqa
from data.dataloader import create_write_dataloader
from models.parametric_qa import ParametricQA
from models.write_phase import WritePhaseTrainer
from evaluation.evaluate_write import evaluate_reconstruction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_write_phase(config_path: str = None, config: dict = None):
    """Write Phase 실행"""

    # Config 로드
    if config is None:
        with open(config_path, "r") as f:
            config = OmegaConf.create(yaml.safe_load(f))

    logger.info("=" * 60)
    logger.info("Write Phase Training")
    logger.info("=" * 60)
    logger.info(f"Config: {OmegaConf.to_yaml(config)}")

    # Device
    device = config.hardware.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ==========================================
    # 1. Data Preparation
    # ==========================================
    logger.info("Loading dataset...")
    dataset = download_dataset(config.data.dataset)

    if config.data.dataset == "natural_questions":
        processed = preprocess_nq(
            dataset,
            corpus_size=config.data.corpus_size,
            max_samples=config.data.train_samples,
        )
    elif config.data.dataset == "hotpot_qa":
        processed = preprocess_hotpotqa(
            dataset,
            corpus_size=config.data.corpus_size,
            max_samples=config.data.train_samples,
        )
    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset}")

    corpus = processed["corpus"]
    logger.info(f"Corpus size: {len(corpus)} documents")
    logger.info(f"Stats: {processed['stats']}")

    # ==========================================
    # 2. Model Initialization
    # ==========================================
    logger.info("Initializing ParametricQA model...")
    model = ParametricQA(
        llm_name=config.model.llm_name,
        num_docs=len(corpus),
        z_dim=config.parametric_qa.z_dim,
        m_tokens=config.parametric_qa.m_tokens,
        selection_method=config.parametric_qa.selection_method,
        quantization=config.model.quantization,
        lora_r=config.model.lora.r,
        lora_alpha=config.model.lora.alpha,
        lora_target_modules=list(config.model.lora.target_modules),
        lora_dropout=config.model.lora.dropout,
        device=device,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,} "
                f"({100*trainable_params/total_params:.2f}%)")

    # ==========================================
    # 3. DataLoader
    # ==========================================
    logger.info("Creating dataloader...")
    dataloader = create_write_dataloader(
        corpus=corpus,
        tokenizer=model.tokenizer,
        batch_size=config.write_phase.batch_size,
        max_length=config.data.max_doc_length,
        shuffle=True,
    )
    logger.info(f"DataLoader: {len(dataloader)} batches")

    # ==========================================
    # 4. Training
    # ==========================================
    save_dir = config.logging.save_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    trainer = WritePhaseTrainer(
        model=model,
        lr=config.write_phase.lr,
        llm_frozen=config.write_phase.llm_frozen,
        device=device,
    )

    # Evaluation function
    def eval_fn(model):
        return evaluate_reconstruction(
            model=model,
            corpus=corpus,
            tokenizer=model.tokenizer,
            num_samples=min(50, len(corpus)),
            device=device,
        )

    # Train
    logger.info("Starting Write Phase training...")
    metrics = trainer.train(
        dataloader=dataloader,
        epochs=config.write_phase.epochs,
        log_every=config.write_phase.log_every,
        eval_fn=eval_fn,
        save_path=save_dir,
    )

    # ==========================================
    # 5. Final Evaluation
    # ==========================================
    logger.info("Final evaluation...")
    final_eval = eval_fn(model)
    logger.info(f"Final Write Phase Results: {final_eval}")

    # Save final results
    results = {
        "config": OmegaConf.to_container(config),
        "training_metrics": metrics,
        "final_eval": final_eval,
    }
    torch.save(results, f"{save_dir}/write_phase_results.pt")

    logger.info("Write Phase complete!")
    return model, results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase1_poc.yaml")
    args = parser.parse_args()

    run_write_phase(config_path=args.config)
