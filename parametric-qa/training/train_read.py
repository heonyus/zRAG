"""
Read Phase 학습 스크립트
- Stage 2: QA Fine-tuning (LLM + z_i + projections)
- 목표: max log P(answer | q, z_selected; θ)
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
from data.dataloader import create_read_dataloader
from models.parametric_qa import ParametricQA
from models.read_phase import ReadPhaseTrainer
from evaluation.evaluate_qa import evaluate_qa

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_read_phase(
    config_path: str = None,
    config: dict = None,
    pretrained_model: ParametricQA = None,
    write_checkpoint: str = None,
):
    """Read Phase 실행"""

    # Config 로드
    if config is None:
        with open(config_path, "r") as f:
            config = OmegaConf.create(yaml.safe_load(f))

    logger.info("=" * 60)
    logger.info("Read Phase Training")
    logger.info("=" * 60)

    device = config.hardware.device if torch.cuda.is_available() else "cpu"

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

    qa_pairs = processed["qa_pairs"]
    corpus = processed["corpus"]
    logger.info(f"QA pairs: {len(qa_pairs)}, Corpus: {len(corpus)} docs")

    # Train/eval split
    split_idx = int(len(qa_pairs) * 0.9)
    train_qa = qa_pairs[:split_idx]
    eval_qa = qa_pairs[split_idx:]
    logger.info(f"Train: {len(train_qa)}, Eval: {len(eval_qa)}")

    # ==========================================
    # 2. Model Loading
    # ==========================================
    if pretrained_model is not None:
        model = pretrained_model
        logger.info("Using pre-trained model from Write Phase")
    elif write_checkpoint:
        logger.info(f"Loading Write Phase checkpoint: {write_checkpoint}")
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
        model.load_checkpoint(write_checkpoint)
    else:
        logger.info("Initializing fresh model (no Write Phase checkpoint)")
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

    # ==========================================
    # 3. DataLoader
    # ==========================================
    train_dataloader = create_read_dataloader(
        qa_pairs=train_qa,
        tokenizer=model.tokenizer,
        batch_size=config.read_phase.batch_size,
        max_query_length=config.data.max_query_length,
        shuffle=True,
    )
    logger.info(f"Train DataLoader: {len(train_dataloader)} batches")

    # ==========================================
    # 4. Training
    # ==========================================
    save_dir = config.logging.save_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    trainer = ReadPhaseTrainer(
        model=model,
        lr_llm=config.read_phase.lr_llm,
        lr_z=config.read_phase.lr_z,
        lr_proj=config.read_phase.lr_proj,
        alpha_retrieval=config.read_phase.alpha_retrieval,
        top_k=config.read_phase.top_k,
        gradient_accumulation_steps=config.read_phase.gradient_accumulation,
        warmup_ratio=config.read_phase.warmup_ratio,
        device=device,
    )

    # Evaluation function
    def eval_fn(model):
        return evaluate_qa(
            model=model,
            qa_pairs=eval_qa,
            tokenizer=model.tokenizer,
            top_k=config.read_phase.top_k,
            max_samples=min(100, len(eval_qa)),
            device=device,
        )

    # Train
    logger.info("Starting Read Phase training...")
    metrics = trainer.train(
        dataloader=train_dataloader,
        epochs=config.read_phase.epochs,
        log_every=50,
        eval_fn=eval_fn,
        save_path=save_dir,
    )

    # ==========================================
    # 5. Final Evaluation
    # ==========================================
    logger.info("Final evaluation...")
    final_eval = evaluate_qa(
        model=model,
        qa_pairs=eval_qa,
        tokenizer=model.tokenizer,
        top_k=config.read_phase.top_k,
        device=device,
    )
    logger.info(f"Final Read Phase Results: {final_eval}")

    # Save
    results = {
        "config": OmegaConf.to_container(config),
        "training_metrics": metrics,
        "final_eval": final_eval,
    }
    torch.save(results, f"{save_dir}/read_phase_results.pt")

    logger.info("Read Phase complete!")
    return model, results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase1_poc.yaml")
    parser.add_argument("--write_checkpoint", type=str, default=None)
    args = parser.parse_args()

    run_read_phase(
        config_path=args.config,
        write_checkpoint=args.write_checkpoint,
    )
