"""
End-to-End 학습 스크립트
- Write Phase → Read Phase 순차 실행
- 또는 joint training (optional)
"""

import sys
import logging
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).parent.parent))

from training.train_write import run_write_phase
from training.train_read import run_read_phase
from evaluation.evaluate_qa import evaluate_qa
from evaluation.evaluate_write import evaluate_reconstruction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_end_to_end(config_path: str = None, config: dict = None):
    """
    End-to-End 실행: Write Phase → Read Phase → Final Evaluation

    이 스크립트는 Phase 1 (POC)에서 사용.
    Phase 3 (Full Eval)에서는 각 단계를 별도로 실행하는 것을 추천.
    """
    # Config
    if config is None:
        with open(config_path, "r") as f:
            config = OmegaConf.create(yaml.safe_load(f))

    logger.info("=" * 60)
    logger.info("End-to-End Training Pipeline")
    logger.info("=" * 60)

    # ==========================================
    # Stage 1: Write Phase
    # ==========================================
    logger.info("\n" + "=" * 40)
    logger.info("STAGE 1: Write Phase")
    logger.info("=" * 40)

    model, write_results = run_write_phase(config=config)

    write_metrics = write_results["final_eval"]
    logger.info(f"Write Phase Done: PPL={write_metrics.get('avg_perplexity', 'N/A')}")

    # Success criteria check
    if write_metrics.get("avg_perplexity", float("inf")) > 50:
        logger.warning("Write Phase perplexity too high! Consider adjusting z_dim or m_tokens.")

    # ==========================================
    # Stage 2: Read Phase
    # ==========================================
    logger.info("\n" + "=" * 40)
    logger.info("STAGE 2: Read Phase")
    logger.info("=" * 40)

    model, read_results = run_read_phase(
        config=config,
        pretrained_model=model,
    )

    read_metrics = read_results["final_eval"]
    logger.info(f"Read Phase Done: EM={read_metrics.get('em', 'N/A')}, "
                f"F1={read_metrics.get('f1', 'N/A')}")

    # ==========================================
    # Final Summary
    # ==========================================
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info("=" * 60)

    summary = {
        "write_phase": {
            "avg_perplexity": write_metrics.get("avg_perplexity"),
            "avg_loss": write_metrics.get("avg_loss"),
        },
        "read_phase": {
            "em": read_metrics.get("em"),
            "f1": read_metrics.get("f1"),
            "recall_at_5": read_metrics.get("recall_at_5"),
        },
    }

    for phase, metrics in summary.items():
        logger.info(f"  {phase}:")
        for k, v in metrics.items():
            if v is not None:
                logger.info(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # Success criteria
    logger.info("\nSuccess Criteria Check:")
    ppl = write_metrics.get("avg_perplexity", float("inf"))
    em = read_metrics.get("em", 0)
    recall = read_metrics.get("recall_at_5", 0)

    checks = [
        ("Reconstruction PPL < 10", ppl < 10),
        ("Selection Recall@5 > 50%", recall > 0.5),
        ("QA EM > 15%", em > 0.15),
    ]

    for desc, passed in checks:
        status = "PASS" if passed else "FAIL"
        logger.info(f"  [{status}] {desc}")

    # Save
    save_dir = config.logging.save_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    torch.save({
        "summary": summary,
        "write_results": write_results,
        "read_results": read_results,
        "config": OmegaConf.to_container(config),
    }, f"{save_dir}/e2e_results.pt")

    return model, summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase1_poc.yaml")
    args = parser.parse_args()

    run_end_to_end(config_path=args.config)
