"""
B1: Alpha Ablation Module

Tests different alpha values to understand its importance:
- alpha_normal: Use trained alpha value
- alpha_zero: Set alpha = 0 (z has no effect)
- alpha_fixed1: Set alpha = 1.0 (no learned scaling)

EVAL-ONLY: No training is performed.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .utils import Timer, get_logger, save_json, temporary_alpha
from .visualization import plot_ablation_comparison
from .confusion_matrix import run_confusion_metrics_only


def run_alpha_ablation(
    model,
    z_pool,
    corpus: Dict[str, str],
    tokenizer,
    run_dir: Path,
    num_docs: int = 200,
    max_eval_tokens: int = 256,
    device: str = "cuda",
    use_amp: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Run B1: Alpha Ablation study.

    Tests three alpha configurations:
    - alpha_normal: Trained alpha value
    - alpha_zero: alpha = 0 (z contributes nothing)
    - alpha_fixed1: alpha = 1.0 (no learned scaling)

    Args:
        model: WritePhaseModel instance
        z_pool: ZPoolManager instance
        corpus: Dict mapping doc_id -> document text
        tokenizer: Tokenizer
        run_dir: Root run directory
        num_docs: Number of documents
        max_eval_tokens: Maximum tokens
        device: Device
        use_amp: Use AMP

    Returns:
        Dictionary mapping variant_name -> metrics
    """
    logger = get_logger()
    out_dir = run_dir / "02_ablations" / "B1_alpha"
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_ids = z_pool.doc_ids[:num_docs]
    n_docs = len(doc_ids)

    # Get original trained alpha
    original_alpha = model.alpha.item()
    logger.info(f"[B1] Starting Alpha Ablation: {n_docs} docs, trained_alpha={original_alpha:.4f}")

    # Define variants
    variants = {
        "alpha_normal": original_alpha,
        "alpha_zero": 0.0,
        "alpha_fixed1": 1.0,
    }

    results = {}

    for variant_name, alpha_value in variants.items():
        variant_dir = out_dir / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[B1] Running variant: {variant_name} (alpha={alpha_value})")

        # Save variant config
        variant_config = {
            "module": "B1_alpha",
            "variant": variant_name,
            "alpha_value": alpha_value,
            "original_alpha": original_alpha,
            "num_docs": n_docs,
            "max_eval_tokens": max_eval_tokens,
        }
        save_json(variant_config, variant_dir / "variant_config.json")

        # Run evaluation with modified alpha
        with Timer(f"B1 {variant_name}"):
            with temporary_alpha(model, alpha_value):
                metrics = run_confusion_metrics_only(
                    model,
                    z_pool,
                    corpus,
                    tokenizer,
                    num_docs=num_docs,
                    max_eval_tokens=max_eval_tokens,
                    device=device,
                    use_amp=use_amp,
                    doc_ids=doc_ids,
                )

        # Add variant info to metrics
        metrics["variant"] = variant_name
        metrics["alpha_value"] = alpha_value

        # Save metrics
        save_json(metrics, variant_dir / "metrics.json")

        # Summary row
        summary_row = {
            "module": "B1_alpha",
            "variant": variant_name,
            "alpha_value": alpha_value,
            "top1_acc": metrics["top1_acc"],
            "top5_acc": metrics["top5_acc"],
            "mean_margin": metrics["mean_margin"],
        }
        save_json(summary_row, variant_dir / "summary_row.json")

        results[variant_name] = metrics

        logger.info(
            f"[B1] {variant_name}: Top-1={metrics['top1_acc']*100:.1f}%, "
            f"Margin={metrics['mean_margin']:.3f}"
        )

    # Generate comparison visualization
    logger.info("[B1] Generating comparison plot...")
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plot_ablation_comparison(
        results,
        artifacts_dir,
        metric_key="top1_acc",
        title="Alpha Ablation: Top-1 Accuracy by Alpha Value"
    )

    # Save overall summary
    summary = {
        "module": "B1_alpha",
        "original_alpha": original_alpha,
        "variants": {k: v["top1_acc"] for k, v in results.items()},
        "best_variant": max(results.keys(), key=lambda k: results[k]["top1_acc"]),
    }
    save_json(summary, out_dir / "ablation_summary.json")

    # Log interpretation
    alpha_zero_acc = results.get("alpha_zero", {}).get("top1_acc", 0)
    alpha_normal_acc = results.get("alpha_normal", {}).get("top1_acc", 0)

    if alpha_zero_acc > 0.5:
        logger.warning(
            f"[B1] WARNING: alpha=0 has {alpha_zero_acc*100:.1f}% accuracy. "
            "z vectors may not be contributing much to predictions."
        )

    if alpha_normal_acc > alpha_zero_acc + 0.1:
        logger.info(
            f"[B1] Good: Trained alpha improves accuracy by "
            f"{(alpha_normal_acc - alpha_zero_acc)*100:.1f}% over alpha=0"
        )

    logger.info("[B1] Alpha ablation complete")

    return results
