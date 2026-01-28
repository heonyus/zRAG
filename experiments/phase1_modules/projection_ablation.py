"""
B2: Projection Ablation Module

Tests different projection configurations:
- proj_normal: Use trained projection
- proj_frozen_ckpt: Same as normal (frozen during eval anyway)
- proj_random_frozen: Random projection weights

EVAL-ONLY: No training is performed.
"""

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .utils import Timer, get_logger, save_json, temporary_projection
from .visualization import plot_ablation_comparison
from .confusion_matrix import run_confusion_metrics_only
from .reports import generate_eyeball_report
from .zonly_generation import generate_from_z_only, compute_rouge_l


def run_projection_ablation(
    model,
    z_pool,
    corpus: Dict[str, str],
    tokenizer,
    run_dir: Path,
    ckpt_dir: Optional[Path] = None,
    num_docs: int = 200,
    max_eval_tokens: int = 256,
    max_new_tokens: int = 256,
    device: str = "cuda",
    use_amp: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Run B2: Projection Ablation study.

    Tests three projection configurations:
    - proj_normal: Trained projection (baseline)
    - proj_frozen_ckpt: Same as trained (frozen during eval)
    - proj_random_frozen: Randomly initialized projection

    Args:
        model: WritePhaseModel instance
        z_pool: ZPoolManager instance
        corpus: Dict mapping doc_id -> document text
        tokenizer: Tokenizer
        run_dir: Root run directory
        ckpt_dir: Checkpoint directory (for loading projection)
        num_docs: Number of documents
        max_eval_tokens: Maximum tokens for NLL
        max_new_tokens: Maximum tokens for generation
        device: Device
        use_amp: Use AMP

    Returns:
        Dictionary mapping variant_name -> metrics
    """
    logger = get_logger()
    out_dir = run_dir / "02_ablations" / "B2_projection"
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_ids = z_pool.doc_ids[:num_docs]
    n_docs = len(doc_ids)

    logger.info(f"[B2] Starting Projection Ablation: {n_docs} docs")

    # Save original projection state
    original_proj_state = copy.deepcopy(model.z_to_embedding.state_dict())

    # Define variants
    variants = ["proj_normal", "proj_frozen_ckpt", "proj_random_frozen"]

    results = {}

    for variant_name in variants:
        variant_dir = out_dir / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        samples_dir = variant_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[B2] Running variant: {variant_name}")

        # Modify projection based on variant
        if variant_name == "proj_random_frozen":
            # Reinitialize with random weights
            logger.info("[B2] Reinitializing projection with random weights...")
            for name, param in model.z_to_embedding.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
        else:
            # Use original trained projection
            model.z_to_embedding.load_state_dict(original_proj_state)

        # Save variant config
        variant_config = {
            "module": "B2_projection",
            "variant": variant_name,
            "num_docs": n_docs,
            "max_eval_tokens": max_eval_tokens,
        }
        save_json(variant_config, variant_dir / "variant_config.json")

        # Run confusion metrics
        with Timer(f"B2 {variant_name} confusion"):
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

        # Add variant info
        metrics["variant"] = variant_name

        # Save metrics
        save_json(metrics, variant_dir / "metrics.json")

        # Summary row
        summary_row = {
            "module": "B2_projection",
            "variant": variant_name,
            "top1_acc": metrics["top1_acc"],
            "top5_acc": metrics["top5_acc"],
            "mean_margin": metrics["mean_margin"],
        }
        save_json(summary_row, variant_dir / "summary_row.json")

        # For random_frozen, generate degradation examples
        if variant_name == "proj_random_frozen":
            logger.info("[B2] Generating degradation examples for random projection...")
            degradation_samples = []

            # Generate for first 10 docs
            sample_doc_ids = doc_ids[:10]
            for doc_id in sample_doc_ids:
                z_i = z_pool.get_z(doc_id).to(device)
                ref_text = corpus.get(doc_id, "")

                try:
                    gen_text = generate_from_z_only(
                        model, z_i, max_new_tokens=max_new_tokens
                    )
                except Exception as e:
                    logger.warning(f"[B2] Generation failed for {doc_id}: {e}")
                    gen_text = "[Generation failed]"

                score = compute_rouge_l(gen_text, ref_text)

                degradation_samples.append({
                    "doc_id": doc_id,
                    "score": score,
                    "ref_snippet": ref_text[:400],
                    "gen_snippet": gen_text[:400],
                })

            # Generate eyeball report
            generate_eyeball_report(
                degradation_samples,
                samples_dir / "eyeball_10_examples.md",
                title="Random Projection Degradation Examples",
                description=(
                    "These examples show text generated with a RANDOM (untrained) projection layer. "
                    "The output should be incoherent garbage, demonstrating that the trained "
                    "projection is essential for meaningful generation."
                ),
            )

            metrics["degradation_samples_count"] = len(degradation_samples)
            metrics["degradation_mean_rouge"] = (
                sum(s["score"] for s in degradation_samples) / len(degradation_samples)
                if degradation_samples else 0
            )

        results[variant_name] = metrics

        logger.info(
            f"[B2] {variant_name}: Top-1={metrics['top1_acc']*100:.1f}%, "
            f"Margin={metrics['mean_margin']:.3f}"
        )

    # Restore original projection
    model.z_to_embedding.load_state_dict(original_proj_state)

    # Generate comparison visualization
    logger.info("[B2] Generating comparison plot...")
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plot_ablation_comparison(
        results,
        artifacts_dir,
        metric_key="top1_acc",
        title="Projection Ablation: Top-1 Accuracy by Projection Type"
    )

    # Save overall summary
    summary = {
        "module": "B2_projection",
        "variants": {k: v["top1_acc"] for k, v in results.items()},
        "best_variant": max(results.keys(), key=lambda k: results[k]["top1_acc"]),
    }
    save_json(summary, out_dir / "ablation_summary.json")

    # Log interpretation
    normal_acc = results.get("proj_normal", {}).get("top1_acc", 0)
    random_acc = results.get("proj_random_frozen", {}).get("top1_acc", 0)

    if random_acc > 0.2:
        logger.warning(
            f"[B2] WARNING: Random projection has {random_acc*100:.1f}% accuracy. "
            "This is higher than expected for random initialization."
        )

    if normal_acc > random_acc + 0.3:
        logger.info(
            f"[B2] Good: Trained projection improves accuracy by "
            f"{(normal_acc - random_acc)*100:.1f}% over random"
        )

    logger.info("[B2] Projection ablation complete")

    return results
