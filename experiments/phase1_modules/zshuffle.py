"""
A3: Z-shuffle Sanity Check Module

Permutes z-to-doc mapping and reruns confusion metrics to verify
that performance degrades (proving z encodes document-specific information).

EVAL-ONLY: No training is performed.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .utils import Timer, get_logger, load_json, save_json
from .visualization import plot_delta_barplot
from .confusion_matrix import run_confusion_metrics_only


class ShuffledZPool:
    """
    A wrapper around ZPoolManager that returns z vectors in shuffled order.
    """

    def __init__(self, original_z_pool, permutation: np.ndarray):
        """
        Args:
            original_z_pool: Original ZPoolManager instance
            permutation: Permutation array mapping original index -> shuffled index
        """
        self.original_z_pool = original_z_pool
        self.permutation = permutation
        self.inverse_perm = np.argsort(permutation)  # For reverse lookup

        # Keep doc_ids in original order (important for evaluation)
        self.doc_ids = original_z_pool.doc_ids

    def get_z(self, doc_id: str) -> torch.Tensor:
        """
        Get z vector for a document, but return a DIFFERENT z (shuffled).

        If doc_i asks for its z, we return z_j where j = permutation[i].
        """
        # Find original index
        if hasattr(self.original_z_pool, "doc_id_to_idx"):
            original_idx = self.original_z_pool.doc_id_to_idx.get(doc_id, 0)
        else:
            try:
                original_idx = self.original_z_pool.doc_ids.index(doc_id)
            except ValueError:
                original_idx = 0

        # Get shuffled index
        shuffled_idx = self.permutation[original_idx]

        # Get the z from the shuffled position
        shuffled_doc_id = self.original_z_pool.doc_ids[shuffled_idx]
        return self.original_z_pool.get_z(shuffled_doc_id)


def run_zshuffle_sanity(
    model,
    z_pool,
    corpus: Dict[str, str],
    tokenizer,
    run_dir: Path,
    baseline_metrics: Optional[Dict[str, Any]] = None,
    num_docs: int = 200,
    max_eval_tokens: int = 256,
    device: str = "cuda",
    use_amp: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run A3: Z-shuffle Sanity Check.

    Args:
        model: WritePhaseModel instance
        z_pool: ZPoolManager instance
        corpus: Dict mapping doc_id -> document text
        tokenizer: Tokenizer
        run_dir: Root run directory
        baseline_metrics: A1 baseline metrics (if None, will load from A1_confusion)
        num_docs: Number of documents
        max_eval_tokens: Maximum tokens
        device: Device
        use_amp: Use AMP
        seed: Random seed for shuffle

    Returns:
        Comparison metrics dictionary
    """
    logger = get_logger()
    out_dir = run_dir / "01_verification" / "A3_zshuffle"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    doc_ids = z_pool.doc_ids[:num_docs]
    n_docs = len(doc_ids)

    logger.info(f"[A3] Starting Z-shuffle Sanity Check: {n_docs} documents")

    # Save variant config
    variant_config = {
        "module": "A3_zshuffle",
        "num_docs": n_docs,
        "max_eval_tokens": max_eval_tokens,
        "seed": seed,
    }
    save_json(variant_config, out_dir / "variant_config.json")

    # Load baseline metrics if not provided
    if baseline_metrics is None:
        baseline_path = run_dir / "01_verification" / "A1_confusion" / "confusion_metrics.json"
        if baseline_path.exists():
            baseline_metrics = load_json(baseline_path)
            logger.info("[A3] Loaded baseline metrics from A1_confusion")
        else:
            logger.error("[A3] Baseline metrics not found. Run A1 first.")
            raise FileNotFoundError(f"Baseline metrics not found at {baseline_path}")

    # Create shuffled z pool
    np.random.seed(seed)
    permutation = np.random.permutation(n_docs)

    # Verify shuffle is non-trivial
    same_position = np.sum(permutation == np.arange(n_docs))
    logger.info(f"[A3] Shuffle: {same_position}/{n_docs} in same position")

    shuffled_z_pool = ShuffledZPool(z_pool, permutation)

    # Run confusion metrics with shuffled z
    with Timer("A3 Z-shuffle evaluation"):
        shuffled_metrics = run_confusion_metrics_only(
            model,
            shuffled_z_pool,
            corpus,
            tokenizer,
            num_docs=num_docs,
            max_eval_tokens=max_eval_tokens,
            device=device,
            use_amp=use_amp,
            doc_ids=doc_ids,
        )

    # Compute comparison
    comparison = {
        # Top-1 accuracy
        "baseline_top1": baseline_metrics.get("top1_acc", 0),
        "shuffled_top1": shuffled_metrics.get("top1_acc", 0),
        "delta_top1": baseline_metrics.get("top1_acc", 0) - shuffled_metrics.get("top1_acc", 0),

        # Top-5 accuracy
        "baseline_top5": baseline_metrics.get("top5_acc", 0),
        "shuffled_top5": shuffled_metrics.get("top5_acc", 0),
        "delta_top5": baseline_metrics.get("top5_acc", 0) - shuffled_metrics.get("top5_acc", 0),

        # Margin
        "baseline_margin": baseline_metrics.get("mean_margin", 0),
        "shuffled_margin": shuffled_metrics.get("mean_margin", 0),
        "delta_margin": baseline_metrics.get("mean_margin", 0) - shuffled_metrics.get("mean_margin", 0),

        # Additional
        "shuffle_seed": seed,
        "same_position_count": int(same_position),
    }

    # Save outputs
    save_json(shuffled_metrics, out_dir / "z_shuffle_metrics.json")
    save_json(comparison, out_dir / "z_shuffle_comparison.json")

    # Save comparison CSV
    csv_path = out_dir / "z_shuffle_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "baseline", "shuffled", "delta"])
        writer.writerow([
            "top1_acc",
            f"{comparison['baseline_top1']:.4f}",
            f"{comparison['shuffled_top1']:.4f}",
            f"{comparison['delta_top1']:.4f}"
        ])
        writer.writerow([
            "top5_acc",
            f"{comparison['baseline_top5']:.4f}",
            f"{comparison['shuffled_top5']:.4f}",
            f"{comparison['delta_top5']:.4f}"
        ])
        writer.writerow([
            "mean_margin",
            f"{comparison['baseline_margin']:.4f}",
            f"{comparison['shuffled_margin']:.4f}",
            f"{comparison['delta_margin']:.4f}"
        ])

    # Summary row
    summary_row = {
        "module": "A3_zshuffle",
        "delta_top1": comparison["delta_top1"],
        "delta_top5": comparison["delta_top5"],
        "delta_margin": comparison["delta_margin"],
    }
    save_json(summary_row, out_dir / "summary_row.json")

    # Generate visualization
    logger.info("[A3] Generating visualization...")
    plot_delta_barplot(
        comparison,
        artifacts_dir,
        title="Z-Shuffle Ablation: Performance Drop After Shuffling"
    )

    # Log results
    logger.info(
        f"[A3] Complete. "
        f"Delta Top-1: {comparison['delta_top1']*100:+.1f}%, "
        f"Delta Margin: {comparison['delta_margin']:+.3f}"
    )

    # Check if shuffle caused expected degradation
    if comparison["delta_top1"] < 0.1:
        logger.warning(
            "[A3] WARNING: Top-1 accuracy did not drop significantly after shuffle. "
            "This may indicate z vectors are not document-specific."
        )

    return comparison
