"""
Phase 1.5: Phase 1 Regression Testing

Verifies that Phase 1.5 training doesn't degrade Phase 1 capabilities.

Tests:
1. A1 Confusion Matrix: z -> doc matching accuracy
2. A3 Z-shuffle: Verify shuffled z degrades performance

Pass/Fail criteria:
- Top-1 accuracy should not drop more than 2% (configurable threshold)
- Z-shuffle should still show significant degradation

Output structure:
    04_regression_phase1/
    ├── baseline_metrics.json       # Phase 1 baseline (before Phase 1.5)
    ├── post_phase15_metrics.json   # After Phase 1.5 training
    ├── delta.json                  # Difference with pass/fail
    ├── phase1_regression_metrics.json  # Combined summary
    └── artifacts/
        └── delta_barplot.png
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.phase1_modules.utils import (
    Timer,
    get_logger,
    save_json,
    set_seed,
)
from experiments.phase1_modules.confusion_matrix import run_confusion_metrics_only


class ShuffledZPool:
    """
    Wrapper around ZPoolManager that returns shuffled z vectors.

    Used for A3 z-shuffle sanity check.
    """

    def __init__(self, original_z_pool, permutation: np.ndarray):
        """
        Args:
            original_z_pool: Original ZPoolManager
            permutation: Permutation array (indices)
        """
        self.original = original_z_pool
        self.permutation = permutation

        # Build mapping: original_idx -> shuffled_idx
        self.doc_ids = original_z_pool.doc_ids
        self._shuffled_doc_ids = [self.doc_ids[i] for i in permutation]

    def get_z(self, doc_id: str) -> torch.Tensor:
        """Get z for doc_id, but return the shuffled version."""
        # Find original index
        try:
            original_idx = self.doc_ids.index(doc_id)
        except ValueError:
            raise KeyError(f"doc_id {doc_id} not found")

        # Get shuffled doc_id
        shuffled_doc_id = self._shuffled_doc_ids[original_idx]

        # Return z from shuffled doc
        return self.original.get_z(shuffled_doc_id)


def load_baseline_metrics(
    phase1_baseline_run_dir: Optional[Path],
    logger,
) -> Optional[Dict]:
    """
    Load baseline metrics from Phase 1 analysis run.

    Args:
        phase1_baseline_run_dir: Path to Phase 1 analysis run directory
        logger: Logger instance

    Returns:
        Baseline metrics dict or None
    """
    if not phase1_baseline_run_dir:
        return None

    baseline_path = Path(phase1_baseline_run_dir) / "01_verification" / "A1_confusion" / "confusion_metrics.json"

    if not baseline_path.exists():
        logger.warning(f"Baseline metrics not found: {baseline_path}")
        return None

    with open(baseline_path, "r") as f:
        metrics = json.load(f)

    logger.info(f"Loaded baseline metrics from {baseline_path}")
    logger.info(f"  Baseline Top-1: {metrics.get('top1_acc', 0)*100:.1f}%")
    logger.info(f"  Baseline Margin: {metrics.get('mean_margin', 0):.3f}")

    return metrics


def run_phase1_regression(
    model,
    z_pool,
    corpus: Dict[str, str],
    tokenizer,
    run_dir: Path,
    phase1_baseline_run_dir: Optional[Path] = None,
    num_docs: int = 50,
    max_eval_tokens: int = 256,
    device: str = "cuda",
    use_amp: bool = True,
    seed: int = 42,
    regression_threshold: float = 0.02,
) -> Dict:
    """
    Run Phase 1 regression tests on Phase 1.5 model.

    Tests:
        1. A1 Confusion Matrix (subset)
        2. A3 Z-shuffle Sanity (subset)

    Args:
        model: Phase 1.5 model (with LoRA)
        z_pool: ZPoolManager
        corpus: Document corpus
        tokenizer: Tokenizer
        run_dir: Output directory
        phase1_baseline_run_dir: Path to Phase 1 baseline run (optional)
        num_docs: Number of docs to test (50 for speed)
        max_eval_tokens: Max tokens for NLL evaluation
        device: Device string
        use_amp: Use automatic mixed precision
        seed: Random seed
        regression_threshold: Max allowed drop (0.02 = 2%)

    Returns:
        Regression results dict with pass/fail status
    """
    logger = get_logger()
    set_seed(seed)

    out_dir = Path(run_dir) / "04_regression_phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PHASE 1 REGRESSION TESTS")
    logger.info("=" * 60)
    logger.info(f"  Num docs: {num_docs}")
    logger.info(f"  Threshold: {regression_threshold*100:.1f}%")

    results = {
        "num_docs": num_docs,
        "regression_threshold": regression_threshold,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
    }

    # Get subset of doc_ids
    doc_ids_subset = z_pool.doc_ids[:num_docs]

    # =========================================================================
    # 1. Load or compute baseline metrics
    # =========================================================================
    baseline_metrics = load_baseline_metrics(phase1_baseline_run_dir, logger)

    if baseline_metrics:
        results["baseline_source"] = str(phase1_baseline_run_dir)
    else:
        # Compute baseline NOW (this should be the same as Phase 1 results)
        logger.info("Computing baseline metrics (no baseline run provided)...")
        logger.warning("Note: For accurate comparison, provide --phase1_baseline_run_dir")

        with Timer("Baseline A1 Confusion"):
            baseline_metrics = run_confusion_metrics_only(
                model, z_pool, corpus, tokenizer,
                num_docs=num_docs,
                max_eval_tokens=max_eval_tokens,
                device=device,
                use_amp=use_amp,
                doc_ids=doc_ids_subset,
            )
        results["baseline_source"] = "computed_at_runtime"

    # Save baseline metrics
    save_json(baseline_metrics, out_dir / "baseline_metrics.json")

    results["baseline"] = {
        "top1_acc": baseline_metrics.get("top1_acc", 0),
        "top5_acc": baseline_metrics.get("top5_acc", 0),
        "mean_margin": baseline_metrics.get("mean_margin", 0),
    }

    # =========================================================================
    # 2. Run A1 Confusion on current model (post Phase 1.5)
    # =========================================================================
    logger.info("\n--- A1: Confusion Matrix (Post Phase 1.5) ---")

    with Timer("Post-Phase1.5 A1 Confusion"):
        post_metrics = run_confusion_metrics_only(
            model, z_pool, corpus, tokenizer,
            num_docs=num_docs,
            max_eval_tokens=max_eval_tokens,
            device=device,
            use_amp=use_amp,
            doc_ids=doc_ids_subset,
        )

    # Save post metrics
    save_json(post_metrics, out_dir / "post_phase15_metrics.json")

    results["post_phase15"] = {
        "top1_acc": post_metrics.get("top1_acc", 0),
        "top5_acc": post_metrics.get("top5_acc", 0),
        "mean_margin": post_metrics.get("mean_margin", 0),
    }

    # Compute delta
    top1_drop = results["baseline"]["top1_acc"] - results["post_phase15"]["top1_acc"]
    top5_drop = results["baseline"]["top5_acc"] - results["post_phase15"]["top5_acc"]
    margin_drop = results["baseline"]["mean_margin"] - results["post_phase15"]["mean_margin"]

    results["a1_delta"] = {
        "top1_drop": top1_drop,
        "top5_drop": top5_drop,
        "margin_drop": margin_drop,
    }
    results["a1_pass"] = top1_drop <= regression_threshold

    logger.info(f"  Baseline Top-1: {results['baseline']['top1_acc']*100:.1f}%")
    logger.info(f"  Post-1.5 Top-1: {results['post_phase15']['top1_acc']*100:.1f}%")
    logger.info(f"  Delta: {top1_drop*100:+.1f}%")
    logger.info(f"  A1 Pass: {results['a1_pass']}")

    if not results["a1_pass"]:
        logger.warning("!" * 50)
        logger.warning(f"A1 REGRESSION FAILED: drop={top1_drop*100:.1f}% > {regression_threshold*100}%")
        logger.warning("!" * 50)

    # =========================================================================
    # 3. Run A3 Z-shuffle Sanity
    # =========================================================================
    logger.info("\n--- A3: Z-shuffle Sanity ---")

    # Create shuffled z_pool
    np.random.seed(seed)
    permutation = np.random.permutation(num_docs)
    shuffled_z_pool = ShuffledZPool(z_pool, permutation)

    with Timer("A3 Z-shuffle"):
        shuffled_metrics = run_confusion_metrics_only(
            model, shuffled_z_pool, corpus, tokenizer,
            num_docs=num_docs,
            max_eval_tokens=max_eval_tokens,
            device=device,
            use_amp=use_amp,
            doc_ids=doc_ids_subset,
        )

    results["shuffled"] = {
        "top1_acc": shuffled_metrics.get("top1_acc", 0),
        "mean_margin": shuffled_metrics.get("mean_margin", 0),
    }

    # Shuffled should be much worse than post
    shuffle_delta = results["post_phase15"]["top1_acc"] - results["shuffled"]["top1_acc"]
    results["a3_delta"] = shuffle_delta

    # Pass if shuffling causes significant drop (> 10%)
    results["a3_pass"] = shuffle_delta > 0.1

    logger.info(f"  Post-1.5 Top-1: {results['post_phase15']['top1_acc']*100:.1f}%")
    logger.info(f"  Shuffled Top-1: {results['shuffled']['top1_acc']*100:.1f}%")
    logger.info(f"  Delta: {shuffle_delta*100:+.1f}%")
    logger.info(f"  A3 Pass: {results['a3_pass']}")

    if not results["a3_pass"]:
        logger.warning("A3 WARNING: Shuffling did not cause expected degradation")

    # =========================================================================
    # 4. Overall pass/fail
    # =========================================================================
    results["all_pass"] = results["a1_pass"] and results["a3_pass"]

    # Save delta summary
    delta_summary = {
        "a1_top1_drop": top1_drop,
        "a1_pass": results["a1_pass"],
        "a3_shuffle_delta": shuffle_delta,
        "a3_pass": results["a3_pass"],
        "all_pass": results["all_pass"],
        "threshold": regression_threshold,
    }
    save_json(delta_summary, out_dir / "delta.json")

    # Save full results
    save_json(results, out_dir / "phase1_regression_metrics.json")

    # =========================================================================
    # 5. Generate plots
    # =========================================================================
    _plot_regression_comparison(results, artifacts_dir, logger)

    # =========================================================================
    # 6. Print summary
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("REGRESSION TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  A1 (Confusion): {'PASS' if results['a1_pass'] else 'FAIL'}")
    logger.info(f"    Top-1 drop: {top1_drop*100:+.1f}% (threshold: {regression_threshold*100}%)")
    logger.info(f"  A3 (Z-shuffle): {'PASS' if results['a3_pass'] else 'WARN'}")
    logger.info(f"    Shuffle degradation: {shuffle_delta*100:.1f}%")
    logger.info(f"  Overall: {'PASS' if results['all_pass'] else 'FAIL'}")
    logger.info("=" * 60)

    return results


def _plot_regression_comparison(results: Dict, artifacts_dir: Path, logger):
    """Generate comparison bar plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Top-1 Accuracy comparison
        ax = axes[0]
        labels = ["Baseline", "Post-1.5", "Shuffled"]
        values = [
            results["baseline"]["top1_acc"] * 100,
            results["post_phase15"]["top1_acc"] * 100,
            results["shuffled"]["top1_acc"] * 100,
        ]
        colors = ["#2ecc71", "#3498db", "#e74c3c"]

        bars = ax.bar(labels, values, color=colors, edgecolor="black")
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_title("Phase 1 Regression: Top-1 Accuracy")
        ax.set_ylim(0, 105)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

        # Add threshold line
        threshold_val = (results["baseline"]["top1_acc"] - results["regression_threshold"]) * 100
        ax.axhline(threshold_val, color="orange", linestyle="--",
                   label=f"Threshold ({results['regression_threshold']*100:.0f}% drop)")
        ax.legend()

        # Plot 2: Delta comparison
        ax = axes[1]
        labels = ["A1 Drop\n(vs Baseline)", "A3 Drop\n(vs Shuffled)"]
        values = [
            results["a1_delta"]["top1_drop"] * 100,
            results["a3_delta"] * 100,
        ]
        colors = ["#e74c3c" if values[0] > results["regression_threshold"]*100 else "#2ecc71",
                  "#2ecc71" if values[1] > 10 else "#e74c3c"]

        bars = ax.bar(labels, values, color=colors, edgecolor="black")
        ax.set_ylabel("Accuracy Change (%)")
        ax.set_title("Phase 1 Regression: Deltas")
        ax.axhline(0, color="black", linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, values):
            y_pos = bar.get_height() + 0.5 if val >= 0 else bar.get_height() - 2
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f"{val:+.1f}%", ha="center", va="bottom" if val >= 0 else "top", fontsize=10)

        plt.tight_layout()
        plt.savefig(artifacts_dir / "delta_barplot.png", dpi=150)
        plt.savefig(artifacts_dir / "delta_barplot.pdf")
        plt.close()

        logger.info(f"Saved regression plot to {artifacts_dir / 'delta_barplot.png'}")

    except Exception as e:
        logger.warning(f"Failed to generate regression plot: {e}")
