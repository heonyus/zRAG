"""
Visualization functions for Phase 1 verification and ablation.

All plots are saved in both PNG (quick preview) and PDF (publication quality) formats.
Uses Agg backend for headless server compatibility.
"""

import matplotlib
matplotlib.use("Agg")  # Headless backend - must be before pyplot import

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .utils import get_logger


# Style settings for consistent appearance
STYLE_CONFIG = {
    "figure.figsize": (10, 8),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
}


def _apply_style():
    """Apply consistent style to all plots."""
    plt.rcParams.update(STYLE_CONFIG)


def _save_figure(fig: plt.Figure, path: Path, name: str) -> Tuple[Path, Path]:
    """
    Save figure in both PNG and PDF formats.

    Args:
        fig: Matplotlib figure
        path: Directory to save to
        name: Base filename (without extension)

    Returns:
        Tuple of (png_path, pdf_path)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    png_path = path / f"{name}.png"
    pdf_path = path / f"{name}.pdf"

    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    get_logger().debug(f"Saved figures: {png_path}, {pdf_path}")

    return png_path, pdf_path


def plot_confusion_heatmap(
    nll_matrix: np.ndarray,
    output_dir: Union[str, Path],
    title: str = "NLL Confusion Matrix",
    cmap: str = "viridis_r",  # Reversed so low NLL (good) is dark
) -> Tuple[Path, Path]:
    """
    Plot NLL confusion matrix as heatmap.

    Args:
        nll_matrix: 2D array [num_docs, num_docs] where entry [i,j] = NLL(doc_i | z_j)
        output_dir: Directory to save plots
        title: Plot title
        cmap: Colormap name (default viridis_r, reversed so low=dark)

    Returns:
        Tuple of (png_path, pdf_path)

    Note:
        Diagonal should be dark (low NLL) if z correctly encodes documents.
        Off-diagonal should be bright (high NLL).
    """
    _apply_style()

    n_docs = nll_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap
    im = ax.imshow(nll_matrix, cmap=cmap, aspect="auto")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("NLL (lower = better)", rotation=270, labelpad=20)

    # Labels
    ax.set_xlabel("z index (which z vector)")
    ax.set_ylabel("Document index (which document)")
    ax.set_title(title)

    # Diagonal highlight
    for i in range(n_docs):
        ax.plot(i, i, "rx", markersize=2, alpha=0.3)

    # Add text annotation for matrix size
    ax.text(
        0.02, 0.98, f"Matrix size: {n_docs}x{n_docs}",
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    # Compute and display key metrics
    diag_mean = np.mean(np.diag(nll_matrix))
    offdiag_mask = ~np.eye(n_docs, dtype=bool)
    offdiag_mean = np.mean(nll_matrix[offdiag_mask])
    top1_acc = np.mean(np.argmin(nll_matrix, axis=1) == np.arange(n_docs)) * 100

    stats_text = f"Diag mean: {diag_mean:.3f}\nOff-diag mean: {offdiag_mean:.3f}\nTop-1 acc: {top1_acc:.1f}%"
    ax.text(
        0.98, 0.02, stats_text,
        transform=ax.transAxes, fontsize=10,
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.tight_layout()

    paths = _save_figure(fig, output_dir, "confusion_heatmap")
    plt.close(fig)

    return paths


def plot_margin_histogram(
    nll_matrix: np.ndarray,
    output_dir: Union[str, Path],
    title: str = "Margin Distribution (correct z vs best wrong z)",
) -> Tuple[Path, Path]:
    """
    Plot histogram of margins: NLL(doc|best_wrong_z) - NLL(doc|correct_z).

    Args:
        nll_matrix: 2D array [num_docs, num_docs]
        output_dir: Directory to save plots
        title: Plot title

    Returns:
        Tuple of (png_path, pdf_path)

    Note:
        Positive margin = correct z is better than best alternative.
        All margins should be positive for a well-trained model.
    """
    _apply_style()

    n_docs = nll_matrix.shape[0]

    # Compute margins
    margins = []
    for i in range(n_docs):
        correct_nll = nll_matrix[i, i]
        # Best wrong z (minimum NLL among non-diagonal)
        row = nll_matrix[i].copy()
        row[i] = np.inf  # Exclude diagonal
        best_wrong_nll = np.min(row)
        margin = best_wrong_nll - correct_nll
        margins.append(margin)

    margins = np.array(margins)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    bins = 50
    counts, bin_edges, patches = ax.hist(margins, bins=bins, edgecolor="black", alpha=0.7)

    # Color bars: positive margins green, negative red
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        if left_edge >= 0:
            patch.set_facecolor("green")
        else:
            patch.set_facecolor("red")

    # Add vertical line at 0
    ax.axvline(x=0, color="black", linestyle="--", linewidth=2, label="Zero margin")

    # Add vertical line at mean
    mean_margin = np.mean(margins)
    ax.axvline(x=mean_margin, color="blue", linestyle="-", linewidth=2, label=f"Mean: {mean_margin:.3f}")

    # Labels
    ax.set_xlabel("Margin (best_wrong_NLL - correct_NLL)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()

    # Stats text
    positive_pct = np.mean(margins > 0) * 100
    stats_text = (
        f"Mean: {np.mean(margins):.3f}\n"
        f"Median: {np.median(margins):.3f}\n"
        f"Min: {np.min(margins):.3f}\n"
        f"Max: {np.max(margins):.3f}\n"
        f"Positive: {positive_pct:.1f}%"
    )
    ax.text(
        0.98, 0.98, stats_text,
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.tight_layout()

    paths = _save_figure(fig, output_dir, "margin_hist")
    plt.close(fig)

    return paths


def plot_delta_barplot(
    comparison: Dict[str, float],
    output_dir: Union[str, Path],
    title: str = "Z-Shuffle Ablation: Baseline vs Shuffled",
) -> Tuple[Path, Path]:
    """
    Plot bar chart comparing baseline and shuffled metrics with deltas.

    Args:
        comparison: Dict with keys like baseline_top1, shuffled_top1, delta_top1, etc.
        output_dir: Directory to save plots
        title: Plot title

    Returns:
        Tuple of (png_path, pdf_path)
    """
    _apply_style()

    # Extract metrics
    metrics = ["top1", "top5", "margin"]
    baseline_values = []
    shuffled_values = []
    deltas = []

    for m in metrics:
        baseline_key = f"baseline_{m}"
        shuffled_key = f"shuffled_{m}"
        delta_key = f"delta_{m}"

        if baseline_key in comparison:
            baseline_values.append(comparison[baseline_key])
            shuffled_values.append(comparison.get(shuffled_key, 0))
            deltas.append(comparison.get(delta_key, 0))

    if not baseline_values:
        # Fallback: just plot whatever we have
        metrics = list(comparison.keys())
        baseline_values = [comparison[k] for k in metrics]
        shuffled_values = [0] * len(metrics)
        deltas = baseline_values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Comparison bar chart
    ax1 = axes[0]
    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax1.bar(x - width/2, baseline_values, width, label="Baseline", color="steelblue")
    bars2 = ax1.bar(x + width/2, shuffled_values, width, label="Shuffled", color="coral")

    ax1.set_xlabel("Metric")
    ax1.set_ylabel("Value")
    ax1.set_title("Baseline vs Shuffled Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)

    # Right: Delta bar chart
    ax2 = axes[1]
    colors = ["green" if d > 0 else "red" for d in deltas]
    bars3 = ax2.bar(x, deltas, color=colors, alpha=0.7, edgecolor="black")

    ax2.set_xlabel("Metric")
    ax2.set_ylabel("Delta (Baseline - Shuffled)")
    ax2.set_title("Performance Drop After Shuffling")
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar, delta in zip(bars3, deltas):
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        offset = 3 if height >= 0 else -3
        ax2.annotate(f'{delta:+.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, offset), textcoords="offset points",
                     ha='center', va=va, fontsize=9)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    paths = _save_figure(fig, output_dir, "delta_barplot")
    plt.close(fig)

    return paths


def plot_score_distribution(
    scores: List[float],
    output_dir: Union[str, Path],
    title: str = "Z-only Generation Score Distribution",
    score_name: str = "ROUGE-L",
) -> Tuple[Path, Path]:
    """
    Plot histogram of generation scores (e.g., ROUGE-L).

    Args:
        scores: List of score values
        output_dir: Directory to save plots
        title: Plot title
        score_name: Name of the score metric

    Returns:
        Tuple of (png_path, pdf_path)
    """
    _apply_style()

    scores = np.array(scores)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    ax.hist(scores, bins=40, edgecolor="black", alpha=0.7, color="steelblue")

    # Add percentile lines
    p50 = np.percentile(scores, 50)
    p90 = np.percentile(scores, 90)

    ax.axvline(x=p50, color="orange", linestyle="--", linewidth=2, label=f"Median: {p50:.3f}")
    ax.axvline(x=p90, color="green", linestyle="--", linewidth=2, label=f"P90: {p90:.3f}")

    # Labels
    ax.set_xlabel(score_name)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()

    # Stats text
    stats_text = (
        f"Mean: {np.mean(scores):.3f}\n"
        f"Std: {np.std(scores):.3f}\n"
        f"Min: {np.min(scores):.3f}\n"
        f"Max: {np.max(scores):.3f}\n"
        f"P10: {np.percentile(scores, 10):.3f}\n"
        f"P50: {np.percentile(scores, 50):.3f}\n"
        f"P90: {np.percentile(scores, 90):.3f}"
    )
    ax.text(
        0.98, 0.98, stats_text,
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.tight_layout()

    paths = _save_figure(fig, output_dir, "score_distribution")
    plt.close(fig)

    return paths


def plot_ablation_comparison(
    results: Dict[str, Dict[str, float]],
    output_dir: Union[str, Path],
    metric_key: str = "top1_acc",
    title: str = "Ablation Comparison",
) -> Tuple[Path, Path]:
    """
    Plot bar chart comparing a specific metric across ablation variants.

    Args:
        results: Dict mapping variant_name -> metrics_dict
        output_dir: Directory to save plots
        metric_key: Which metric to compare
        title: Plot title

    Returns:
        Tuple of (png_path, pdf_path)
    """
    _apply_style()

    variants = list(results.keys())
    values = [results[v].get(metric_key, 0) for v in variants]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(variants)))
    bars = ax.bar(variants, values, color=colors, edgecolor="black")

    ax.set_xlabel("Variant")
    ax.set_ylabel(metric_key.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    # Rotate labels if needed
    plt.xticks(rotation=45, ha="right")

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    paths = _save_figure(fig, output_dir, f"ablation_{metric_key}")
    plt.close(fig)

    return paths
