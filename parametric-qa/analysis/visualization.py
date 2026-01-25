"""
시각화 모듈
- Ablation plots
- t-SNE/UMAP z_i 시각화
- Training curves
- Efficiency comparison charts
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import List, Optional

matplotlib.use('Agg')  # Non-interactive backend


def plot_ablation(
    ablation_name: str,
    x_values: list,
    metrics: dict,
    x_label: str = "Parameter Value",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
):
    """
    Ablation study 결과 시각화

    Args:
        ablation_name: 제목
        x_values: x축 값 (e.g., [64, 128, 256, 512])
        metrics: {metric_name: [values], ...}
        x_label: x축 레이블
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(figsize[0] * num_metrics / 3, figsize[1]))

    if num_metrics == 1:
        axes = [axes]

    for ax, (metric_name, values) in zip(axes, metrics.items()):
        ax.plot(x_values, values, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f"{metric_name} vs {x_label}", fontsize=13)
        ax.grid(True, alpha=0.3)

        # Annotate best value
        best_idx = np.argmax(values) if "loss" not in metric_name.lower() and "ppl" not in metric_name.lower() \
            else np.argmin(values)
        ax.annotate(
            f"Best: {values[best_idx]:.3f}",
            xy=(x_values[best_idx], values[best_idx]),
            fontsize=10, color='red', fontweight='bold',
            xytext=(10, 10), textcoords='offset points',
        )

    fig.suptitle(f"Ablation: {ablation_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_tsne(
    z_vectors: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Document Vectors (t-SNE)",
    save_path: Optional[str] = None,
    perplexity: int = 30,
    figsize: tuple = (10, 8),
):
    """
    z_i 벡터 t-SNE 시각화

    Args:
        z_vectors: [num_docs, z_dim] (mean-pooled)
        labels: [num_docs] (optional, for coloring)
    """
    from sklearn.manifold import TSNE

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    z_2d = tsne.fit_transform(z_vectors)

    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='tab10',
                             alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax, label="Cluster/Category")
    else:
        ax.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.5, s=15, color='steelblue')

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.2)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(
    metrics_history: List[dict],
    metrics_to_plot: List[str] = None,
    title: str = "Training Curves",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
):
    """
    학습 과정 시각화

    Args:
        metrics_history: [{loss: x, perplexity: y, ...}, ...]
        metrics_to_plot: 표시할 metric 이름 목록
    """
    if not metrics_history:
        return

    if metrics_to_plot is None:
        metrics_to_plot = [k for k in metrics_history[0].keys()
                           if k not in ("epoch", "global_step")]

    num_plots = len(metrics_to_plot)
    fig, axes = plt.subplots(1, num_plots, figsize=(figsize[0], figsize[1]))

    if num_plots == 1:
        axes = [axes]

    epochs = range(1, len(metrics_history) + 1)

    for ax, metric in zip(axes, metrics_to_plot):
        values = [m.get(metric, None) for m in metrics_history]
        values = [v for v in values if v is not None]

        if values:
            ax.plot(epochs[:len(values)], values, linewidth=2)
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel(metric, fontsize=11)
            ax.set_title(metric.replace("_", " ").title(), fontsize=12)
            ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_efficiency_comparison(
    results: dict,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 5),
):
    """
    Efficiency 비교 차트 (Latency vs EM, Storage comparison)

    Args:
        results: {method: {em, latency_ms, storage_mb_per_1k, ...}, ...}
    """
    methods = list(results.keys())
    ems = [results[m].get("em", results[m].get("em_mean", 0)) for m in methods]
    latencies = [results[m].get("latency_mean_ms", results[m].get("latency_ms", 0)) for m in methods]
    storages = [results[m].get("storage_mb_per_1k", 0) for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. EM comparison (bar)
    colors = ['steelblue'] * len(methods)
    if "parametric_qa" in methods:
        colors[methods.index("parametric_qa")] = 'coral'
    axes[0].barh(methods, ems, color=colors)
    axes[0].set_xlabel("Exact Match")
    axes[0].set_title("Answer Quality (EM)")

    # 2. Latency comparison (bar)
    axes[1].barh(methods, latencies, color=colors)
    axes[1].set_xlabel("Latency (ms)")
    axes[1].set_title("Inference Latency")

    # 3. EM vs Latency scatter (Pareto front)
    axes[2].scatter(latencies, ems, s=100, c=colors, zorder=5)
    for i, m in enumerate(methods):
        axes[2].annotate(m, (latencies[i], ems[i]), fontsize=8,
                         xytext=(5, 5), textcoords='offset points')
    axes[2].set_xlabel("Latency (ms)")
    axes[2].set_ylabel("Exact Match")
    axes[2].set_title("Accuracy-Latency Trade-off")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_adaptation_curve(
    steps_list: List[int],
    perplexities: List[float],
    qa_metrics: Optional[dict] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 5),
):
    """
    z adaptation steps vs performance 곡선

    Args:
        steps_list: [10, 20, 50, 100, 200]
        perplexities: 각 step 수에서의 PPL
        qa_metrics: {steps: em, ...} (optional)
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    # PPL curve
    color1 = 'steelblue'
    ax1.plot(steps_list, perplexities, 'o-', color=color1, linewidth=2, label='Perplexity')
    ax1.set_xlabel("Adaptation Steps", fontsize=12)
    ax1.set_ylabel("Perplexity", color=color1, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='PPL=10 threshold')

    # QA metric (secondary axis)
    if qa_metrics:
        ax2 = ax1.twinx()
        color2 = 'coral'
        ems = [qa_metrics.get(s, 0) for s in steps_list]
        ax2.plot(steps_list, ems, 's--', color=color2, linewidth=2, label='EM')
        ax2.set_ylabel("Exact Match", color=color2, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_title("z Adaptation: Steps vs Performance", fontsize=14)
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc='upper right')

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
