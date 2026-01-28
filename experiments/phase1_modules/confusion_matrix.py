"""
A1: Full NLL Confusion Matrix Module

Computes NLL[i,j] = NLL(D_i | z_j) for all document-z pairs.
This is the core verification that z_i encodes document-specific information.

EVAL-ONLY: No training is performed.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from .utils import (
    Timer,
    append_jsonl,
    get_logger,
    load_json,
    load_jsonl_cache,
    save_json,
)
from .visualization import plot_confusion_heatmap, plot_margin_histogram
from .reports import generate_worst_cases_report, generate_top_confusions_report


def compute_nll(
    model,
    z: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    use_amp: bool = True,
) -> float:
    """
    Compute NLL of document given z vector.

    Args:
        model: WritePhaseModel instance
        z: Z vector [m_tokens, z_dim] or [1, m_tokens, z_dim]
        input_ids: Document token IDs [1, seq_len]
        attention_mask: Attention mask [1, seq_len]
        use_amp: Whether to use automatic mixed precision

    Returns:
        NLL value (float)
    """
    model.eval()

    # Ensure z has batch dimension
    if z.dim() == 2:
        z = z.unsqueeze(0)

    with torch.no_grad():
        if use_amp and torch.cuda.is_available():
            with autocast(dtype=torch.bfloat16):
                outputs = model(z, input_ids, attention_mask)
        else:
            outputs = model(z, input_ids, attention_mask)

        nll = outputs["loss"].item()

    return nll


def compute_confusion_metrics(nll_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Compute metrics from NLL confusion matrix.

    Args:
        nll_matrix: 2D array [num_docs, num_docs] where [i,j] = NLL(doc_i | z_j)

    Returns:
        Dictionary with:
        - top1_acc: Fraction where argmin(row) == diagonal index
        - top5_acc: Fraction where diagonal index is in top-5 lowest NLL
        - mean_margin: Average (best_wrong_NLL - correct_NLL)
        - median_margin, p10_margin, p90_margin
        - worst_docs: List of worst-margin documents
    """
    n_docs = nll_matrix.shape[0]

    # Top-1 accuracy: argmin of each row should be the diagonal
    predictions = np.argmin(nll_matrix, axis=1)
    top1_correct = predictions == np.arange(n_docs)
    top1_acc = np.mean(top1_correct)

    # Top-5 accuracy
    top5_indices = np.argsort(nll_matrix, axis=1)[:, :5]
    top5_correct = [i in top5_indices[i] for i in range(n_docs)]
    top5_acc = np.mean(top5_correct)

    # Margins: best_wrong_NLL - correct_NLL
    margins = []
    worst_docs = []
    for i in range(n_docs):
        correct_nll = nll_matrix[i, i]
        row = nll_matrix[i].copy()
        row[i] = np.inf  # Exclude diagonal
        best_wrong_idx = np.argmin(row)
        best_wrong_nll = row[best_wrong_idx]
        margin = best_wrong_nll - correct_nll
        margins.append(margin)

        worst_docs.append({
            "doc_idx": int(i),
            "doc_id": f"doc_{i}",
            "correct_nll": float(correct_nll),
            "best_wrong_idx": int(best_wrong_idx),
            "best_wrong_nll": float(best_wrong_nll),
            "margin": float(margin),
        })

    margins = np.array(margins)

    # Sort by margin to get worst cases
    worst_docs.sort(key=lambda x: x["margin"])

    # Compute stats
    metrics = {
        "num_docs": n_docs,
        "top1_acc": float(top1_acc),
        "top5_acc": float(top5_acc),
        "mean_margin": float(np.mean(margins)),
        "median_margin": float(np.median(margins)),
        "std_margin": float(np.std(margins)),
        "min_margin": float(np.min(margins)),
        "max_margin": float(np.max(margins)),
        "p10_margin": float(np.percentile(margins, 10)),
        "p90_margin": float(np.percentile(margins, 90)),
        "positive_margin_pct": float(np.mean(margins > 0) * 100),
        "worst_docs": worst_docs[:20],  # Top 20 worst
    }

    return metrics


def run_confusion_matrix(
    model,
    z_pool,
    corpus: Dict[str, str],
    tokenizer,
    run_dir: Path,
    num_docs: int = 200,
    max_eval_tokens: int = 256,
    device: str = "cuda",
    use_amp: bool = True,
) -> Dict[str, Any]:
    """
    Run A1: Full NLL Confusion Matrix computation.

    Args:
        model: WritePhaseModel instance
        z_pool: ZPoolManager instance
        corpus: Dict mapping doc_id -> document text
        tokenizer: Tokenizer for encoding documents
        run_dir: Root run directory
        num_docs: Number of documents to evaluate
        max_eval_tokens: Maximum tokens per document
        device: Device to use
        use_amp: Whether to use AMP

    Returns:
        Dictionary with confusion metrics
    """
    logger = get_logger()
    out_dir = run_dir / "01_verification" / "A1_confusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / "nll_partial.jsonl"

    # Get document IDs
    doc_ids = z_pool.doc_ids[:num_docs]
    n_docs = len(doc_ids)

    logger.info(f"[A1] Starting NLL Confusion Matrix: {n_docs}x{n_docs}")

    # Save variant config
    variant_config = {
        "module": "A1_confusion",
        "num_docs": n_docs,
        "max_eval_tokens": max_eval_tokens,
        "use_amp": use_amp,
    }
    save_json(variant_config, out_dir / "variant_config.json")

    # Tokenize all documents
    logger.info("[A1] Tokenizing documents...")
    tokenized_docs = {}
    for doc_id in tqdm(doc_ids, desc="Tokenizing"):
        doc_text = corpus.get(doc_id, "")
        encoded = tokenizer(
            doc_text,
            max_length=max_eval_tokens,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        tokenized_docs[doc_id] = {
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
        }

    # Load cached progress
    cached_rows = load_jsonl_cache(cache_file, key_field="i")
    logger.info(f"[A1] Loaded {len(cached_rows)} cached rows")

    # Initialize NLL matrix
    nll_matrix = np.zeros((n_docs, n_docs), dtype=np.float32)

    # Fill from cache
    for i, record in cached_rows.items():
        if i < n_docs and "nll_row" in record:
            nll_matrix[i] = np.array(record["nll_row"])[:n_docs]

    # Compute remaining rows
    with Timer("A1 NLL computation"):
        model.eval()
        for i in tqdm(range(n_docs), desc="NLL rows"):
            if i in cached_rows:
                continue

            doc_id_i = doc_ids[i]
            doc_data = tokenized_docs[doc_id_i]

            # Compute NLL for all z vectors
            for j in range(n_docs):
                doc_id_j = doc_ids[j]
                z_j = z_pool.get_z(doc_id_j).to(device)

                nll = compute_nll(
                    model,
                    z_j,
                    doc_data["input_ids"],
                    doc_data["attention_mask"],
                    use_amp=use_amp,
                )
                nll_matrix[i, j] = nll

            # Checkpoint this row
            append_jsonl(cache_file, {
                "i": i,
                "doc_id": doc_id_i,
                "nll_row": nll_matrix[i].tolist(),
            })

            # Log progress periodically
            if (i + 1) % 20 == 0:
                logger.debug(f"[A1] Processed {i + 1}/{n_docs} rows")

    # Compute metrics
    logger.info("[A1] Computing metrics...")
    metrics = compute_confusion_metrics(nll_matrix)

    # Save outputs
    logger.info("[A1] Saving outputs...")

    # NLL matrix
    np.save(out_dir / "nll_matrix.npy", nll_matrix)

    # Confusion table CSV
    csv_path = out_dir / "confusion_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_i", "pred_z", "correct_nll", "best_wrong_nll", "margin", "top5_hit"])

        for i in range(n_docs):
            pred_z = int(np.argmin(nll_matrix[i]))
            correct_nll = nll_matrix[i, i]
            row = nll_matrix[i].copy()
            row[i] = np.inf
            best_wrong_nll = np.min(row)
            margin = best_wrong_nll - correct_nll
            top5 = np.argsort(nll_matrix[i])[:5]
            top5_hit = 1 if i in top5 else 0

            writer.writerow([
                doc_ids[i], doc_ids[pred_z],
                f"{correct_nll:.4f}", f"{best_wrong_nll:.4f}",
                f"{margin:.4f}", top5_hit
            ])

    # Metrics JSON
    save_json(metrics, out_dir / "confusion_metrics.json")

    # Summary row
    summary_row = {
        "module": "A1_confusion",
        "top1_acc": metrics["top1_acc"],
        "top5_acc": metrics["top5_acc"],
        "mean_margin": metrics["mean_margin"],
        "median_margin": metrics["median_margin"],
        "min_margin": metrics["min_margin"],
    }
    save_json(summary_row, out_dir / "summary_row.json")

    # Generate visualizations
    logger.info("[A1] Generating visualizations...")
    plot_confusion_heatmap(nll_matrix, artifacts_dir, title=f"NLL Confusion Matrix ({n_docs} docs)")
    plot_margin_histogram(nll_matrix, artifacts_dir)

    # Generate reports
    logger.info("[A1] Generating reports...")
    generate_worst_cases_report(nll_matrix, corpus, samples_dir, n_worst=20, doc_ids=doc_ids)
    generate_top_confusions_report(nll_matrix, corpus, artifacts_dir, n_top=20, doc_ids=doc_ids)

    logger.info(f"[A1] Complete. Top-1 Acc: {metrics['top1_acc']*100:.1f}%, Mean Margin: {metrics['mean_margin']:.3f}")

    return metrics


def run_confusion_metrics_only(
    model,
    z_pool,
    corpus: Dict[str, str],
    tokenizer,
    num_docs: int = 200,
    max_eval_tokens: int = 256,
    device: str = "cuda",
    use_amp: bool = True,
    doc_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run confusion matrix metrics without saving full artifacts.
    Used for ablation studies.

    Args:
        model: WritePhaseModel instance
        z_pool: ZPoolManager or shuffled z_pool
        corpus: Dict mapping doc_id -> document text
        tokenizer: Tokenizer
        num_docs: Number of documents
        max_eval_tokens: Maximum tokens
        device: Device
        use_amp: Use AMP
        doc_ids: Optional list of doc IDs (default: from z_pool)

    Returns:
        Metrics dictionary
    """
    logger = get_logger()

    if doc_ids is None:
        doc_ids = z_pool.doc_ids[:num_docs]
    n_docs = len(doc_ids)

    # Tokenize documents
    tokenized_docs = {}
    for doc_id in doc_ids:
        doc_text = corpus.get(doc_id, "")
        encoded = tokenizer(
            doc_text,
            max_length=max_eval_tokens,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        tokenized_docs[doc_id] = {
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
        }

    # Compute NLL matrix
    nll_matrix = np.zeros((n_docs, n_docs), dtype=np.float32)

    model.eval()
    for i in tqdm(range(n_docs), desc="NLL (ablation)", leave=False):
        doc_id_i = doc_ids[i]
        doc_data = tokenized_docs[doc_id_i]

        for j in range(n_docs):
            doc_id_j = doc_ids[j]
            z_j = z_pool.get_z(doc_id_j).to(device)

            nll = compute_nll(
                model,
                z_j,
                doc_data["input_ids"],
                doc_data["attention_mask"],
                use_amp=use_amp,
            )
            nll_matrix[i, j] = nll

    # Compute and return metrics
    metrics = compute_confusion_metrics(nll_matrix)
    return metrics
