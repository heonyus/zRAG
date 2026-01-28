"""
A2: Z-only Autoregressive Generation Module

Generates text from z_i alone (NO document tokens) and measures reconstruction quality.
This verifies that z_i encodes enough information to reproduce document content.

EVAL-ONLY: No training is performed.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
from tqdm import tqdm

from .utils import (
    Timer,
    append_jsonl,
    compute_percentiles,
    get_logger,
    load_jsonl,
    safe_mean,
    safe_std,
    save_json,
)
from .visualization import plot_score_distribution
from .reports import generate_eyeball_report


def compute_rouge_l(prediction: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 score using LCS (Longest Common Subsequence).

    Args:
        prediction: Generated text
        reference: Reference text

    Returns:
        ROUGE-L F1 score (0-1)
    """
    if not prediction or not reference:
        return 0.0

    # Tokenize by whitespace (simple approach)
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    # Compute LCS length
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]

    if lcs_length == 0:
        return 0.0

    precision = lcs_length / m
    recall = lcs_length / n
    f1 = 2 * precision * recall / (precision + recall)

    return f1


def generate_from_z_only(
    model,
    z: torch.Tensor,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """
    Generate text from z vector only (no document tokens).

    Args:
        model: WritePhaseModel instance
        z: Z vector [m_tokens, z_dim] or [1, m_tokens, z_dim]
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        do_sample: Whether to sample (True) or greedy (False)

    Returns:
        Generated text string
    """
    model.eval()

    # Ensure z has batch dimension
    if z.dim() == 2:
        z = z.unsqueeze(0)

    with torch.no_grad():
        # Use model's generate_from_z method if available
        if hasattr(model, "generate_from_z"):
            generated = model.generate_from_z(
                z,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
            return generated
        else:
            # Manual generation fallback
            # Project z to embedding space
            alpha_clamped = torch.clamp(model.alpha, min=0.5)
            z_embed = alpha_clamped * model.z_to_embedding(z.float())

            # Convert to model dtype
            if hasattr(model, "llm"):
                dtype = next(model.llm.parameters()).dtype
                z_embed = z_embed.to(dtype)

            # Create attention mask for z tokens
            batch_size = z_embed.size(0)
            m_tokens = z_embed.size(1)
            attention_mask = torch.ones(batch_size, m_tokens, device=z_embed.device)

            # Generate
            outputs = model.llm.generate(
                inputs_embeds=z_embed,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=model.tokenizer.eos_token_id,
            )

            # Decode
            generated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text


def run_zonly_generation(
    model,
    z_pool,
    corpus: Dict[str, str],
    run_dir: Path,
    num_docs: int = 200,
    max_new_tokens: int = 512,
    seed: int = 42,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run A2: Z-only Autoregressive Generation.

    Args:
        model: WritePhaseModel instance
        z_pool: ZPoolManager instance
        corpus: Dict mapping doc_id -> document text
        run_dir: Root run directory
        num_docs: Number of documents to evaluate
        max_new_tokens: Maximum tokens to generate
        seed: Random seed for generation
        device: Device to use

    Returns:
        Dictionary with generation metrics
    """
    logger = get_logger()
    out_dir = run_dir / "01_verification" / "A2_zonly"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    samples_file = out_dir / "z_only_samples.jsonl"

    # Get document IDs
    doc_ids = z_pool.doc_ids[:num_docs]
    n_docs = len(doc_ids)

    logger.info(f"[A2] Starting Z-only Generation: {n_docs} documents, max_new_tokens={max_new_tokens}")

    # Save variant config
    variant_config = {
        "module": "A2_zonly",
        "num_docs": n_docs,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
    }
    save_json(variant_config, out_dir / "variant_config.json")

    # Load cached samples
    cached_samples = load_jsonl(samples_file)
    completed_ids: Set[str] = {s["doc_id"] for s in cached_samples if "doc_id" in s}
    logger.info(f"[A2] Loaded {len(completed_ids)} cached samples")

    # Generate for remaining documents
    results = list(cached_samples)

    with Timer("A2 Z-only generation"):
        model.eval()
        for doc_id in tqdm(doc_ids, desc="Z-only gen"):
            if doc_id in completed_ids:
                continue

            # Get z vector
            z_i = z_pool.get_z(doc_id).to(device)

            # Get reference text
            ref_text = corpus.get(doc_id, "")

            # Generate from z only
            try:
                gen_text = generate_from_z_only(
                    model,
                    z_i,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )
            except Exception as e:
                logger.warning(f"[A2] Generation failed for {doc_id}: {e}")
                gen_text = ""

            # Compute score
            score = compute_rouge_l(gen_text, ref_text)

            # Create record
            record = {
                "doc_id": doc_id,
                "seed": seed,
                "ref_snippet": ref_text[:500],
                "gen_snippet": gen_text[:500],
                "score": score,
                "ref_len": len(ref_text),
                "gen_len": len(gen_text),
                "len_ratio": len(gen_text) / max(len(ref_text), 1),
            }

            # Append to file (crash-safe)
            append_jsonl(samples_file, record)
            results.append(record)

            # Log periodically
            if len(results) % 20 == 0:
                logger.debug(f"[A2] Generated {len(results)}/{n_docs}, last score: {score:.4f}")

    # Compute aggregate metrics
    scores = [r["score"] for r in results if "score" in r]
    len_ratios = [r["len_ratio"] for r in results if "len_ratio" in r]

    metrics = {
        "num_samples": len(results),
        "mean_rouge_l": safe_mean(scores),
        "std_rouge_l": safe_std(scores),
        "min_rouge_l": min(scores) if scores else 0,
        "max_rouge_l": max(scores) if scores else 0,
        **compute_percentiles(scores, [10, 25, 50, 75, 90]),
        "mean_len_ratio": safe_mean(len_ratios),
    }

    # Save metrics
    save_json(metrics, out_dir / "z_only_metrics.json")

    # Summary row
    summary_row = {
        "module": "A2_zonly",
        "mean_rouge_l": metrics["mean_rouge_l"],
        "std_rouge_l": metrics["std_rouge_l"],
        "p50": metrics.get("p50", 0),
        "p90": metrics.get("p90", 0),
    }
    save_json(summary_row, out_dir / "summary_row.json")

    # Generate visualizations
    logger.info("[A2] Generating visualizations...")
    plot_score_distribution(scores, artifacts_dir, title="Z-only Generation ROUGE-L Distribution")

    # Generate eyeball reports
    logger.info("[A2] Generating eyeball reports...")

    # Sort by score
    sorted_results = sorted(results, key=lambda x: x.get("score", 0))

    # Worst 20
    worst_20 = sorted_results[:20]
    generate_eyeball_report(
        worst_20,
        samples_dir / "eyeball_20_worst.md",
        title="Z-only Generation: 20 Worst Cases",
        description="Documents where z-only generation has lowest ROUGE-L with reference.",
    )

    # Best 20
    best_20 = sorted_results[-20:][::-1]  # Reverse to show best first
    generate_eyeball_report(
        best_20,
        samples_dir / "eyeball_20_best.md",
        title="Z-only Generation: 20 Best Cases",
        description="Documents where z-only generation has highest ROUGE-L with reference.",
    )

    logger.info(f"[A2] Complete. Mean ROUGE-L: {metrics['mean_rouge_l']:.4f}")

    return metrics
