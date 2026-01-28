#!/usr/bin/env python3
"""
Phase 1.5: Evidence Dataset Builder

Builds training dataset for evidence generation from QA pairs.
Extracts evidence spans from gold documents using two methods:
1. answer_span (default): Find answer in document, extract surrounding sentences
2. sentence_ranker (fallback): Rank sentences by keyword overlap

Usage:
    python data/build_phase1_5_evidence_dataset.py \\
        --corpus_dir checkpoints/phase2_corpus \\
        --output_dir results/phase1_5/20260128_123456/01_data \\
        --max_evidence_tokens 256

Author: zRAG Team
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import random

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_corpus_and_qa(
    corpus_dir: str = "checkpoints/phase2_corpus",
    split: str = "val",
) -> Tuple[Dict[str, str], List[Dict]]:
    """
    Load corpus and QA pairs from Phase 2 corpus directory.

    This reuses the same loading logic as Track A evaluation.

    Args:
        corpus_dir: Path to phase2_corpus directory
        split: "val" or "train"

    Returns:
        Tuple of (corpus dict, qa_pairs list)
    """
    corpus_dir = Path(corpus_dir)

    # Load corpus
    corpus_path = corpus_dir / "corpus.json"
    if not corpus_path.exists():
        raise FileNotFoundError(f"corpus.json not found: {corpus_path}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # Load QA pairs
    qa_file = f"qa_{split}.json"
    qa_path = corpus_dir / qa_file
    if not qa_path.exists():
        raise FileNotFoundError(f"{qa_file} not found: {qa_path}")

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    # Handle different formats
    if isinstance(qa_data, dict) and "data" in qa_data:
        qa_list = qa_data["data"]
    elif isinstance(qa_data, list):
        qa_list = qa_data
    else:
        raise ValueError(f"Unknown qa format: {type(qa_data)}")

    # Normalize QA pairs
    qa_pairs = []
    for item in qa_list:
        gold_doc_ids = item.get("gold_doc_ids", [])
        if isinstance(gold_doc_ids, str):
            gold_doc_ids = [gold_doc_ids]

        qa_pairs.append({
            "question": item["question"],
            "answer": item["answer"],
            "gold_doc_ids": gold_doc_ids,
            "gold_titles": item.get("gold_titles", []),
            "evidence": item.get("evidence", ""),
            "type": item.get("type", "unknown"),
            "level": item.get("level", "unknown"),
        })

    return corpus, qa_pairs


def extract_evidence_answer_span(
    document: str,
    answer: str,
    tokenizer,
    max_evidence_tokens: int = 256,
    context_sentences: int = 2,
) -> Tuple[Optional[str], Dict]:
    """
    Token-based evidence extraction with sentence boundary snap.

    1. Split document into sentences
    2. Find sentence containing the answer
    3. Extract that sentence ± context_sentences
    4. Truncate to max_evidence_tokens

    Args:
        document: Full document text
        answer: Target answer string
        tokenizer: Tokenizer for token-based truncation
        max_evidence_tokens: Maximum evidence tokens
        context_sentences: Number of sentences before/after answer sentence

    Returns:
        Tuple of (evidence_text or None, metadata dict)
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', document)

    if not sentences:
        return None, {"method": "answer_span", "status": "no_sentences"}

    # Find sentence containing answer (case-insensitive)
    answer_lower = answer.lower()
    answer_sent_idx = None

    for i, sent in enumerate(sentences):
        if answer_lower in sent.lower():
            answer_sent_idx = i
            break

    if answer_sent_idx is None:
        return None, {"method": "answer_span", "status": "not_found"}

    # Extract window around answer sentence
    start_idx = max(0, answer_sent_idx - context_sentences)
    end_idx = min(len(sentences), answer_sent_idx + context_sentences + 1)

    evidence = " ".join(sentences[start_idx:end_idx])

    # Token-based truncation
    tokens = tokenizer.encode(evidence, add_special_tokens=False)
    original_tokens = len(tokens)

    if len(tokens) > max_evidence_tokens:
        tokens = tokens[:max_evidence_tokens]
        evidence = tokenizer.decode(tokens, skip_special_tokens=True)

    meta = {
        "method": "answer_span",
        "status": "success",
        "answer_sent_idx": answer_sent_idx,
        "window": [start_idx, end_idx],
        "num_sentences": end_idx - start_idx,
        "original_tokens": original_tokens,
        "final_tokens": len(tokens),
        "truncated": original_tokens > max_evidence_tokens,
    }

    return evidence, meta


def extract_evidence_sentence_ranker(
    document: str,
    question: str,
    answer: str,
    tokenizer,
    max_evidence_tokens: int = 256,
    top_k: int = 3,
) -> Tuple[str, Dict]:
    """
    Sentence ranking based evidence extraction.

    1. Split document into sentences
    2. Score each sentence by keyword overlap with question + answer
    3. Boost sentences containing the answer
    4. Select top-k sentences (preserving original order)
    5. Truncate to max_evidence_tokens

    Args:
        document: Full document text
        question: Query string
        answer: Target answer string
        tokenizer: Tokenizer for token-based truncation
        max_evidence_tokens: Maximum evidence tokens
        top_k: Number of top sentences to include

    Returns:
        Tuple of (evidence_text, metadata dict)
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', document)

    if not sentences:
        return document[:500], {"method": "sentence_ranker", "status": "no_sentences"}

    # Build keyword set from question and answer
    question_words = set(question.lower().split())
    answer_words = set(answer.lower().split())
    keywords = question_words | answer_words

    # Remove common stopwords
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                 "being", "have", "has", "had", "do", "does", "did", "will",
                 "would", "could", "should", "may", "might", "must", "shall",
                 "can", "to", "of", "in", "for", "on", "with", "at", "by",
                 "from", "as", "into", "through", "during", "before", "after",
                 "above", "below", "between", "under", "again", "further",
                 "then", "once", "here", "there", "when", "where", "why",
                 "how", "all", "each", "few", "more", "most", "other", "some",
                 "such", "no", "nor", "not", "only", "own", "same", "so",
                 "than", "too", "very", "just", "and", "but", "if", "or",
                 "because", "until", "while", "what", "which", "who", "whom",
                 "this", "that", "these", "those", "am", "it", "its", "i",
                 "you", "he", "she", "they", "we", "my", "your", "his", "her",
                 "their", "our"}
    keywords = keywords - stopwords

    # Score each sentence
    scored = []
    for i, sent in enumerate(sentences):
        sent_words = set(sent.lower().split())
        overlap = len(keywords & sent_words)

        # Strong boost for sentences containing the answer
        if answer.lower() in sent.lower():
            overlap += 10

        scored.append((overlap, i, sent))

    # Sort by score (descending)
    scored.sort(reverse=True, key=lambda x: x[0])

    # Take top-k (but preserve original order in output)
    top_indices = sorted([s[1] for s in scored[:top_k]])
    top_sentences = [sentences[i] for i in top_indices]
    top_scores = {s[1]: s[0] for s in scored[:top_k]}

    evidence = " ".join(top_sentences)

    # Token-based truncation
    tokens = tokenizer.encode(evidence, add_special_tokens=False)
    original_tokens = len(tokens)

    if len(tokens) > max_evidence_tokens:
        tokens = tokens[:max_evidence_tokens]
        evidence = tokenizer.decode(tokens, skip_special_tokens=True)

    meta = {
        "method": "sentence_ranker",
        "status": "success",
        "top_k": top_k,
        "sentence_indices": top_indices,
        "sentence_scores": [top_scores.get(i, 0) for i in top_indices],
        "original_tokens": original_tokens,
        "final_tokens": len(tokens),
        "truncated": original_tokens > max_evidence_tokens,
    }

    return evidence, meta


def build_evidence_dataset(
    corpus_dir: str,
    output_dir: str,
    split: str = "val",
    primary_method: str = "answer_span",
    fallback_method: str = "sentence_ranker",
    max_evidence_tokens: int = 256,
    context_sentences: int = 2,
    top_k_sentences: int = 3,
    tokenizer_name: str = "Qwen/Qwen3-8B",
    seed: int = 42,
    num_preview_samples: int = 20,
) -> Dict:
    """
    Build evidence dataset from QA pairs and corpus.

    Args:
        corpus_dir: Path to phase2_corpus directory
        output_dir: Output directory for dataset files
        split: "val" or "train"
        primary_method: Primary extraction method ("answer_span" or "sentence_ranker")
        fallback_method: Fallback method if primary fails
        max_evidence_tokens: Maximum tokens per evidence
        context_sentences: Sentences around answer for answer_span method
        top_k_sentences: Top sentences for sentence_ranker method
        tokenizer_name: Tokenizer for token counting
        seed: Random seed
        num_preview_samples: Number of samples for preview

    Returns:
        Dataset statistics dict

    Output files:
        - dataset.jsonl: Main dataset
        - dataset_manifest.json: Statistics and metadata
        - samples_preview.md: Sample preview for manual inspection
    """
    from transformers import AutoTokenizer

    random.seed(seed)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    # Load corpus and QA
    print(f"Loading corpus and QA from {corpus_dir}...")
    corpus, qa_pairs = load_corpus_and_qa(corpus_dir, split)
    print(f"  Loaded {len(corpus)} documents, {len(qa_pairs)} QA pairs")

    # Build dataset
    dataset_path = output_dir / "dataset.jsonl"
    samples = []

    stats = {
        "total": len(qa_pairs),
        "success": 0,
        "primary_success": 0,
        "fallback_used": 0,
        "no_gold_doc": 0,
        "extraction_failed": 0,
        "total_evidence_tokens": 0,
        "evidence_lengths": [],
    }

    print("Building evidence dataset...")
    with open(dataset_path, "w", encoding="utf-8") as f:
        for idx, qa in enumerate(qa_pairs):
            sample = {
                "sample_id": idx,
                "question": qa["question"],
                "answer": qa["answer"],
                "gold_doc_ids": qa["gold_doc_ids"],
                "qa_type": qa.get("type", "unknown"),
                "qa_level": qa.get("level", "unknown"),
            }

            # Get gold document
            gold_doc_ids = qa["gold_doc_ids"]
            if not gold_doc_ids:
                stats["no_gold_doc"] += 1
                sample["evidence_text"] = ""
                sample["evidence_method"] = "none"
                sample["evidence_meta"] = {"status": "no_gold_doc"}
                samples.append(sample)
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                continue

            # Use first gold doc
            doc_id = gold_doc_ids[0]
            document = corpus.get(doc_id, "")

            if not document:
                stats["no_gold_doc"] += 1
                sample["evidence_text"] = ""
                sample["evidence_method"] = "none"
                sample["evidence_meta"] = {"status": "doc_not_found", "doc_id": doc_id}
                samples.append(sample)
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                continue

            sample["doc_id"] = doc_id

            # Try primary method
            if primary_method == "answer_span":
                evidence, meta = extract_evidence_answer_span(
                    document, qa["answer"], tokenizer,
                    max_evidence_tokens, context_sentences
                )
            else:
                evidence, meta = extract_evidence_sentence_ranker(
                    document, qa["question"], qa["answer"], tokenizer,
                    max_evidence_tokens, top_k_sentences
                )

            # Fallback if primary failed
            if evidence is None:
                stats["fallback_used"] += 1
                if fallback_method == "sentence_ranker":
                    evidence, meta = extract_evidence_sentence_ranker(
                        document, qa["question"], qa["answer"], tokenizer,
                        max_evidence_tokens, top_k_sentences
                    )
                else:
                    evidence, meta = extract_evidence_answer_span(
                        document, qa["answer"], tokenizer,
                        max_evidence_tokens, context_sentences
                    )

            if evidence:
                stats["success"] += 1
                if meta["method"] == primary_method:
                    stats["primary_success"] += 1

                evidence_tokens = len(tokenizer.encode(evidence, add_special_tokens=False))
                stats["total_evidence_tokens"] += evidence_tokens
                stats["evidence_lengths"].append(evidence_tokens)
            else:
                stats["extraction_failed"] += 1
                evidence = ""

            sample["evidence_text"] = evidence
            sample["evidence_method"] = meta.get("method", "unknown")
            sample["evidence_meta"] = meta

            samples.append(sample)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Compute final stats
    if stats["evidence_lengths"]:
        import numpy as np
        lengths = np.array(stats["evidence_lengths"])
        stats["avg_evidence_tokens"] = float(np.mean(lengths))
        stats["std_evidence_tokens"] = float(np.std(lengths))
        stats["min_evidence_tokens"] = int(np.min(lengths))
        stats["max_evidence_tokens"] = int(np.max(lengths))
        stats["median_evidence_tokens"] = float(np.median(lengths))

        # Length histogram buckets
        stats["length_histogram"] = {
            "0-50": int(np.sum(lengths < 50)),
            "50-100": int(np.sum((lengths >= 50) & (lengths < 100))),
            "100-150": int(np.sum((lengths >= 100) & (lengths < 150))),
            "150-200": int(np.sum((lengths >= 150) & (lengths < 200))),
            "200-256": int(np.sum((lengths >= 200) & (lengths <= 256))),
            "256+": int(np.sum(lengths > 256)),
        }
    else:
        stats["avg_evidence_tokens"] = 0
        stats["length_histogram"] = {}

    stats["primary_success_rate"] = stats["primary_success"] / stats["total"] if stats["total"] > 0 else 0
    stats["fallback_rate"] = stats["fallback_used"] / stats["total"] if stats["total"] > 0 else 0
    stats["success_rate"] = stats["success"] / stats["total"] if stats["total"] > 0 else 0

    # Remove evidence_lengths from saved stats (too large)
    del stats["evidence_lengths"]

    # Save manifest
    manifest = {
        "created_at": datetime.now().isoformat(),
        "corpus_dir": str(corpus_dir),
        "split": split,
        "primary_method": primary_method,
        "fallback_method": fallback_method,
        "max_evidence_tokens": max_evidence_tokens,
        "context_sentences": context_sentences,
        "top_k_sentences": top_k_sentences,
        "tokenizer": tokenizer_name,
        "seed": seed,
        "stats": stats,
    }

    manifest_path = output_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Generate preview
    preview_samples = random.sample(samples, min(num_preview_samples, len(samples)))
    preview_path = output_dir / "samples_preview.md"

    with open(preview_path, "w", encoding="utf-8") as f:
        f.write("# Phase 1.5 Evidence Dataset: Sample Preview\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("## Statistics\n\n")
        f.write(f"- Total samples: {stats['total']}\n")
        f.write(f"- Success rate: {stats['success_rate']*100:.1f}%\n")
        f.write(f"- Primary method success: {stats['primary_success_rate']*100:.1f}%\n")
        f.write(f"- Fallback used: {stats['fallback_rate']*100:.1f}%\n")
        f.write(f"- Avg evidence tokens: {stats.get('avg_evidence_tokens', 0):.1f}\n\n")
        f.write("---\n\n")

        for i, sample in enumerate(preview_samples):
            f.write(f"## Sample {i+1} (ID: {sample['sample_id']})\n\n")
            f.write(f"**Question**: {sample['question']}\n\n")
            f.write(f"**Answer**: {sample['answer']}\n\n")
            f.write(f"**Doc ID**: {sample.get('doc_id', 'N/A')}\n\n")
            f.write(f"**Method**: {sample['evidence_method']}\n\n")

            meta = sample.get("evidence_meta", {})
            if meta.get("method") == "answer_span":
                f.write(f"- Answer sentence index: {meta.get('answer_sent_idx', 'N/A')}\n")
                f.write(f"- Window: {meta.get('window', 'N/A')}\n")
            elif meta.get("method") == "sentence_ranker":
                f.write(f"- Top-k indices: {meta.get('sentence_indices', 'N/A')}\n")

            f.write(f"- Tokens: {meta.get('final_tokens', 'N/A')}\n")
            f.write(f"- Truncated: {meta.get('truncated', False)}\n\n")

            f.write("**Evidence**:\n```\n")
            evidence = sample.get("evidence_text", "")
            f.write(evidence[:500] if len(evidence) > 500 else evidence)
            if len(evidence) > 500:
                f.write("\n... (truncated)")
            f.write("\n```\n\n")
            f.write("---\n\n")

    print(f"\nDataset built successfully!")
    print(f"  Output: {dataset_path}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Preview: {preview_path}")
    print(f"\nStatistics:")
    print(f"  Total: {stats['total']}")
    print(f"  Success: {stats['success']} ({stats['success_rate']*100:.1f}%)")
    print(f"  Primary success: {stats['primary_success']} ({stats['primary_success_rate']*100:.1f}%)")
    print(f"  Fallback used: {stats['fallback_used']} ({stats['fallback_rate']*100:.1f}%)")
    print(f"  Avg tokens: {stats.get('avg_evidence_tokens', 0):.1f}")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Phase 1.5 evidence dataset")
    parser.add_argument(
        "--corpus_dir",
        type=str,
        default="checkpoints/phase2_corpus",
        help="Path to Phase 2 corpus directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "train"],
        help="QA split to use",
    )
    parser.add_argument(
        "--primary_method",
        type=str,
        default="answer_span",
        choices=["answer_span", "sentence_ranker"],
        help="Primary evidence extraction method",
    )
    parser.add_argument(
        "--fallback_method",
        type=str,
        default="sentence_ranker",
        choices=["answer_span", "sentence_ranker"],
        help="Fallback method if primary fails",
    )
    parser.add_argument(
        "--max_evidence_tokens",
        type=int,
        default=256,
        help="Maximum evidence tokens",
    )
    parser.add_argument(
        "--context_sentences",
        type=int,
        default=2,
        help="Sentences around answer for answer_span",
    )
    parser.add_argument(
        "--top_k_sentences",
        type=int,
        default=3,
        help="Top sentences for sentence_ranker",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Tokenizer name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    build_evidence_dataset(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        split=args.split,
        primary_method=args.primary_method,
        fallback_method=args.fallback_method,
        max_evidence_tokens=args.max_evidence_tokens,
        context_sentences=args.context_sentences,
        top_k_sentences=args.top_k_sentences,
        tokenizer_name=args.tokenizer,
        seed=args.seed,
    )
