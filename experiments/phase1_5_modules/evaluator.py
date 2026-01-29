"""
Phase 1.5: Evidence Generation Evaluation

This module evaluates the quality of generated evidence:
- Answer coverage: Does generated evidence contain the gold answer?
- Source overlap: How much of generated evidence comes from source doc? (extractive-ness)
- N-gram overlap: 4-gram overlap with source document
- ROUGE-L: Standard overlap metric vs target evidence
- Length statistics: Generation length vs target length

Also generates human-checkable eyeball reports:
- eyeball_20_best.md: Best samples by ROUGE-L
- eyeball_20_worst.md: Worst samples by ROUGE-L
- failure_cases.md: Cases where answer_coverage = 0
"""

import json
import os
import sys
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.phase1_modules.utils import (
    Timer,
    get_logger,
    save_json,
    append_jsonl,
    load_jsonl_cache,
)
from experiments.phase1_5_modules.model_wrapper import Phase15ForwardWrapper


# =============================================================================
# METRICS
# =============================================================================

def compute_answer_coverage(generated: str, answer: str) -> float:
    """
    Compute if the gold answer appears in generated evidence.

    Args:
        generated: Generated evidence text
        answer: Gold answer string

    Returns:
        1.0 if answer is substring of evidence (case-insensitive), 0.0 otherwise
    """
    if not generated or not answer:
        return 0.0

    return 1.0 if answer.lower() in generated.lower() else 0.0


def compute_source_overlap(generated: str, source_doc: str) -> float:
    """
    Compute token-level overlap between generated evidence and source document.

    This measures "extractive-ness" - how much of the generation comes from the source.

    = |tokens(gen) ∩ tokens(doc)| / |tokens(gen)|

    Args:
        generated: Generated evidence text
        source_doc: Source document text

    Returns:
        Overlap ratio [0, 1]. Higher = more extractive.
    """
    if not generated:
        return 0.0

    gen_tokens = set(generated.lower().split())
    doc_tokens = set(source_doc.lower().split())

    if not gen_tokens:
        return 0.0

    overlap = len(gen_tokens & doc_tokens)
    return overlap / len(gen_tokens)


def compute_ngram_overlap(generated: str, source_doc: str, n: int = 4) -> float:
    """
    Compute n-gram overlap between generated evidence and source document.

    = |n-grams(gen) ∩ n-grams(doc)| / |n-grams(gen)|

    Args:
        generated: Generated evidence text
        source_doc: Source document text
        n: N-gram size (default: 4)

    Returns:
        N-gram overlap ratio [0, 1]
    """
    def get_ngrams(text: str, n: int) -> Counter:
        words = text.lower().split()
        if len(words) < n:
            return Counter()
        return Counter(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

    if not generated:
        return 0.0

    gen_ngrams = get_ngrams(generated, n)
    doc_ngrams = get_ngrams(source_doc, n)

    if not gen_ngrams:
        return 0.0

    # Count overlapping n-grams
    overlap = sum((gen_ngrams & doc_ngrams).values())
    total = sum(gen_ngrams.values())

    return overlap / total if total > 0 else 0.0


def compute_rouge_l(generated: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 score.

    Args:
        generated: Generated text
        reference: Reference text

    Returns:
        ROUGE-L F1 score
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, generated)
        return scores['rougeL'].fmeasure
    except ImportError:
        # Fallback: simple LCS-based calculation
        return _compute_lcs_f1(generated, reference)


def _compute_lcs_f1(generated: str, reference: str) -> float:
    """Fallback LCS-based ROUGE-L calculation."""
    if not generated or not reference:
        return 0.0

    gen_words = generated.lower().split()
    ref_words = reference.lower().split()

    if not gen_words or not ref_words:
        return 0.0

    # Compute LCS length
    m, n = len(gen_words), len(ref_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gen_words[i-1] == ref_words[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs_len = dp[m][n]

    precision = lcs_len / m if m > 0 else 0
    recall = lcs_len / n if n > 0 else 0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_length_stats(generated: str, reference: str, tokenizer=None) -> Dict:
    """
    Compute length-related statistics.

    Args:
        generated: Generated text
        reference: Reference text
        tokenizer: Optional tokenizer for token-level stats

    Returns:
        Dict with length statistics
    """
    gen_words = len(generated.split()) if generated else 0
    ref_words = len(reference.split()) if reference else 0

    stats = {
        "gen_words": gen_words,
        "ref_words": ref_words,
        "word_ratio": gen_words / ref_words if ref_words > 0 else 0,
        "gen_chars": len(generated) if generated else 0,
        "ref_chars": len(reference) if reference else 0,
    }

    if tokenizer:
        gen_tokens = len(tokenizer.encode(generated, add_special_tokens=False)) if generated else 0
        ref_tokens = len(tokenizer.encode(reference, add_special_tokens=False)) if reference else 0
        stats["gen_tokens"] = gen_tokens
        stats["ref_tokens"] = ref_tokens
        stats["token_ratio"] = gen_tokens / ref_tokens if ref_tokens > 0 else 0

    return stats


def find_exact_span_in_doc(generated: str, source_doc: str) -> Optional[Tuple[int, int]]:
    """
    Find the longest matching substring position in source document.

    Args:
        generated: Generated text
        source_doc: Source document

    Returns:
        Tuple of (start, end) positions in source_doc, or None if no significant match
    """
    if not generated or not source_doc:
        return None

    matcher = SequenceMatcher(None, source_doc.lower(), generated.lower())
    match = matcher.find_longest_match(0, len(source_doc), 0, len(generated))

    # Require minimum match length (20 chars)
    if match.size > 20:
        return (match.a, match.a + match.size)

    return None


# =============================================================================
# EVALUATOR CLASS
# =============================================================================

class Phase15Evaluator:
    """
    Evaluator for Phase 1.5 evidence generation.
    """

    def __init__(
        self,
        model,
        z_pool,
        corpus: Dict[str, str],
        tokenizer,
        run_dir: Path,
        device: str = "cuda",
    ):
        """
        Args:
            model: WritePhaseModel with LoRA
            z_pool: ZPoolManager
            corpus: Dict mapping doc_id -> document text
            tokenizer: LLM tokenizer
            run_dir: Run directory
            device: Device string
        """
        self.model = model
        self.z_pool = z_pool
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.run_dir = Path(run_dir)
        self.device = device
        self.logger = get_logger()

        # Directories
        self.eval_dir = self.run_dir / "03_eval"
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir = self.eval_dir / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = self.eval_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Cache
        self.cache_dir = self.run_dir / "05_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.eval_cache_path = self.cache_dir / "eval_results.jsonl"

        # Create forward wrapper
        self.wrapper = Phase15ForwardWrapper(model, tokenizer, device)
        # Reset logging flag for fresh logging on each evaluation run
        self.wrapper._first_generation_logged = False

    def generate_evidence(
        self,
        doc_id: str,
        query: str,
        max_new_tokens: int = 256,
    ) -> str:
        """
        Generate evidence for a single query.

        Args:
            doc_id: Document ID to get z vector
            query: Query string
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated evidence string
        """
        self.model.eval()

        with torch.no_grad():
            z = self.z_pool.get_z(doc_id).to(self.device)
            evidence = self.wrapper.generate_evidence(
                z=z,
                query_text=query,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        return evidence

    def evaluate(
        self,
        dataset_path: Path,
        max_new_tokens: int = 256,
        num_samples: Optional[int] = None,
    ) -> Dict:
        """
        Run full evaluation.

        Args:
            dataset_path: Path to dataset.jsonl
            max_new_tokens: Max tokens for generation
            num_samples: Optional sample limit

        Returns:
            Evaluation metrics dict
        """
        self.logger.info("=" * 60)
        self.logger.info("PHASE 1.5 EVALUATION")
        self.logger.info("=" * 60)

        # Log generation config
        self.logger.info("-" * 40)
        self.logger.info("GENERATION CONFIG:")
        self.logger.info(f"  max_new_tokens: {max_new_tokens}")
        self.logger.info(f"  do_sample: False (greedy)")
        self.logger.info(f"  temperature: 1.0 (unused when do_sample=False)")
        self.logger.info(f"  top_p: 0.9 (unused when do_sample=False)")
        self.logger.info(f"  repetition_penalty: 1.1 (prevents END collapse)")
        self.logger.info(f"  dynamic_max_tokens: target_len + 20 (capped at {max_new_tokens})")
        self.logger.info(f"  post_processing: _clean_evidence_output (removes END patterns)")
        self.logger.info("-" * 40)

        # Load dataset
        samples = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)
                if sample.get("evidence_text") and sample.get("doc_id"):
                    samples.append(sample)
                    if num_samples and len(samples) >= num_samples:
                        break

        self.logger.info(f"Loaded {len(samples)} samples for evaluation")

        # Load cached results
        cache = load_jsonl_cache(self.eval_cache_path, key_field="sample_id")
        self.logger.info(f"Cached results: {len(cache)}")

        # Evaluate each sample
        results = []
        all_metrics = {
            "answer_coverage": [],
            "source_overlap": [],
            "ngram_overlap_4": [],
            "rouge_l": [],
            "length_ratios": [],
        }

        self.model.eval()

        for sample in tqdm(samples, desc="Evaluating"):
            sample_id = sample["sample_id"]

            # Check cache
            if sample_id in cache:
                result = cache[sample_id]
            else:
                # Generate evidence
                doc_id = sample["doc_id"]
                query = sample["question"]
                answer = sample["answer"]
                target_evidence = sample["evidence_text"]
                source_doc = self.corpus.get(doc_id, "")

                # Dynamic max_new_tokens based on target evidence length (+ margin)
                # This prevents over-generation and topic drift
                target_tokens = len(self.tokenizer.encode(target_evidence, add_special_tokens=False))
                dynamic_max_tokens = min(target_tokens + 20, max_new_tokens)  # target + margin, capped

                generated = self.generate_evidence(doc_id, query, dynamic_max_tokens)

                # Compute metrics
                ans_cov = compute_answer_coverage(generated, answer)
                src_overlap = compute_source_overlap(generated, source_doc)
                ngram_overlap = compute_ngram_overlap(generated, source_doc, n=4)
                rouge_l = compute_rouge_l(generated, target_evidence)
                len_stats = compute_length_stats(generated, target_evidence, self.tokenizer)

                # Find exact span
                span = find_exact_span_in_doc(generated, source_doc)

                result = {
                    "sample_id": sample_id,
                    "doc_id": doc_id,
                    "question": query,
                    "answer": answer,
                    "target_evidence": target_evidence,
                    "generated_evidence": generated,
                    "answer_coverage": ans_cov,
                    "source_overlap": src_overlap,
                    "ngram_overlap_4": ngram_overlap,
                    "rouge_l": rouge_l,
                    "length_stats": len_stats,
                    "exact_span": span,
                }

                # Cache result
                append_jsonl(self.eval_cache_path, result)

            results.append(result)

            # Collect metrics
            all_metrics["answer_coverage"].append(result["answer_coverage"])
            all_metrics["source_overlap"].append(result["source_overlap"])
            all_metrics["ngram_overlap_4"].append(result["ngram_overlap_4"])
            all_metrics["rouge_l"].append(result["rouge_l"])
            if "length_stats" in result:
                ratio = result["length_stats"].get("word_ratio", 1.0)
                all_metrics["length_ratios"].append(ratio)

        # Compute aggregate metrics
        import numpy as np

        summary = {
            "num_samples": len(results),
            "answer_coverage": float(np.mean(all_metrics["answer_coverage"])),
            "answer_coverage_std": float(np.std(all_metrics["answer_coverage"])),
            "source_overlap": float(np.mean(all_metrics["source_overlap"])),
            "source_overlap_std": float(np.std(all_metrics["source_overlap"])),
            "ngram_overlap_4": float(np.mean(all_metrics["ngram_overlap_4"])),
            "ngram_overlap_4_std": float(np.std(all_metrics["ngram_overlap_4"])),
            "rouge_l": float(np.mean(all_metrics["rouge_l"])),
            "rouge_l_std": float(np.std(all_metrics["rouge_l"])),
            "mean_length_ratio": float(np.mean(all_metrics["length_ratios"])) if all_metrics["length_ratios"] else 0,
        }

        # Pass/fail checks
        summary["warnings"] = []
        if summary["source_overlap"] < 0.5:
            summary["warnings"].append("source_overlap < 50% - may be paraphrasing")
        if summary["answer_coverage"] < 0.7:
            summary["warnings"].append("answer_coverage < 70% - missing key info")

        # Save metrics
        save_json(summary, self.eval_dir / "evidence_metrics.json")

        # Save per-sample CSV
        self._save_eval_table(results)

        # Generate eyeball reports
        self._generate_eyeball_reports(results)

        # Generate plots
        self._generate_plots(all_metrics)

        # Log summary
        self.logger.info("\nEvaluation Summary:")
        self.logger.info(f"  Answer Coverage: {summary['answer_coverage']*100:.1f}%")
        self.logger.info(f"  Source Overlap: {summary['source_overlap']*100:.1f}%")
        self.logger.info(f"  4-gram Overlap: {summary['ngram_overlap_4']*100:.1f}%")
        self.logger.info(f"  ROUGE-L: {summary['rouge_l']:.3f}")

        if summary["warnings"]:
            for warn in summary["warnings"]:
                self.logger.warning(f"  WARNING: {warn}")

        return summary

    def _save_eval_table(self, results: List[Dict]):
        """Save per-sample evaluation table as CSV."""
        import csv

        csv_path = self.eval_dir / "evidence_eval_table.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sample_id", "doc_id", "answer_coverage", "source_overlap",
                "ngram_overlap_4", "rouge_l", "length_ratio", "has_exact_span"
            ])

            for r in results:
                len_ratio = r.get("length_stats", {}).get("word_ratio", 0)
                has_span = 1 if r.get("exact_span") else 0

                writer.writerow([
                    r["sample_id"],
                    r["doc_id"],
                    f"{r['answer_coverage']:.3f}",
                    f"{r['source_overlap']:.3f}",
                    f"{r['ngram_overlap_4']:.3f}",
                    f"{r['rouge_l']:.3f}",
                    f"{len_ratio:.3f}",
                    has_span,
                ])

    def _generate_eyeball_reports(self, results: List[Dict]):
        """Generate markdown eyeball reports."""
        # Sort by ROUGE-L
        sorted_by_rouge = sorted(results, key=lambda x: x["rouge_l"], reverse=True)

        # Best 20
        self._write_eyeball_report(
            sorted_by_rouge[:20],
            self.samples_dir / "eyeball_20_best.md",
            "Best 20 Samples by ROUGE-L",
            "These are the best performing samples by ROUGE-L score.",
        )

        # Worst 20
        self._write_eyeball_report(
            sorted_by_rouge[-20:],
            self.samples_dir / "eyeball_20_worst.md",
            "Worst 20 Samples by ROUGE-L",
            "These are the worst performing samples by ROUGE-L score.",
        )

        # Failure cases (answer_coverage = 0)
        failures = [r for r in results if r["answer_coverage"] == 0]
        self._write_eyeball_report(
            failures[:20],
            self.samples_dir / "failure_cases.md",
            "Failure Cases (Answer Coverage = 0)",
            f"These {len(failures)} samples have answer_coverage=0, meaning the generated evidence does not contain the gold answer.",
        )

    def _write_eyeball_report(
        self,
        samples: List[Dict],
        path: Path,
        title: str,
        description: str,
    ):
        """Write a single eyeball report."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            f.write(f"{description}\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write("---\n\n")

            for i, r in enumerate(samples):
                f.write(f"## Sample {i+1} (ID: {r['sample_id']})\n\n")
                f.write(f"**Doc ID**: {r['doc_id']}\n\n")
                f.write(f"**Question**: {r['question']}\n\n")
                f.write(f"**Answer**: {r['answer']}\n\n")

                f.write("### Metrics\n\n")
                f.write(f"| Metric | Value |\n")
                f.write(f"|--------|-------|\n")
                f.write(f"| Answer Coverage | {r['answer_coverage']:.3f} |\n")
                f.write(f"| Source Overlap | {r['source_overlap']:.3f} |\n")
                f.write(f"| 4-gram Overlap | {r['ngram_overlap_4']:.3f} |\n")
                f.write(f"| ROUGE-L | {r['rouge_l']:.3f} |\n")

                # Exact span info
                if r.get("exact_span"):
                    start, end = r["exact_span"]
                    f.write(f"| Exact Span | chars {start}-{end} |\n")
                else:
                    f.write(f"| Exact Span | not found |\n")
                f.write("\n")

                f.write("### Target Evidence\n\n")
                f.write("```\n")
                target = r.get("target_evidence", "")
                f.write(target[:500] if len(target) > 500 else target)
                if len(target) > 500:
                    f.write("\n... (truncated)")
                f.write("\n```\n\n")

                f.write("### Generated Evidence\n\n")
                f.write("```\n")
                generated = r.get("generated_evidence", "")
                f.write(generated[:500] if len(generated) > 500 else generated)
                if len(generated) > 500:
                    f.write("\n... (truncated)")
                f.write("\n```\n\n")

                f.write("---\n\n")

    def _generate_plots(self, metrics: Dict[str, List[float]]):
        """Generate evaluation plots."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            # Source overlap histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(metrics["source_overlap"], bins=20, edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(metrics["source_overlap"]), color='r', linestyle='--',
                       label=f'Mean: {np.mean(metrics["source_overlap"]):.3f}')
            ax.axvline(0.5, color='orange', linestyle=':', label='Threshold: 0.5')
            ax.set_xlabel("Source Overlap")
            ax.set_ylabel("Count")
            ax.set_title("Source Overlap Distribution (Extractive-ness)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.artifacts_dir / "source_overlap_hist.png", dpi=150)
            plt.savefig(self.artifacts_dir / "source_overlap_hist.pdf")
            plt.close()

            # ROUGE-L histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(metrics["rouge_l"], bins=20, edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(metrics["rouge_l"]), color='r', linestyle='--',
                       label=f'Mean: {np.mean(metrics["rouge_l"]):.3f}')
            ax.set_xlabel("ROUGE-L")
            ax.set_ylabel("Count")
            ax.set_title("ROUGE-L Distribution")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.artifacts_dir / "rouge_hist.png", dpi=150)
            plt.savefig(self.artifacts_dir / "rouge_hist.pdf")
            plt.close()

            # Answer coverage by length
            if metrics["length_ratios"]:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(metrics["length_ratios"], metrics["answer_coverage"], alpha=0.5)
                ax.set_xlabel("Length Ratio (gen/ref)")
                ax.set_ylabel("Answer Coverage")
                ax.set_title("Answer Coverage vs Length Ratio")
                ax.axhline(0.5, color='r', linestyle='--', alpha=0.5)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.artifacts_dir / "coverage_by_length.png", dpi=150)
                plt.savefig(self.artifacts_dir / "coverage_by_length.pdf")
                plt.close()

        except Exception as e:
            self.logger.warning(f"Failed to generate plots: {e}")


def run_phase15_evaluation(
    model,
    z_pool,
    corpus: Dict[str, str],
    tokenizer,
    dataset_path: Path,
    run_dir: Path,
    max_new_tokens: int = 256,
    num_samples: Optional[int] = None,
    device: str = "cuda",
) -> Dict:
    """
    Entry point for Phase 1.5 evaluation.

    Args:
        model: WritePhaseModel with LoRA
        z_pool: ZPoolManager
        corpus: Document corpus
        tokenizer: LLM tokenizer
        dataset_path: Path to dataset.jsonl
        run_dir: Run directory
        max_new_tokens: Max tokens for generation
        num_samples: Optional sample limit
        device: Device string

    Returns:
        Evaluation metrics dict
    """
    evaluator = Phase15Evaluator(
        model=model,
        z_pool=z_pool,
        corpus=corpus,
        tokenizer=tokenizer,
        run_dir=run_dir,
        device=device,
    )

    return evaluator.evaluate(
        dataset_path=dataset_path,
        max_new_tokens=max_new_tokens,
        num_samples=num_samples,
    )
