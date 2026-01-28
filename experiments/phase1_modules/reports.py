"""
Markdown report generation for Phase 1 verification and ablation.

Generates human-readable reports with:
- Eyeball samples (worst/best cases for manual inspection)
- Dashboard with links to key artifacts
- README with interpretation guide
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .utils import get_logger, save_json


def _truncate_text(text: str, max_chars: int = 500) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _escape_markdown(text: str) -> str:
    """Escape special markdown characters in text."""
    # For code blocks, we don't need to escape
    return text


def generate_worst_cases_report(
    nll_matrix: np.ndarray,
    corpus: Dict[str, str],
    output_dir: Union[str, Path],
    n_worst: int = 20,
    doc_ids: Optional[List[str]] = None,
) -> Path:
    """
    Generate markdown report of worst-margin documents for manual inspection.

    Args:
        nll_matrix: 2D array [num_docs, num_docs]
        corpus: Dict mapping doc_id -> document text
        output_dir: Directory to save report
        n_worst: Number of worst cases to include
        doc_ids: List of document IDs (default: doc_0, doc_1, ...)

    Returns:
        Path to generated report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_docs = nll_matrix.shape[0]
    if doc_ids is None:
        doc_ids = [f"doc_{i}" for i in range(n_docs)]

    # Compute margins
    cases = []
    for i in range(n_docs):
        correct_nll = nll_matrix[i, i]
        row = nll_matrix[i].copy()
        row[i] = np.inf
        best_wrong_idx = np.argmin(row)
        best_wrong_nll = row[best_wrong_idx]
        margin = best_wrong_nll - correct_nll

        cases.append({
            "doc_idx": i,
            "doc_id": doc_ids[i],
            "correct_nll": float(correct_nll),
            "best_wrong_idx": int(best_wrong_idx),
            "best_wrong_id": doc_ids[best_wrong_idx],
            "best_wrong_nll": float(best_wrong_nll),
            "margin": float(margin),
        })

    # Sort by margin (ascending = worst first)
    cases.sort(key=lambda x: x["margin"])
    worst_cases = cases[:n_worst]

    # Generate markdown
    lines = [
        "# Confusion Matrix: Worst Margin Cases",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
        f"Showing {n_worst} documents with lowest margin (correct_NLL - best_wrong_NLL).",
        "",
        "**Interpretation:**",
        "- Positive margin = correct z is better than best alternative (GOOD)",
        "- Negative margin = some other z produces lower NLL (BAD)",
        "- Low positive margin = the document is easily confused with another",
        "",
        "---",
        "",
    ]

    for rank, case in enumerate(worst_cases, 1):
        doc_id = case["doc_id"]
        doc_text = corpus.get(doc_id, "[Document text not found]")

        lines.extend([
            f"## #{rank}: {doc_id}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Correct NLL | {case['correct_nll']:.4f} |",
            f"| Best Wrong NLL | {case['best_wrong_nll']:.4f} (z from {case['best_wrong_id']}) |",
            f"| Margin | {case['margin']:.4f} |",
            "",
            "### Reference Text (first 500 chars)",
            "",
            "```",
            _truncate_text(doc_text, 500),
            "```",
            "",
            "---",
            "",
        ])

    # Write report
    report_path = output_dir / "eyeball_20_worst.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    get_logger().info(f"Generated worst cases report: {report_path}")
    return report_path


def generate_top_confusions_report(
    nll_matrix: np.ndarray,
    corpus: Dict[str, str],
    output_dir: Union[str, Path],
    n_top: int = 20,
    doc_ids: Optional[List[str]] = None,
) -> Path:
    """
    Generate report of top confused document pairs.

    Args:
        nll_matrix: 2D array [num_docs, num_docs]
        corpus: Dict mapping doc_id -> document text
        output_dir: Directory to save report
        n_top: Number of top confusions to include
        doc_ids: List of document IDs

    Returns:
        Path to generated report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_docs = nll_matrix.shape[0]
    if doc_ids is None:
        doc_ids = [f"doc_{i}" for i in range(n_docs)]

    # Find pairs where doc_i has low NLL with z_j (i != j)
    confusions = []
    for i in range(n_docs):
        for j in range(n_docs):
            if i == j:
                continue
            # NLL of doc_i when using z_j
            nll_ij = nll_matrix[i, j]
            # Compare to correct NLL
            nll_ii = nll_matrix[i, i]
            confusion_score = nll_ii - nll_ij  # Positive = z_j is better than z_i for doc_i

            confusions.append({
                "doc_i": i,
                "doc_j": j,
                "doc_id_i": doc_ids[i],
                "doc_id_j": doc_ids[j],
                "nll_correct": float(nll_ii),
                "nll_confused": float(nll_ij),
                "confusion_score": float(confusion_score),
            })

    # Sort by confusion_score descending (most confused first)
    confusions.sort(key=lambda x: x["confusion_score"], reverse=True)
    top_confusions = confusions[:n_top]

    # Generate markdown
    lines = [
        "# Top Confused Document Pairs",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
        f"Showing {n_top} document pairs where z_j produces lower NLL for doc_i than z_i does.",
        "",
        "**Interpretation:**",
        "- High confusion_score = z_j is much better than z_i for predicting doc_i",
        "- These pairs may share similar content or structure",
        "",
        "---",
        "",
    ]

    for rank, conf in enumerate(top_confusions, 1):
        doc_text_i = corpus.get(conf["doc_id_i"], "[Not found]")
        doc_text_j = corpus.get(conf["doc_id_j"], "[Not found]")

        lines.extend([
            f"## #{rank}: {conf['doc_id_i']} confused with {conf['doc_id_j']}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| NLL({conf['doc_id_i']} | z_{conf['doc_id_i']}) | {conf['nll_correct']:.4f} |",
            f"| NLL({conf['doc_id_i']} | z_{conf['doc_id_j']}) | {conf['nll_confused']:.4f} |",
            f"| Confusion Score | {conf['confusion_score']:.4f} |",
            "",
            f"### {conf['doc_id_i']} (first 300 chars)",
            "",
            "```",
            _truncate_text(doc_text_i, 300),
            "```",
            "",
            f"### {conf['doc_id_j']} (first 300 chars)",
            "",
            "```",
            _truncate_text(doc_text_j, 300),
            "```",
            "",
            "---",
            "",
        ])

    # Write report
    report_path = output_dir / "top_confusions.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    get_logger().info(f"Generated top confusions report: {report_path}")
    return report_path


def generate_eyeball_report(
    samples: List[Dict[str, Any]],
    output_path: Union[str, Path],
    title: str = "Eyeball Samples",
    description: str = "",
) -> Path:
    """
    Generate markdown report for eyeball inspection of generation samples.

    Args:
        samples: List of sample dicts with doc_id, ref_snippet, gen_snippet, score, etc.
        output_path: Full path to output file
        title: Report title
        description: Description of what this report shows

    Returns:
        Path to generated report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# {title}",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
    ]

    if description:
        lines.extend([description, ""])

    lines.extend([
        f"Total samples: {len(samples)}",
        "",
        "---",
        "",
    ])

    for rank, sample in enumerate(samples, 1):
        doc_id = sample.get("doc_id", f"sample_{rank}")
        score = sample.get("score", 0)
        ref_snippet = sample.get("ref_snippet", "[No reference]")
        gen_snippet = sample.get("gen_snippet", "[No generation]")

        lines.extend([
            f"## #{rank}: {doc_id} (Score: {score:.4f})",
            "",
        ])

        # Add any additional metrics
        for key in ["ref_len", "gen_len", "len_ratio"]:
            if key in sample:
                lines.append(f"- {key}: {sample[key]}")

        lines.extend([
            "",
            "### Reference",
            "",
            "```",
            _truncate_text(str(ref_snippet), 600),
            "```",
            "",
            "### Generated",
            "",
            "```",
            _truncate_text(str(gen_snippet), 600),
            "```",
            "",
            "---",
            "",
        ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    get_logger().info(f"Generated eyeball report: {output_path}")
    return output_path


def generate_dashboard(
    results: Dict[str, Any],
    output_dir: Union[str, Path],
    run_dir_name: str,
) -> Path:
    """
    Generate dashboard.md with links to top artifacts to check.

    Args:
        results: Dict with module results
        output_dir: Directory to save dashboard (03_summary/)
        run_dir_name: Name of the run directory (timestamp)

    Returns:
        Path to generated dashboard
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Phase 1 Analysis Dashboard",
        "",
        f"Generated: {datetime.now().isoformat()}",
        f"Run directory: `{run_dir_name}`",
        "",
        "---",
        "",
        "## Top 5 Artifacts to Check First",
        "",
        "### 1. Confusion Heatmap",
        "**Path:** `01_verification/A1_confusion/artifacts/confusion_heatmap.png`",
        "",
        "**What to look for:**",
        "- Diagonal should be dark (low NLL = correct z works well)",
        "- Off-diagonal should be bright (high NLL = wrong z produces poor predictions)",
        "- Any dark off-diagonal clusters indicate potentially confused document pairs",
        "",
        "### 2. Worst Cases Report",
        "**Path:** `01_verification/A1_confusion/samples/eyeball_20_worst.md`",
        "",
        "**What to look for:**",
        "- Review 20 docs with lowest margin between correct and best-wrong z",
        "- Are the confusions reasonable (similar documents)?",
        "- Are there any negative margins (failure cases)?",
        "",
        "### 3. Z-only Best Examples",
        "**Path:** `01_verification/A2_zonly/samples/eyeball_20_best.md`",
        "",
        "**What to look for:**",
        "- Do high-ROUGE generations actually capture the document content?",
        "- Is the generated text coherent and factually similar to reference?",
        "",
        "### 4. Z-only Worst Examples",
        "**Path:** `01_verification/A2_zonly/samples/eyeball_20_worst.md`",
        "",
        "**What to look for:**",
        "- What goes wrong in low-ROUGE cases?",
        "- Is it hallucination, wrong topic, or just different wording?",
        "",
        "### 5. Random Projection Degradation",
        "**Path:** `02_ablations/B2_projection/proj_random_frozen/samples/eyeball_10_examples.md`",
        "",
        "**What to look for:**",
        "- Confirm outputs are garbage when projection is random",
        "- This validates that the trained projection is essential",
        "",
        "---",
        "",
        "## Manual Sample Count",
        "",
        "| Category | Count | File |",
        "|----------|-------|------|",
        "| Confusion worst cases | 20 | `A1_confusion/samples/eyeball_20_worst.md` |",
        "| Z-only best | 20 | `A2_zonly/samples/eyeball_20_best.md` |",
        "| Z-only worst | 20 | `A2_zonly/samples/eyeball_20_worst.md` |",
        "| Random projection degradation | 10 | `B2_projection/proj_random_frozen/samples/eyeball_10_examples.md` |",
        "| **Total** | **~70** | |",
        "",
        "---",
        "",
        "## Quick Metrics Summary",
        "",
        "| Module | Key Metric | Value | Status |",
        "|--------|------------|-------|--------|",
    ]

    # Add metrics from results
    if "A1" in results and isinstance(results["A1"], dict):
        a1 = results["A1"]
        top1 = a1.get("top1_acc", 0) * 100
        margin = a1.get("mean_margin", 0)
        status1 = "PASS" if top1 > 95 else "WARN" if top1 > 80 else "FAIL"
        status2 = "PASS" if margin > 0.5 else "WARN" if margin > 0 else "FAIL"
        lines.append(f"| A1 Confusion | Top-1 Accuracy | {top1:.1f}% | {status1} |")
        lines.append(f"| A1 Confusion | Mean Margin | {margin:.3f} | {status2} |")

    if "A2" in results and isinstance(results["A2"], dict):
        a2 = results["A2"]
        rouge = a2.get("mean_rouge_l", 0)
        status = "PASS" if rouge > 0.3 else "WARN" if rouge > 0.1 else "FAIL"
        lines.append(f"| A2 Z-only | Mean ROUGE-L | {rouge:.3f} | {status} |")

    if "A3" in results and isinstance(results["A3"], dict):
        a3 = results["A3"]
        delta = a3.get("delta_top1", 0) * 100
        status = "PASS" if delta > 50 else "WARN" if delta > 20 else "FAIL"
        lines.append(f"| A3 Z-shuffle | Delta Top-1 | {delta:+.1f}% | {status} |")

    if "B1" in results and isinstance(results["B1"], dict):
        b1 = results["B1"]
        if "alpha_zero" in b1:
            zero_acc = b1["alpha_zero"].get("top1_acc", 0) * 100
            status = "PASS" if zero_acc < 10 else "WARN" if zero_acc < 50 else "FAIL"
            lines.append(f"| B1 Alpha=0 | Top-1 Accuracy | {zero_acc:.1f}% | {status} |")

    if "B2" in results and isinstance(results["B2"], dict):
        b2 = results["B2"]
        if "proj_random_frozen" in b2:
            rand_acc = b2["proj_random_frozen"].get("top1_acc", 0) * 100
            status = "PASS" if rand_acc < 10 else "WARN" if rand_acc < 50 else "FAIL"
            lines.append(f"| B2 Random Proj | Top-1 Accuracy | {rand_acc:.1f}% | {status} |")

    lines.extend([
        "",
        "---",
        "",
        "## File Structure",
        "",
        "```",
        f"{run_dir_name}/",
        "├── 00_meta/              # Run metadata and config",
        "├── 00_logs/              # Log files (run.log, debug.log, warnings.log)",
        "├── 01_verification/      # Verification modules (A1, A2, A3)",
        "│   ├── A1_confusion/     # NLL confusion matrix",
        "│   ├── A2_zonly/         # Z-only generation",
        "│   └── A3_zshuffle/      # Z-shuffle sanity check",
        "├── 02_ablations/         # Ablation studies (B1, B2)",
        "│   ├── B1_alpha/         # Alpha ablation",
        "│   └── B2_projection/    # Projection ablation",
        "├── 03_summary/           # This dashboard and summary files",
        "└── 04_cache/             # Partial computation cache",
        "```",
        "",
    ])

    # Write dashboard
    dashboard_path = output_dir / "dashboard.md"
    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    get_logger().info(f"Generated dashboard: {dashboard_path}")
    return dashboard_path


def generate_readme(
    output_dir: Union[str, Path],
    args: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Generate README.md with how to run and interpret results.

    Args:
        output_dir: Directory to save README (03_summary/)
        args: Command line arguments used

    Returns:
        Path to generated README
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Phase 1 Verification & Ablation Results",
        "",
        "This directory contains the results of Phase 1 (Write Phase) verification and ablation experiments.",
        "",
        "## How to Run",
        "",
        "```bash",
        "# Full run with verification + ablations",
        "python experiments/phase1_runner.py \\",
        "  --ckpt_dir checkpoints/phase1_v2 \\",
        "  --out_root results/phase1_analysis \\",
        "  --run_verification \\",
        "  --run_ablations",
        "",
        "# Quick smoke test (10 docs)",
        "python experiments/phase1_runner.py \\",
        "  --ckpt_dir checkpoints/phase1_v2 \\",
        "  --out_root results/phase1_analysis \\",
        "  --num_docs 10 \\",
        "  --run_verification",
        "```",
        "",
        "## Interpretation Guide",
        "",
        "### A1: Confusion Matrix",
        "",
        "- **Top-1 Accuracy**: Percentage of documents where the correct z produces the lowest NLL",
        "  - Target: >95% (100% is ideal)",
        "- **Mean Margin**: Average difference between best-wrong NLL and correct NLL",
        "  - Target: >0.5 (higher is better)",
        "",
        "### A2: Z-only Generation",
        "",
        "- **Mean ROUGE-L**: Average ROUGE-L score between generated and reference text",
        "  - Target: >0.3 for meaningful generation",
        "- Check eyeball samples to verify generations capture document semantics",
        "",
        "### A3: Z-shuffle",
        "",
        "- **Delta Top-1**: Drop in top-1 accuracy after shuffling z-to-doc mapping",
        "  - Target: Large drop (>50%) proves z encodes document-specific information",
        "",
        "### B1: Alpha Ablation",
        "",
        "- **alpha=0**: Should have near-random performance (proves alpha scaling is needed)",
        "- **alpha=1**: Baseline without learned scaling",
        "- **alpha=trained**: Should be best",
        "",
        "### B2: Projection Ablation",
        "",
        "- **Random projection**: Should have near-random performance (proves projection is learned)",
        "",
        "## Key Files",
        "",
        "- `dashboard.md`: Quick overview with links to top artifacts",
        "- `ablation_summary.csv`: One row per variant with all metrics",
        "- `ablation_summary.json`: Same data in JSON format",
        "",
        "## Eyeball Samples",
        "",
        "Manual inspection of ~70 samples is recommended:",
        "",
        "1. `A1_confusion/samples/eyeball_20_worst.md` - Documents with lowest confidence",
        "2. `A2_zonly/samples/eyeball_20_best.md` - Best z-only generations",
        "3. `A2_zonly/samples/eyeball_20_worst.md` - Worst z-only generations",
        "4. `B2_projection/proj_random_frozen/samples/eyeball_10_examples.md` - Random projection failures",
        "",
    ]

    # Add args if provided
    if args:
        lines.extend([
            "## Run Configuration",
            "",
            "```json",
            str(args),
            "```",
            "",
        ])

    # Write README
    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    get_logger().info(f"Generated README: {readme_path}")
    return readme_path


def generate_ablation_summary(
    all_results: Dict[str, Dict[str, Any]],
    output_dir: Union[str, Path],
) -> tuple:
    """
    Generate ablation_summary.csv and ablation_summary.json.

    Args:
        all_results: Dict mapping module/variant name -> metrics
        output_dir: Directory to save summary files

    Returns:
        Tuple of (csv_path, json_path)
    """
    import csv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Flatten results into rows
    rows = []
    for module_name, module_results in all_results.items():
        if isinstance(module_results, dict):
            # Check if it's a dict of variants (B1, B2) or direct metrics (A1, A2, A3)
            first_value = next(iter(module_results.values()), None)
            if isinstance(first_value, dict):
                # It's a dict of variants
                for variant_name, metrics in module_results.items():
                    row = {"module": module_name, "variant": variant_name}
                    if isinstance(metrics, dict):
                        row.update(metrics)
                    rows.append(row)
            else:
                # It's direct metrics
                row = {"module": module_name, "variant": "default"}
                row.update(module_results)
                rows.append(row)

    # Get all unique keys
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    all_keys = sorted(all_keys)

    # Ensure module and variant are first
    key_order = ["module", "variant"] + [k for k in all_keys if k not in ["module", "variant"]]

    # Write CSV
    csv_path = output_dir / "ablation_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=key_order, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Write JSON
    json_path = output_dir / "ablation_summary.json"
    save_json(rows, json_path)

    get_logger().info(f"Generated ablation summary: {csv_path}, {json_path}")
    return csv_path, json_path
