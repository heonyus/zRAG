#!/usr/bin/env python3
"""
Phase 1 Verification & Ablation Runner

Paper-quality verification and inference-only ablation suite for Phase 1 (Write Phase).

IMPORTANT: DEFAULT BEHAVIOR IS EVAL-ONLY. NO TRAINING IS PERFORMED.
Training-related features are behind explicit flags that are OFF by default.

Usage:
    # Full verification + ablations (EVAL-ONLY)
    python experiments/phase1_runner.py \\
        --ckpt_dir checkpoints/phase1_v2 \\
        --out_root results/phase1_analysis \\
        --run_verification \\
        --run_ablations

    # Quick smoke test (10 docs)
    python experiments/phase1_runner.py \\
        --ckpt_dir checkpoints/phase1_v2 \\
        --num_docs 10 \\
        --run_verification

Author: zRAG Team
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from experiments.phase1_modules.utils import (
    Timer,
    get_env_info,
    get_logger,
    save_json,
    set_seed,
    setup_logging,
)
from experiments.phase1_modules.reports import (
    generate_ablation_summary,
    generate_dashboard,
    generate_readme,
)


# ==============================================================================
# CRITICAL: Training flags are OFF by default
# ==============================================================================
TRAINING_FLAGS = [
    "run_proj_only_baseline",
    "sweep_dropout",
    "sweep_mtokens",
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 1 Verification & Ablation Runner (EVAL-ONLY by default)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full evaluation run
  python experiments/phase1_runner.py --ckpt_dir checkpoints/phase1_v2 --run_verification --run_ablations

  # Quick smoke test
  python experiments/phase1_runner.py --ckpt_dir checkpoints/phase1_v2 --num_docs 10 --run_verification
        """,
    )

    # Required arguments
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="checkpoints/phase1_v2",
        help="Checkpoint directory containing z_pool.pt, projection.pt",
    )

    # Output settings
    parser.add_argument(
        "--out_root",
        type=str,
        default="results/phase1_analysis",
        help="Root directory for output (default: results/phase1_analysis)",
    )
    parser.add_argument(
        "--resume_dir",
        type=str,
        default=None,
        help="Resume from existing run directory (e.g., results/phase1_analysis/20260128_123048)",
    )

    # Evaluation settings
    parser.add_argument(
        "--num_docs",
        type=int,
        default=200,
        help="Number of documents to evaluate (default: 200)",
    )
    parser.add_argument(
        "--max_eval_tokens",
        type=int,
        default=256,
        help="Maximum tokens per document for NLL evaluation (default: 256)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens for z-only generation (default: 512)",
    )

    # Device settings
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )

    # Module selection (EVAL-ONLY modules)
    parser.add_argument(
        "--run_verification",
        action="store_true",
        help="Run verification modules (A1, A2, A3)",
    )
    parser.add_argument(
        "--run_ablations",
        action="store_true",
        help="Run ablation modules (B1, B2)",
    )

    # Individual module control
    parser.add_argument("--skip_a1", action="store_true", help="Skip A1: Confusion Matrix")
    parser.add_argument("--skip_a2", action="store_true", help="Skip A2: Z-only Generation")
    parser.add_argument("--skip_a3", action="store_true", help="Skip A3: Z-shuffle")
    parser.add_argument("--skip_b1", action="store_true", help="Skip B1: Alpha Ablation")
    parser.add_argument("--skip_b2", action="store_true", help="Skip B2: Projection Ablation")

    # =========================================================================
    # TRAINING FLAGS (OFF by default - these trigger training if enabled)
    # =========================================================================
    parser.add_argument(
        "--run_proj_only_baseline",
        action="store_true",
        help="[TRAINING] Run projection-only training baseline (OFF by default)",
    )
    parser.add_argument(
        "--sweep_dropout",
        action="store_true",
        help="[TRAINING] Run dropout sweep training variants (OFF by default)",
    )
    parser.add_argument(
        "--sweep_mtokens",
        action="store_true",
        help="[TRAINING] Run m_tokens sweep training variants (OFF by default)",
    )

    # Corpus location
    parser.add_argument(
        "--corpus_path",
        type=str,
        default=None,
        help="Path to corpus.json (default: auto-detect)",
    )

    args = parser.parse_args()

    # Warn if training flags are enabled
    for flag in TRAINING_FLAGS:
        if getattr(args, flag, False):
            print(f"WARNING: Training flag --{flag} is enabled. This will trigger training.")

    return args


def load_model_and_data(args, logger):
    """
    Load Phase 1 model, z_pool, and corpus.

    Returns:
        tuple: (model, z_pool, corpus, tokenizer)
    """
    logger.info("Loading model and data...")

    ckpt_dir = Path(args.ckpt_dir)

    # Import model classes
    from models.write_phase_model import WritePhaseModel, ZPoolManager

    # Load config from checkpoint if available
    config_path = ckpt_dir / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        llm_name = config.get("model", {}).get("llm_name", "Qwen/Qwen3-8B")
        m_tokens = config.get("memory", {}).get("m_tokens", 16)
        z_dim = config.get("memory", {}).get("z_dim", 256)
    else:
        # Default values
        llm_name = "Qwen/Qwen3-8B"
        m_tokens = 16
        z_dim = 256

    logger.info(f"Model: {llm_name}, m_tokens={m_tokens}, z_dim={z_dim}")

    # Initialize model
    model = WritePhaseModel(
        llm_name=llm_name,
        m_tokens=m_tokens,
        z_dim=z_dim,
        quantization="4bit",
    )
    model.to(args.device)
    model.eval()

    # Load projection
    proj_path = ckpt_dir / "projection.pt"
    if proj_path.exists():
        model.load_projection(str(proj_path))
        logger.info(f"Loaded projection from {proj_path}")
        logger.info(f"Alpha value: {model.alpha.item():.4f}")
    else:
        logger.warning(f"Projection not found at {proj_path}")

    # Load z_pool
    z_pool = ZPoolManager(m_tokens=m_tokens, z_dim=z_dim)

    z_pool_path = ckpt_dir / "z_pool.pt"
    if z_pool_path.exists():
        z_pool.load(str(z_pool_path))
        logger.info(f"Loaded z_pool from {z_pool_path} ({len(z_pool.doc_ids)} docs)")
    else:
        # Try epoch checkpoint
        epoch_path = ckpt_dir / "z_pool_epoch50.pt"
        if epoch_path.exists():
            # Load from epoch checkpoint format
            ckpt = torch.load(epoch_path, map_location="cpu")
            if "z_vectors" in ckpt:
                for doc_id, z_tensor in ckpt["z_vectors"].items():
                    z_pool.add(doc_id, z_tensor)
                logger.info(f"Loaded z_pool from {epoch_path} ({len(z_pool.doc_ids)} docs)")
            else:
                logger.error(f"Invalid checkpoint format at {epoch_path}")
        else:
            logger.error(f"No z_pool found in {ckpt_dir}")

    # Load corpus
    corpus = {}
    corpus_paths = [
        args.corpus_path,
        ckpt_dir.parent / "phase2_corpus" / "corpus.json",
        ckpt_dir / "corpus.json",
        PROJECT_ROOT / "checkpoints" / "phase2_corpus" / "corpus.json",
    ]

    for corpus_path in corpus_paths:
        if corpus_path and Path(corpus_path).exists():
            with open(corpus_path, "r", encoding="utf-8") as f:
                corpus = json.load(f)
            logger.info(f"Loaded corpus from {corpus_path} ({len(corpus)} docs)")
            break

    if not corpus:
        logger.warning("Corpus not found. Some reports may be incomplete.")

    # Get tokenizer
    tokenizer = model.tokenizer

    return model, z_pool, corpus, tokenizer


def save_metadata(run_dir: Path, args, start_time: datetime, logger):
    """Save run metadata to 00_meta/."""
    meta_dir = run_dir / "00_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Run manifest
    manifest = {
        "start_time": start_time.isoformat(),
        "args": vars(args),
        "env": get_env_info(),
    }
    save_json(manifest, meta_dir / "run_manifest.json")

    # Run command
    with open(meta_dir / "run_command.txt", "w") as f:
        f.write(" ".join(sys.argv))

    # Effective config
    effective_config = {
        "ckpt_dir": str(args.ckpt_dir),
        "out_root": str(args.out_root),
        "num_docs": args.num_docs,
        "max_eval_tokens": args.max_eval_tokens,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "seed": args.seed,
        "use_amp": not args.no_amp,
        "run_verification": args.run_verification,
        "run_ablations": args.run_ablations,
    }
    save_json(effective_config, meta_dir / "effective_config.json")

    logger.info(f"Saved metadata to {meta_dir}")


def print_final_summary(results: dict, run_dir: Path, logger):
    """Print console summary at end of run."""
    print("\n" + "=" * 70)
    print("PHASE 1 ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults directory: {run_dir}")
    print("\nKey Results:")

    # A1 Confusion
    if "A1" in results:
        a1 = results["A1"]
        print(f"  A1 Confusion Matrix:")
        print(f"    - Top-1 Accuracy: {a1.get('top1_acc', 0)*100:.1f}%")
        print(f"    - Mean Margin: {a1.get('mean_margin', 0):.3f}")

    # A2 Z-only
    if "A2" in results:
        a2 = results["A2"]
        print(f"  A2 Z-only Generation:")
        print(f"    - Mean ROUGE-L: {a2.get('mean_rouge_l', 0):.3f}")

    # A3 Z-shuffle
    if "A3" in results:
        a3 = results["A3"]
        print(f"  A3 Z-shuffle:")
        print(f"    - Delta Top-1: {a3.get('delta_top1', 0)*100:+.1f}%")

    # B1 Alpha
    if "B1" in results:
        b1 = results["B1"]
        print(f"  B1 Alpha Ablation:")
        for name, metrics in b1.items():
            if isinstance(metrics, dict):
                print(f"    - {name}: {metrics.get('top1_acc', 0)*100:.1f}%")

    # B2 Projection
    if "B2" in results:
        b2 = results["B2"]
        print(f"  B2 Projection Ablation:")
        for name, metrics in b2.items():
            if isinstance(metrics, dict):
                print(f"    - {name}: {metrics.get('top1_acc', 0)*100:.1f}%")

    print("\nTop 5 artifacts to check:")
    print("  1. 01_verification/A1_confusion/artifacts/confusion_heatmap.png")
    print("  2. 01_verification/A1_confusion/samples/eyeball_20_worst.md")
    print("  3. 01_verification/A2_zonly/samples/eyeball_20_best.md")
    print("  4. 01_verification/A2_zonly/samples/eyeball_20_worst.md")
    print("  5. 02_ablations/B2_projection/proj_random_frozen/samples/eyeball_10_examples.md")
    print("\nSee 03_summary/dashboard.md for full details.")
    print("=" * 70 + "\n")


def main():
    """Main entry point."""
    args = parse_args()

    # Create or resume run directory
    if args.resume_dir:
        run_dir = Path(args.resume_dir)
        if not run_dir.exists():
            print(f"ERROR: Resume directory does not exist: {run_dir}")
            sys.exit(1)
        timestamp = run_dir.name
        is_resume = True
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.out_root) / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        is_resume = False

    # Setup logging
    logger = setup_logging(run_dir / "00_logs")
    start_time = datetime.now()

    logger.info("=" * 70)
    logger.info("PHASE 1 VERIFICATION & ABLATION RUNNER")
    logger.info("=" * 70)
    if is_resume:
        logger.info(f"RESUMING from: {run_dir}")
    else:
        logger.info(f"Run directory: {run_dir}")
    logger.info(f"Checkpoint: {args.ckpt_dir}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Seed: {args.seed}")

    # =========================================================================
    # CRITICAL CHECK: Ensure no training flags are set by accident
    # =========================================================================
    training_enabled = any(getattr(args, flag, False) for flag in TRAINING_FLAGS)
    if training_enabled:
        logger.warning("!" * 70)
        logger.warning("TRAINING FLAGS ARE ENABLED - This run will include training.")
        logger.warning("!" * 70)
    else:
        logger.info("Mode: EVAL-ONLY (no training)")

    # Set seed for reproducibility
    set_seed(args.seed)

    # Save metadata
    save_metadata(run_dir, args, start_time, logger)

    # Load model and data
    try:
        model, z_pool, corpus, tokenizer = load_model_and_data(args, logger)
    except Exception as e:
        logger.error(f"Failed to load model/data: {e}")
        raise

    # Ensure num_docs doesn't exceed available docs
    args.num_docs = min(args.num_docs, len(z_pool.doc_ids))
    logger.info(f"Evaluating {args.num_docs} documents")

    # Create necessary directories
    (run_dir / "01_verification").mkdir(exist_ok=True)
    (run_dir / "02_ablations").mkdir(exist_ok=True)
    (run_dir / "03_summary").mkdir(exist_ok=True)
    (run_dir / "04_cache").mkdir(exist_ok=True)

    # Run modules
    results = {}
    use_amp = not args.no_amp

    # =========================================================================
    # VERIFICATION MODULES (A1, A2, A3) - EVAL-ONLY
    # =========================================================================
    if args.run_verification:
        logger.info("\n" + "=" * 50)
        logger.info("VERIFICATION MODULES")
        logger.info("=" * 50)

        # A1: Confusion Matrix
        if not args.skip_a1:
            logger.info("\n--- A1: Confusion Matrix ---")
            from experiments.phase1_modules.confusion_matrix import run_confusion_matrix

            with Timer("A1 Total"):
                results["A1"] = run_confusion_matrix(
                    model=model,
                    z_pool=z_pool,
                    corpus=corpus,
                    tokenizer=tokenizer,
                    run_dir=run_dir,
                    num_docs=args.num_docs,
                    max_eval_tokens=args.max_eval_tokens,
                    device=args.device,
                    use_amp=use_amp,
                )

        # A2: Z-only Generation
        if not args.skip_a2:
            logger.info("\n--- A2: Z-only Generation ---")
            from experiments.phase1_modules.zonly_generation import run_zonly_generation

            with Timer("A2 Total"):
                results["A2"] = run_zonly_generation(
                    model=model,
                    z_pool=z_pool,
                    corpus=corpus,
                    run_dir=run_dir,
                    num_docs=args.num_docs,
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed,
                    device=args.device,
                )

        # A3: Z-shuffle
        if not args.skip_a3:
            logger.info("\n--- A3: Z-shuffle Sanity ---")
            from experiments.phase1_modules.zshuffle import run_zshuffle_sanity

            with Timer("A3 Total"):
                results["A3"] = run_zshuffle_sanity(
                    model=model,
                    z_pool=z_pool,
                    corpus=corpus,
                    tokenizer=tokenizer,
                    run_dir=run_dir,
                    baseline_metrics=results.get("A1"),
                    num_docs=args.num_docs,
                    max_eval_tokens=args.max_eval_tokens,
                    device=args.device,
                    use_amp=use_amp,
                    seed=args.seed,
                )

    # =========================================================================
    # ABLATION MODULES (B1, B2) - EVAL-ONLY
    # =========================================================================
    if args.run_ablations:
        logger.info("\n" + "=" * 50)
        logger.info("ABLATION MODULES")
        logger.info("=" * 50)

        # B1: Alpha Ablation
        if not args.skip_b1:
            logger.info("\n--- B1: Alpha Ablation ---")
            from experiments.phase1_modules.alpha_ablation import run_alpha_ablation

            with Timer("B1 Total"):
                results["B1"] = run_alpha_ablation(
                    model=model,
                    z_pool=z_pool,
                    corpus=corpus,
                    tokenizer=tokenizer,
                    run_dir=run_dir,
                    num_docs=args.num_docs,
                    max_eval_tokens=args.max_eval_tokens,
                    device=args.device,
                    use_amp=use_amp,
                )

        # B2: Projection Ablation
        if not args.skip_b2:
            logger.info("\n--- B2: Projection Ablation ---")
            from experiments.phase1_modules.projection_ablation import run_projection_ablation

            with Timer("B2 Total"):
                results["B2"] = run_projection_ablation(
                    model=model,
                    z_pool=z_pool,
                    corpus=corpus,
                    tokenizer=tokenizer,
                    run_dir=run_dir,
                    ckpt_dir=Path(args.ckpt_dir),
                    num_docs=args.num_docs,
                    max_eval_tokens=args.max_eval_tokens,
                    max_new_tokens=args.max_new_tokens,
                    device=args.device,
                    use_amp=use_amp,
                )

    # =========================================================================
    # TRAINING MODULES (C) - OFF BY DEFAULT
    # =========================================================================
    if training_enabled:
        logger.warning("\n" + "=" * 50)
        logger.warning("TRAINING MODULES (EXPLICITLY ENABLED)")
        logger.warning("=" * 50)
        logger.warning("Training modules are not yet implemented.")
        # TODO: Implement training modules if needed

    # =========================================================================
    # GENERATE SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 50)
    logger.info("GENERATING SUMMARY")
    logger.info("=" * 50)

    summary_dir = run_dir / "03_summary"

    # Generate ablation summary CSV/JSON
    generate_ablation_summary(results, summary_dir)

    # Generate dashboard
    generate_dashboard(results, summary_dir, timestamp)

    # Generate README
    generate_readme(summary_dir, vars(args))

    # Update run manifest with end time
    end_time = datetime.now()
    manifest_path = run_dir / "00_meta" / "run_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        manifest["end_time"] = end_time.isoformat()
        manifest["total_runtime_seconds"] = (end_time - start_time).total_seconds()
        save_json(manifest, manifest_path)

    # Print final summary
    print_final_summary(results, run_dir, logger)

    logger.info(f"Total runtime: {(end_time - start_time).total_seconds():.2f}s")
    logger.info("Done!")

    return results


if __name__ == "__main__":
    main()
