#!/usr/bin/env python3
"""
Phase 1.5: Evidence Generation Training Runner

Trains LoRA adapters on frozen Phase 1 z_pool + projection to generate
extractive evidence from (z_i, query) pairs.

Goal: Align the model to extract relevant evidence spans from documents
      encoded in z_i vectors, preparing for full Phase 2 evidence generation.

Pipeline:
    1. Build evidence dataset from QA pairs (answer_span / sentence_ranker)
    2. Train LoRA adapters (z and projection FROZEN)
    3. Evaluate evidence quality (answer_coverage, source_overlap, ROUGE-L)
    4. Run Phase 1 regression tests (verify z storage not degraded)

Usage:
    # Full run: dataset + train + eval + regression
    python experiments/phase1_5_runner.py \\
        --phase1_ckpt checkpoints/phase1_v2 \\
        --corpus_dir checkpoints/phase2_corpus \\
        --out_root results/phase1_5 \\
        --epochs 10

    # Eval-only (with pre-trained checkpoint)
    python experiments/phase1_5_runner.py \\
        --phase1_ckpt checkpoints/phase1_v2 \\
        --phase15_ckpt results/phase1_5/20260128_123456/02_train/checkpoints/best.pt_lora \\
        --eval_only

    # Smoke test (quick validation)
    python experiments/phase1_5_runner.py \\
        --phase1_ckpt checkpoints/phase1_v2 \\
        --smoke_test

Author: zRAG Team
"""

import argparse
import json
import os
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 1.5 Evidence Generation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full training + evaluation
  python experiments/phase1_5_runner.py --phase1_ckpt checkpoints/phase1_v2

  # Smoke test
  python experiments/phase1_5_runner.py --phase1_ckpt checkpoints/phase1_v2 --smoke_test

  # Eval-only
  python experiments/phase1_5_runner.py --phase1_ckpt checkpoints/phase1_v2 --phase15_ckpt <path> --eval_only
        """,
    )

    # Input paths
    parser.add_argument(
        "--phase1_ckpt",
        type=str,
        default="checkpoints/phase1_v2",
        help="Phase 1 checkpoint directory (z_pool.pt, projection.pt)",
    )
    parser.add_argument(
        "--corpus_dir",
        type=str,
        default="checkpoints/phase2_corpus",
        help="Phase 2 corpus directory (corpus.json, qa_val.json)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "train"],
        help="QA split to use",
    )

    # Output settings
    parser.add_argument(
        "--out_root",
        type=str,
        default="results/phase1_5",
        help="Root directory for output",
    )
    parser.add_argument(
        "--resume_dir",
        type=str,
        default=None,
        help="Resume from existing run directory",
    )

    # Pre-trained checkpoint (for eval-only)
    parser.add_argument(
        "--phase15_ckpt",
        type=str,
        default=None,
        help="Pre-trained Phase 1.5 LoRA checkpoint (for eval-only)",
    )

    # Phase 1 baseline for regression
    parser.add_argument(
        "--phase1_baseline_run_dir",
        type=str,
        default=None,
        help="Phase 1 analysis run directory for regression baseline",
    )

    # Mode flags
    parser.add_argument(
        "--skip_dataset",
        action="store_true",
        help="Skip dataset building",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation",
    )
    parser.add_argument(
        "--skip_regression",
        action="store_true",
        help="Skip Phase 1 regression tests",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip dataset and training, only run evaluation",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Quick smoke test (small dataset, 1 epoch)",
    )
    parser.add_argument(
        "--overfit_test",
        action="store_true",
        help="1-sample overfit test (verify pipeline can memorize single sample)",
    )
    parser.add_argument(
        "--overfit_steps",
        type=int,
        default=200,
        help="Number of steps for overfit test (default: 200)",
    )

    # Dataset settings
    parser.add_argument(
        "--evidence_method",
        type=str,
        default="answer_span",
        choices=["answer_span", "sentence_ranker"],
        help="Primary evidence extraction method",
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
        "--drop_fallback",
        action="store_true",
        help="Exclude samples where primary evidence extraction failed. "
             "Produces cleaner dataset with higher answer coverage.",
    )
    parser.add_argument(
        "--add_end_marker",
        action="store_true",
        help="DEPRECATED: Causes LoRA collapse (END END END repetition). Use --use_eos_token instead.",
    )
    parser.add_argument(
        "--use_eos_token",
        action="store_true",
        help="RECOMMENDED: Append EOS token to evidence. Single token enables clean generation stopping.",
    )
    parser.add_argument(
        "--drop_multi_entity_incomplete",
        action="store_true",
        help="Exclude samples where evidence doesn't cover all entities in the question. "
             "Prevents hallucination of missing entity info.",
    )

    # Training settings
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr_lora",
        type=float,
        default=2e-5,
        help="LoRA learning rate",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=32,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="LoRA alpha",
    )

    # Optional flags
    parser.add_argument(
        "--tune_projection",
        action="store_true",
        help="Allow projection fine-tuning (default: frozen)",
    )
    parser.add_argument(
        "--unfreeze_z",
        action="store_true",
        help="Allow z training (default: frozen, NOT RECOMMENDED)",
    )

    # Evaluation settings
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max tokens for evidence generation",
    )
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=None,
        help="Limit evaluation samples (None = all)",
    )

    # Regression settings
    parser.add_argument(
        "--regression_num_docs",
        type=int,
        default=50,
        help="Number of docs for regression tests",
    )
    parser.add_argument(
        "--regression_threshold",
        type=float,
        default=0.02,
        help="Max allowed Top-1 accuracy drop (default: 2%%)",
    )

    # Device settings
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )

    args = parser.parse_args()

    # Handle smoke_test
    if args.smoke_test:
        args.epochs = 1
        args.num_eval_samples = 10
        args.regression_num_docs = 20

    # Handle overfit_test (1-sample pipeline verification)
    if args.overfit_test:
        args.skip_eval = True       # Normal eval not useful
        args.skip_regression = True  # Not relevant for overfit

    # Handle eval_only
    if args.eval_only:
        args.skip_dataset = True
        args.skip_training = True

    # Warning for dangerous flags
    if args.unfreeze_z:
        print("!" * 60)
        print("WARNING: --unfreeze_z is enabled!")
        print("This may degrade Phase 1 storage property.")
        print("Regression tests will be mandatory.")
        print("!" * 60)
        args.skip_regression = False

    return args


def load_phase1_model_and_zpool(args, logger):
    """Load Phase 1 model and z_pool."""
    from models.write_phase_model import WritePhaseModel, ZPoolManager

    ckpt_dir = Path(args.phase1_ckpt)

    # Load config if available
    config_path = ckpt_dir / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        llm_name = config.get("model", {}).get("llm_name", "Qwen/Qwen3-8B")
        m_tokens = config.get("memory", {}).get("m_tokens", 16)
        z_dim = config.get("memory", {}).get("z_dim", 256)
    else:
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

    # Load projection
    proj_path = ckpt_dir / "projection.pt"
    if proj_path.exists():
        model.load_projection(str(proj_path))
        logger.info(f"Loaded projection from {proj_path}")
        logger.info(f"Alpha: {model.alpha.item():.4f}")

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
            ckpt = torch.load(epoch_path, map_location="cpu")
            if "z_vectors" in ckpt:
                for doc_id, z_tensor in ckpt["z_vectors"].items():
                    z_pool.add(doc_id, z_tensor)
                logger.info(f"Loaded z_pool from {epoch_path} ({len(z_pool.doc_ids)} docs)")

    return model, z_pool


def run_overfit_verification(model, z_pool, tokenizer, dataset_path, max_new_tokens, device, logger):
    """
    Verify that the model can memorize a single sample.

    This is a sanity check for the training pipeline:
    - If the model CAN'T memorize 1 sample after 200 steps, there's a bug
    - If it CAN, then low metrics are due to data/difficulty, not implementation

    Returns:
        Dict with verification results
    """
    from experiments.phase1_5_modules.model_wrapper import Phase15ForwardWrapper

    # Load the single sample from dataset
    with open(dataset_path, "r") as f:
        sample = json.loads(f.readline())

    doc_id = sample.get("doc_id")
    query = sample["question"]
    target_evidence = sample["evidence_text"]
    answer = sample["answer"]

    if not doc_id or doc_id not in z_pool.doc_ids:
        logger.error(f"doc_id {doc_id} not in z_pool")
        return {"status": "error", "message": "doc_id not found"}

    logger.info(f"Sample doc_id: {doc_id}")
    logger.info(f"Query: {query[:80]}...")
    logger.info(f"Target evidence ({len(target_evidence)} chars): {target_evidence[:100]}...")
    logger.info(f"Answer: {answer}")

    # Generate with trained model
    wrapper = Phase15ForwardWrapper(model, tokenizer, device)
    model.eval()

    with torch.no_grad():
        z = z_pool.get_z(doc_id).to(device)
        generated = wrapper.generate_evidence(z, query, max_new_tokens=max_new_tokens)

    logger.info(f"\nGenerated ({len(generated)} chars): {generated[:200]}...")

    # Compute metrics
    from difflib import SequenceMatcher

    # 1. Answer in generated (most important - did we get the key info?)
    ans_in_gen = answer.lower() in generated.lower()

    # 2. Prefix match - does generated START with target? (handles over-generation)
    target_clean = target_evidence.strip().lower()
    generated_clean = generated.strip().lower()

    # Check if generated starts with first N chars of target (allowing minor variations)
    prefix_len = min(50, len(target_clean))
    prefix_similarity = SequenceMatcher(
        None,
        generated_clean[:prefix_len],
        target_clean[:prefix_len]
    ).ratio()
    prefix_match = prefix_similarity > 0.8

    # 3. Truncated comparison - truncate generated to target length for fair comparison
    generated_truncated = generated[:len(target_evidence)]

    # Token overlap on truncated
    gen_tokens = set(generated_truncated.lower().split())
    target_tokens = set(target_evidence.lower().split())
    if gen_tokens:
        token_overlap_truncated = len(gen_tokens & target_tokens) / len(gen_tokens)
    else:
        token_overlap_truncated = 0.0

    # Char similarity on truncated
    char_similarity_truncated = SequenceMatcher(
        None,
        generated_truncated.lower(),
        target_evidence.lower()
    ).ratio()

    # 4. Full comparison (for reference)
    gen_tokens_full = set(generated.lower().split())
    if gen_tokens_full:
        token_overlap_full = len(gen_tokens_full & target_tokens) / len(gen_tokens_full)
    else:
        token_overlap_full = 0.0
    char_similarity_full = SequenceMatcher(None, generated.lower(), target_evidence.lower()).ratio()

    # 5. Exact match (very strict - rarely expected)
    exact_match = generated.strip() == target_evidence.strip()

    result = {
        "status": "success",
        "doc_id": doc_id,
        "query": query,
        "target_len": len(target_evidence),
        "generated_len": len(generated),
        "exact_match": exact_match,
        "answer_in_generated": ans_in_gen,
        "prefix_match": prefix_match,
        "prefix_similarity": prefix_similarity,
        # Truncated metrics (fair comparison)
        "token_overlap_truncated": token_overlap_truncated,
        "char_similarity_truncated": char_similarity_truncated,
        # Full metrics (for reference)
        "token_overlap_full": token_overlap_full,
        "char_similarity_full": char_similarity_full,
        "target_snippet": target_evidence[:200],
        "generated_snippet": generated[:200],
    }

    # Log verdict
    logger.info("\n" + "=" * 50)
    logger.info("OVERFIT TEST RESULTS")
    logger.info("=" * 50)
    logger.info(f"  Answer in generated: {ans_in_gen}")
    logger.info(f"  Prefix match (first 50 chars): {prefix_match} ({prefix_similarity*100:.1f}%)")
    logger.info(f"  --- Truncated comparison (gen[:target_len]) ---")
    logger.info(f"  Token overlap (truncated): {token_overlap_truncated*100:.1f}%")
    logger.info(f"  Char similarity (truncated): {char_similarity_truncated*100:.1f}%")
    logger.info(f"  --- Full comparison (reference) ---")
    logger.info(f"  Token overlap (full): {token_overlap_full*100:.1f}%")
    logger.info(f"  Char similarity (full): {char_similarity_full*100:.1f}%")
    logger.info(f"  Exact match: {exact_match}")

    # Verdict based on truncated metrics (fair comparison for generative models)
    # Primary: prefix_match + answer_in_gen → pipeline is working
    # Secondary: char_similarity_truncated → degree of memorization
    if prefix_match and ans_in_gen:
        logger.info("VERDICT: PASS - Generated starts correctly and contains answer (pipeline OK)")
        result["verdict"] = "PASS"
    elif char_similarity_truncated > 0.7:
        logger.info("VERDICT: PASS - High similarity when truncated (pipeline OK)")
        result["verdict"] = "PASS"
    elif ans_in_gen and char_similarity_truncated > 0.4:
        logger.warning("VERDICT: PARTIAL - Contains answer but incomplete memorization")
        result["verdict"] = "PARTIAL"
    else:
        logger.error("VERDICT: FAIL - Cannot memorize sample (check pipeline)")
        result["verdict"] = "FAIL"

    logger.info("=" * 50)

    # Save result
    save_json(result, Path(dataset_path).parent.parent / "02_train" / "overfit_test_result.json")

    return result


def create_run_directory(args) -> Path:
    """Create timestamped run directory with folder structure."""
    if args.resume_dir:
        run_dir = Path(args.resume_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {run_dir}")
        return run_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_root) / timestamp

    # Create folder structure
    folders = [
        "00_meta",
        "00_logs",
        "01_data",
        "02_train/checkpoints",
        "02_train/artifacts",
        "03_eval/samples",
        "03_eval/artifacts",
        "04_regression_phase1/artifacts",
        "05_cache",
    ]

    for folder in folders:
        (run_dir / folder).mkdir(parents=True, exist_ok=True)

    return run_dir


def save_metadata(run_dir: Path, args, start_time: datetime, logger):
    """Save run metadata."""
    meta_dir = run_dir / "00_meta"

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
    config = {
        "phase1_ckpt": args.phase1_ckpt,
        "corpus_dir": args.corpus_dir,
        "split": args.split,
        "evidence_method": args.evidence_method,
        "max_evidence_tokens": args.max_evidence_tokens,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr_lora": args.lr_lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "tune_projection": args.tune_projection,
        "unfreeze_z": args.unfreeze_z,
        "device": args.device,
        "seed": args.seed,
        "use_amp": not args.no_amp,
    }
    save_json(config, meta_dir / "effective_config.json")

    logger.info(f"Saved metadata to {meta_dir}")


def print_summary(results: dict, run_dir: Path, logger):
    """Print final summary."""
    print("\n" + "=" * 70)
    print("PHASE 1.5 COMPLETE")
    print("=" * 70)
    print(f"\nResults directory: {run_dir}")

    if "dataset" in results:
        ds = results["dataset"]
        print(f"\nDataset:")
        print(f"  Total samples: {ds.get('total', 0)}")
        print(f"  Success rate: {ds.get('success_rate', 0)*100:.1f}%")

    if "training" in results:
        tr = results["training"]
        print(f"\nTraining:")
        print(f"  Final loss: {tr.get('final_loss', 0):.4f}")
        print(f"  Best loss: {tr.get('best_loss', 0):.4f}")

    if "evaluation" in results:
        ev = results["evaluation"]
        print(f"\nEvaluation:")
        print(f"  Answer Coverage: {ev.get('answer_coverage', 0)*100:.1f}%")
        print(f"  Source Overlap: {ev.get('source_overlap', 0)*100:.1f}%")
        print(f"  ROUGE-L: {ev.get('rouge_l', 0):.3f}")

        if ev.get("warnings"):
            print(f"  Warnings: {', '.join(ev['warnings'])}")

    if "regression" in results:
        reg = results["regression"]
        status = "PASS" if reg.get("all_pass", True) else "FAIL"
        print(f"\nRegression Tests: {status}")
        if "a1_delta" in reg:
            print(f"  A1 Top-1 drop: {reg['a1_delta']['top1_drop']*100:+.1f}%")

    if "overfit_test" in results:
        ot = results["overfit_test"]
        print(f"\nOverfit Test: {ot.get('verdict', 'N/A')}")
        print(f"  Prefix match: {ot.get('prefix_match', False)} ({ot.get('prefix_similarity', 0)*100:.1f}%)")
        print(f"  Char similarity (truncated): {ot.get('char_similarity_truncated', 0)*100:.1f}%")
        print(f"  Token overlap (truncated): {ot.get('token_overlap_truncated', 0)*100:.1f}%")
        print(f"  Answer in generated: {ot.get('answer_in_generated', False)}")

    print("\nKey artifacts:")
    print("  1. 02_train/checkpoints/best.pt_lora/")
    print("  2. 03_eval/samples/eyeball_20_best.md")
    print("  3. 03_eval/samples/eyeball_20_worst.md")
    print("  4. 03_eval/samples/failure_cases.md")
    print("  5. 04_regression_phase1/delta.json")
    print("=" * 70 + "\n")


def main():
    """Main entry point."""
    args = parse_args()

    # Create run directory
    run_dir = create_run_directory(args)

    # Setup logging
    logger = setup_logging(run_dir / "00_logs", logger_name="phase1_5_runner")
    start_time = datetime.now()

    logger.info("=" * 70)
    logger.info("PHASE 1.5: EVIDENCE GENERATION TRAINING")
    logger.info("=" * 70)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Phase 1 checkpoint: {args.phase1_ckpt}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Seed: {args.seed}")

    if args.overfit_test:
        logger.info("MODE: OVERFIT TEST (1-sample pipeline verification)")
        logger.info(f"  Steps: {args.overfit_steps}")
    elif args.smoke_test:
        logger.info("MODE: SMOKE TEST")
    elif args.eval_only:
        logger.info("MODE: EVAL ONLY")
    else:
        logger.info("MODE: FULL TRAINING")

    # Set seed
    set_seed(args.seed)

    # Save metadata
    save_metadata(run_dir, args, start_time, logger)

    # Load corpus
    corpus_path = Path(args.corpus_dir) / "corpus.json"
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    logger.info(f"Loaded corpus: {len(corpus)} documents")

    # Load model and z_pool
    model, z_pool = load_phase1_model_and_zpool(args, logger)
    tokenizer = model.tokenizer

    results = {}

    # =========================================================================
    # DELIVERABLE 1: Build Dataset
    # =========================================================================
    dataset_path = run_dir / "01_data" / "dataset.jsonl"

    if not args.skip_dataset:
        logger.info("\n" + "=" * 50)
        logger.info("DELIVERABLE 1: BUILD EVIDENCE DATASET")
        logger.info("=" * 50)

        from data.build_phase1_5_evidence_dataset import build_evidence_dataset

        # Warn if deprecated option used
        if args.add_end_marker and not args.use_eos_token:
            logger.warning("=" * 60)
            logger.warning("WARNING: --add_end_marker is DEPRECATED")
            logger.warning("This causes LoRA collapse (model outputs 'END END END...')")
            logger.warning("Use --use_eos_token instead for clean single-token stopping.")
            logger.warning("=" * 60)

        with Timer("Dataset Build"):
            dataset_stats = build_evidence_dataset(
                corpus_dir=args.corpus_dir,
                output_dir=str(run_dir / "01_data"),
                split=args.split,
                primary_method=args.evidence_method,
                max_evidence_tokens=args.max_evidence_tokens,
                context_sentences=args.context_sentences,
                tokenizer_name="Qwen/Qwen3-8B",
                seed=args.seed,
                drop_fallback=args.drop_fallback,
                add_end_marker=args.add_end_marker,
                use_eos_token=args.use_eos_token,
                drop_multi_entity_incomplete=args.drop_multi_entity_incomplete,
            )
            results["dataset"] = dataset_stats

    # =========================================================================
    # DELIVERABLE 2: Training
    # =========================================================================
    if not args.skip_training:
        logger.info("\n" + "=" * 50)
        logger.info("DELIVERABLE 2: LORA TRAINING")
        logger.info("=" * 50)

        from experiments.phase1_5_modules.trainer import run_phase15_training

        train_config = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr_lora": args.lr_lora,
            "lora": {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "dropout": 0.05,
            },
            "max_query_length": 128,
            "max_evidence_length": args.max_evidence_tokens,
            "use_amp": not args.no_amp,
            "max_grad_norm": 1.0,
            "weight_decay": 0.01,
            "device": args.device,
            "tune_projection": args.tune_projection,
        }

        if args.overfit_test:
            max_samples = 1
            # For overfit test, we want many iterations over 1 sample
            train_config["epochs"] = args.overfit_steps
        elif args.smoke_test:
            max_samples = 20
        else:
            max_samples = None

        with Timer("LoRA Training"):
            train_results = run_phase15_training(
                model=model,
                z_pool=z_pool,
                tokenizer=tokenizer,
                dataset_path=dataset_path,
                run_dir=run_dir,
                config=train_config,
                max_samples=max_samples,
            )
            results["training"] = train_results

        # For overfit test: generate from the same sample and compare
        if args.overfit_test:
            logger.info("\n" + "=" * 50)
            logger.info("OVERFIT TEST: Checking if model memorized the sample")
            logger.info("=" * 50)
            results["overfit_test"] = run_overfit_verification(
                model, z_pool, tokenizer, dataset_path, args.max_new_tokens, args.device, logger
            )

    # Load LoRA checkpoint for evaluation
    # Priority: 1) --phase15_ckpt (explicit), 2) current run's best.pt_lora (after training or resume)
    lora_ckpt_path = None

    if args.phase15_ckpt:
        lora_ckpt_path = Path(args.phase15_ckpt)
    else:
        # Auto-load from current run directory (works for both just-completed training and resume)
        auto_lora_path = run_dir / "02_train" / "checkpoints" / "best.pt_lora"
        if auto_lora_path.exists():
            lora_ckpt_path = auto_lora_path
            if args.skip_training:
                logger.info(f"Auto-detected LoRA checkpoint from resume_dir: {lora_ckpt_path}")
            else:
                logger.info(f"Auto-loading LoRA from just-completed training: {lora_ckpt_path}")

    if lora_ckpt_path:
        if not lora_ckpt_path.exists():
            raise FileNotFoundError(f"LoRA checkpoint not found: {lora_ckpt_path}")

        # Check if model already has LoRA (from just-completed training)
        from peft import PeftModel
        if isinstance(model.llm, PeftModel):
            # Training just completed - LoRA already in memory, no need to reload
            logger.info("LoRA already attached from training - skipping checkpoint reload")
            logger.info(f"Active LoRA adapters: {model.llm.active_adapters}")
        else:
            # Need to load LoRA checkpoint (skip_training or eval_only mode)
            logger.info(f"Loading Phase 1.5 LoRA checkpoint: {lora_ckpt_path}")
            model.llm = PeftModel.from_pretrained(model.llm, str(lora_ckpt_path))
            logger.info("LoRA checkpoint loaded successfully")

            # Verify LoRA is active
            if hasattr(model.llm, 'active_adapters'):
                logger.info(f"Active LoRA adapters: {model.llm.active_adapters}")
    elif not args.skip_eval:
        logger.warning("=" * 50)
        logger.warning("WARNING: No LoRA checkpoint loaded!")
        logger.warning("Evaluation will use base model without LoRA fine-tuning.")
        logger.warning("Results may not reflect trained model performance.")
        logger.warning("=" * 50)

    # =========================================================================
    # DELIVERABLE 3: Evaluation
    # =========================================================================
    if not args.skip_eval:
        logger.info("\n" + "=" * 50)
        logger.info("DELIVERABLE 3: EVALUATION")
        logger.info("=" * 50)

        from experiments.phase1_5_modules.evaluator import run_phase15_evaluation

        with Timer("Evaluation"):
            eval_results = run_phase15_evaluation(
                model=model,
                z_pool=z_pool,
                corpus=corpus,
                tokenizer=tokenizer,
                dataset_path=dataset_path,
                run_dir=run_dir,
                max_new_tokens=args.max_new_tokens,
                num_samples=args.num_eval_samples,
                device=args.device,
            )
            results["evaluation"] = eval_results

    # =========================================================================
    # DELIVERABLE 4: Phase 1 Regression
    # =========================================================================
    if not args.skip_regression:
        logger.info("\n" + "=" * 50)
        logger.info("DELIVERABLE 4: PHASE 1 REGRESSION")
        logger.info("=" * 50)

        from experiments.phase1_5_modules.regression import run_phase1_regression

        with Timer("Regression Tests"):
            regression_results = run_phase1_regression(
                model=model,
                z_pool=z_pool,
                corpus=corpus,
                tokenizer=tokenizer,
                run_dir=run_dir,
                phase1_baseline_run_dir=Path(args.phase1_baseline_run_dir) if args.phase1_baseline_run_dir else None,
                num_docs=args.regression_num_docs,
                max_eval_tokens=128,
                device=args.device,
                use_amp=not args.no_amp,
                seed=args.seed,
                regression_threshold=args.regression_threshold,
            )
            results["regression"] = regression_results

            if not regression_results.get("all_pass", True):
                logger.warning("!" * 50)
                logger.warning("REGRESSION TESTS FAILED!")
                logger.warning("Phase 1 storage property may be degraded.")
                logger.warning("!" * 50)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    end_time = datetime.now()

    # Update manifest
    manifest_path = run_dir / "00_meta" / "run_manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    manifest["end_time"] = end_time.isoformat()
    manifest["total_runtime_seconds"] = (end_time - start_time).total_seconds()
    manifest["results_summary"] = {
        k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict)) or len(str(vv)) < 100}
        for k, v in results.items()
    }
    save_json(manifest, manifest_path)

    # Print summary
    print_summary(results, run_dir, logger)

    logger.info(f"Total runtime: {(end_time - start_time).total_seconds():.2f}s")
    logger.info("Done!")

    return results


if __name__ == "__main__":
    main()
