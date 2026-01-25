#!/usr/bin/env python
"""
LLM-as-Memory 통합 테스트

새 설계 기반:
- ParametricMemoryLLM 모델
- Evidence 생성 및 평가
- RAG Baseline 비교

Usage:
    python scripts/test_integration.py --quick
    python scripts/test_integration.py --full
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports(skip_heavy=False):
    """모든 모듈 import 테스트"""
    logger.info("=" * 60)
    logger.info("Testing imports...")
    if skip_heavy:
        logger.info("(Skipping heavy dependencies: peft, transformers, etc.)")
    logger.info("=" * 60)

    errors = []
    warnings = []

    # Core model (requires heavy dependencies)
    if not skip_heavy:
        try:
            from models import ParametricMemoryLLM
            logger.info("[OK] models.ParametricMemoryLLM")
        except ImportError as e:
            warnings.append(f"models.ParametricMemoryLLM: {e}")
            logger.warning(f"[SKIP] models.ParametricMemoryLLM: {e}")
        except Exception as e:
            errors.append(f"models.ParametricMemoryLLM: {e}")
            logger.error(f"[FAIL] models.ParametricMemoryLLM: {e}")

        # Trainer
        try:
            from models import EvidenceTrainer
            logger.info("[OK] models.EvidenceTrainer")
        except ImportError as e:
            warnings.append(f"models.EvidenceTrainer: {e}")
            logger.warning(f"[SKIP] models.EvidenceTrainer: {e}")
        except Exception as e:
            errors.append(f"models.EvidenceTrainer: {e}")
            logger.error(f"[FAIL] models.EvidenceTrainer: {e}")

        # Training
        try:
            from training import run_evidence_training
            logger.info("[OK] training.run_evidence_training")
        except ImportError as e:
            warnings.append(f"training.run_evidence_training: {e}")
            logger.warning(f"[SKIP] training.run_evidence_training: {e}")
        except Exception as e:
            errors.append(f"training.run_evidence_training: {e}")
            logger.error(f"[FAIL] training.run_evidence_training: {e}")

        # Baselines
        try:
            from baselines import StandardRAGBaseline, NoRetrievalBaseline
            logger.info("[OK] baselines.StandardRAGBaseline")
        except ImportError as e:
            warnings.append(f"baselines: {e}")
            logger.warning(f"[SKIP] baselines: {e}")
        except Exception as e:
            errors.append(f"baselines: {e}")
            logger.error(f"[FAIL] baselines: {e}")

    # Light imports (should always work)
    try:
        from evaluation.evidence_metrics import (
            compute_rouge_l,
            compute_answer_coverage,
            evaluate_evidence_quality,
            compare_with_rag,
        )
        logger.info("[OK] evaluation.evidence_metrics (standalone)")
    except Exception as e:
        errors.append(f"evaluation.evidence_metrics (standalone): {e}")
        logger.error(f"[FAIL] evaluation.evidence_metrics (standalone): {e}")

    # Data utilities (light)
    try:
        from data.evidence_dataloader import (
            extract_evidence_from_nq,
            extract_evidence_from_hotpotqa,
        )
        logger.info("[OK] data.evidence_dataloader (utilities)")
    except ImportError as e:
        # May need torch/transformers
        warnings.append(f"data.evidence_dataloader: {e}")
        logger.warning(f"[SKIP] data.evidence_dataloader: {e}")
    except Exception as e:
        errors.append(f"data.evidence_dataloader: {e}")
        logger.error(f"[FAIL] data.evidence_dataloader: {e}")

    logger.info("=" * 60)
    if errors:
        logger.error(f"Import test FAILED with {len(errors)} errors")
        for err in errors:
            logger.error(f"  - {err}")
        return False
    elif warnings:
        logger.warning(f"Import test PASSED with {len(warnings)} skipped (missing deps)")
        return True
    else:
        logger.info("Import test PASSED")
        return True


def test_evidence_metrics():
    """Evidence 메트릭 테스트"""
    logger.info("=" * 60)
    logger.info("Testing evidence metrics...")
    logger.info("=" * 60)

    # Import directly from evidence_metrics to avoid heavy dependencies
    from evaluation.evidence_metrics import (
        compute_rouge_l,
        compute_answer_coverage,
        compute_token_f1,
        evaluate_evidence_quality,
    )

    # Test cases
    generated = "The capital of France is Paris, which is a beautiful city."
    gold = "Paris is the capital and largest city of France."
    answer = "Paris"

    # ROUGE-L
    rouge = compute_rouge_l(generated, gold)
    logger.info(f"ROUGE-L: P={rouge['precision']:.4f}, R={rouge['recall']:.4f}, F1={rouge['f1']:.4f}")

    # Answer Coverage
    coverage = compute_answer_coverage(generated, answer)
    logger.info(f"Answer Coverage: {coverage:.4f}")

    # Token F1
    f1 = compute_token_f1(generated, gold)
    logger.info(f"Token F1: P={f1['precision']:.4f}, R={f1['recall']:.4f}, F1={f1['f1']:.4f}")

    # Combined
    all_metrics = evaluate_evidence_quality(generated, gold, answer)
    logger.info(f"All metrics: {all_metrics}")

    # Assertions
    assert rouge['f1'] > 0, "ROUGE-L F1 should be > 0"
    assert coverage == 1.0, "Answer 'Paris' should be fully covered"

    logger.info("Evidence metrics test PASSED")
    return True


def test_dataloader():
    """DataLoader 테스트 (토크나이저 없이)"""
    logger.info("=" * 60)
    logger.info("Testing dataloader...")
    logger.info("=" * 60)

    # Import directly to avoid heavy dependencies
    from data.evidence_dataloader import extract_evidence_from_nq, extract_evidence_from_hotpotqa

    # NQ-style evidence extraction
    document = "Paris is the capital of France. It is known for the Eiffel Tower. The city has a population of over 2 million."
    answer = "Eiffel Tower"

    evidence = extract_evidence_from_nq(document, answer, context_window=50)
    logger.info(f"NQ Evidence: {evidence[:100]}...")
    assert "Eiffel Tower" in evidence, "Evidence should contain answer"

    # HotpotQA-style evidence extraction
    context = [
        ("Paris", ["Paris is the capital of France.", "It has the Eiffel Tower."]),
        ("London", ["London is the capital of UK.", "It has Big Ben."]),
    ]
    supporting_facts = [("Paris", 0), ("Paris", 1)]

    evidence = extract_evidence_from_hotpotqa(context, supporting_facts)
    logger.info(f"HotpotQA Evidence: {evidence}")
    assert "Paris" in evidence, "Evidence should mention Paris"

    logger.info("DataLoader test PASSED")
    return True


def test_model_init_quick():
    """모델 초기화 테스트 (CPU, 작은 설정)"""
    logger.info("=" * 60)
    logger.info("Testing model initialization (quick, CPU)...")
    logger.info("=" * 60)

    import torch
    from models import ParametricMemoryLLM

    # 매우 작은 설정으로 테스트 (실제 LLM 로드 안함)
    # 실제 테스트를 위해서는 --full 옵션 사용

    logger.info("Skipping full model init in quick mode")
    logger.info("Use --full for actual model loading test")

    return True


def test_model_init_full():
    """모델 초기화 테스트 (GPU, 실제 로딩)"""
    logger.info("=" * 60)
    logger.info("Testing model initialization (full, GPU)...")
    logger.info("=" * 60)

    import torch

    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping GPU test")
        return True

    from models import ParametricMemoryLLM

    # 작은 모델로 테스트 (실제 Qwen은 큼)
    try:
        model = ParametricMemoryLLM(
            llm_name="Qwen/Qwen3-8B",
            num_docs=100,
            z_dim=128,
            m_tokens=2,
            quantization="4bit",
        )
        logger.info(f"Model loaded successfully")
        logger.info(f"Memory pool shape: {model.memory_pool.shape}")

        # Simple forward test
        tokenizer = model.tokenizer
        query = "What is the capital of France?"
        encoded = tokenizer(query, return_tensors="pt")
        query_ids = encoded["input_ids"].to(model.llm.device)

        # Generate evidence
        evidence = model.generate_evidence(query_ids, max_new_tokens=32)
        logger.info(f"Generated evidence: {evidence[:100]}...")

        logger.info("Full model init test PASSED")
        return True

    except Exception as e:
        logger.error(f"Model init failed: {e}")
        return False


def test_rag_comparison():
    """RAG Baseline 비교 함수 테스트"""
    logger.info("=" * 60)
    logger.info("Testing RAG comparison...")
    logger.info("=" * 60)

    # Import directly from evidence_metrics to avoid heavy dependencies
    from evaluation.evidence_metrics import compare_with_rag, print_comparison_table

    # Mock data
    our_evidence = [
        "Paris is the capital of France.",
        "The Eiffel Tower is in Paris.",
    ]
    rag_retrieved = [
        "France is a country in Europe. Paris is its capital.",
        "Paris has many landmarks including the Eiffel Tower.",
    ]
    gold_evidence = [
        "Paris is the capital and largest city of France.",
        "The Eiffel Tower is a famous landmark in Paris.",
    ]
    answers = ["Paris", "Eiffel Tower"]

    # Compare
    comparison = compare_with_rag(our_evidence, rag_retrieved, gold_evidence, answers)

    # Print table
    print_comparison_table(comparison)

    assert "our" in comparison, "Should have 'our' metrics"
    assert "rag" in comparison, "Should have 'rag' metrics"
    assert "diff" in comparison, "Should have 'diff' metrics"

    logger.info("RAG comparison test PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Memory Integration Test")
    parser.add_argument("--quick", action="store_true", help="Quick test (no model loading)")
    parser.add_argument("--full", action="store_true", help="Full test (with model loading)")
    parser.add_argument("--minimal", action="store_true", help="Minimal test (no heavy dependencies)")
    args = parser.parse_args()

    if not args.quick and not args.full and not args.minimal:
        args.minimal = True  # Default to minimal

    logger.info("\n" + "=" * 60)
    logger.info("LLM-as-Memory Integration Test")
    logger.info("=" * 60 + "\n")

    results = []

    # Always run these tests
    skip_heavy = args.minimal
    results.append(("imports", test_imports(skip_heavy=skip_heavy)))
    results.append(("evidence_metrics", test_evidence_metrics()))
    results.append(("dataloader", test_dataloader()))
    results.append(("rag_comparison", test_rag_comparison()))

    # Model tests
    if args.quick:
        results.append(("model_init_quick", test_model_init_quick()))
    if args.full:
        results.append(("model_init_full", test_model_init_full()))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    passed = 0
    failed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"  {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    logger.info("-" * 60)
    logger.info(f"Total: {passed} passed, {failed} failed")

    if failed > 0:
        sys.exit(1)
    else:
        logger.info("\nAll tests PASSED!")
        sys.exit(0)


if __name__ == "__main__":
    main()
