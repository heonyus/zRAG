"""
Error Analysis 모듈
- Selection 실패 분석
- Generation 실패 분석
- Multi-hop 실패 패턴 분류
"""

import torch
from typing import List, Optional
from collections import Counter
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from evaluation.metrics import compute_em, compute_f1, compute_recall_at_k

logger = logging.getLogger(__name__)


def analyze_errors(
    model,
    qa_pairs: list,
    corpus: dict,
    tokenizer,
    top_k: int = 5,
    max_samples: int = 200,
    device: str = "cuda",
) -> dict:
    """
    종합 에러 분석

    Returns:
        dict with error categories, examples, statistics
    """
    model.eval()

    error_categories = {
        "selection_failure": [],    # gold doc not in top-k
        "generation_failure": [],   # correct selection but wrong answer
        "both_failure": [],         # wrong selection AND wrong answer
        "success": [],              # correct answer
    }

    all_predictions = []

    for item in qa_pairs[:max_samples]:
        question = item["question"]
        answer = item["answer"]
        gold_doc_ids = item["gold_doc_ids"]

        # Tokenize
        q_encoded = tokenizer(
            question, max_length=128, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        query_ids = q_encoded["input_ids"].to(device)
        query_mask = q_encoded["attention_mask"].to(device)

        # Selection
        with torch.no_grad():
            selected_ids, scores = model.select_documents(query_ids, query_mask, k=top_k)
        selected_list = selected_ids[0].cpu().tolist()

        # Selection success?
        recall = compute_recall_at_k(selected_list, gold_doc_ids, k=top_k)
        selection_correct = recall > 0

        # Generation
        try:
            with torch.no_grad():
                generated_ids = model.generate(
                    query_ids=query_ids,
                    doc_indices=selected_ids,
                    query_attention_mask=query_mask,
                    max_new_tokens=64,
                )
            prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        except Exception:
            prediction = ""

        em = compute_em(prediction, answer)
        generation_correct = em > 0

        # Categorize
        entry = {
            "question": question,
            "answer": answer,
            "prediction": prediction[:200],
            "gold_doc_ids": gold_doc_ids,
            "selected_ids": selected_list,
            "recall_at_k": recall,
            "em": em,
        }

        if selection_correct and generation_correct:
            error_categories["success"].append(entry)
        elif not selection_correct and not generation_correct:
            error_categories["both_failure"].append(entry)
        elif not selection_correct:
            error_categories["selection_failure"].append(entry)
        else:
            error_categories["generation_failure"].append(entry)

    # Statistics
    total = sum(len(v) for v in error_categories.values())
    stats = {
        cat: {"count": len(entries), "ratio": len(entries) / max(total, 1)}
        for cat, entries in error_categories.items()
    }

    logger.info("Error Analysis Results:")
    for cat, s in stats.items():
        logger.info(f"  {cat}: {s['count']} ({s['ratio']*100:.1f}%)")

    return {
        "categories": error_categories,
        "stats": stats,
        "total_samples": total,
    }


def analyze_multihop_errors(
    model,
    qa_pairs: list,
    corpus: dict,
    tokenizer,
    top_k: int = 10,
    max_samples: int = 200,
    device: str = "cuda",
) -> dict:
    """
    Multi-hop 전용 에러 분석 (HotpotQA)

    Error categories:
    - bridge_not_found: bridge entity가 포함된 문서가 검색 안 됨
    - bridge_not_recognized: 문서는 있지만 bridge entity 활용 못 함
    - propagation_error: bridge까지 맞았지만 최종 추론 실패
    """
    model.eval()

    error_types = Counter()
    error_examples = {"bridge_not_found": [], "bridge_not_recognized": [],
                      "propagation_error": [], "success": []}

    for item in qa_pairs[:max_samples]:
        if "type" not in item:
            continue  # HotpotQA specific

        question = item["question"]
        answer = item["answer"]
        gold_doc_ids = item["gold_doc_ids"]
        q_type = item.get("type", "unknown")

        # Selection
        q_encoded = tokenizer(
            question, max_length=256, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        query_ids = q_encoded["input_ids"].to(device)
        query_mask = q_encoded["attention_mask"].to(device)

        with torch.no_grad():
            selected_ids, _ = model.select_documents(query_ids, query_mask, k=top_k)
        selected_list = selected_ids[0].cpu().tolist()

        # Check gold doc coverage
        gold_covered = [gid in selected_list for gid in gold_doc_ids]

        # Generate
        try:
            with torch.no_grad():
                generated_ids = model.generate(
                    query_ids=query_ids,
                    doc_indices=selected_ids,
                    query_attention_mask=query_mask,
                    max_new_tokens=64,
                )
            prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        except Exception:
            prediction = ""

        em = compute_em(prediction, answer)

        # Categorize multi-hop error
        entry = {
            "question": question,
            "answer": answer,
            "prediction": prediction[:200],
            "type": q_type,
            "gold_covered": gold_covered,
        }

        if em > 0:
            error_types["success"] += 1
            error_examples["success"].append(entry)
        elif not all(gold_covered):
            # 일부 gold doc이 검색 안 됨 → bridge not found
            error_types["bridge_not_found"] += 1
            error_examples["bridge_not_found"].append(entry)
        elif all(gold_covered) and not em:
            # Gold docs 다 있는데 답 틀림
            # F1으로 부분 정답 확인
            f1 = compute_f1(prediction, answer)
            if f1 > 0.3:
                # 부분적으로 맞음 → propagation error
                error_types["propagation_error"] += 1
                error_examples["propagation_error"].append(entry)
            else:
                # 완전히 틀림 → bridge not recognized
                error_types["bridge_not_recognized"] += 1
                error_examples["bridge_not_recognized"].append(entry)

    total = sum(error_types.values())
    stats = {k: {"count": v, "ratio": v / max(total, 1)} for k, v in error_types.items()}

    logger.info("Multi-hop Error Analysis:")
    for cat, s in stats.items():
        logger.info(f"  {cat}: {s['count']} ({s['ratio']*100:.1f}%)")

    return {
        "error_types": dict(error_types),
        "stats": stats,
        "examples": {k: v[:5] for k, v in error_examples.items()},  # top 5 examples
        "total": total,
    }
