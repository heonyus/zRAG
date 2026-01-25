"""
QA 평가 모듈
- EM, F1 (Answer Quality)
- Recall@K, MRR (Selection Quality)
- 종합 결과 테이블 생성
"""

import torch
from tqdm import tqdm
from typing import Optional, List
import logging

from .metrics import (
    compute_em, compute_f1, compute_recall_at_k,
    compute_mrr, compute_selection_precision, aggregate_metrics,
)

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_qa(
    model,
    qa_pairs: list,
    tokenizer,
    top_k: int = 5,
    max_samples: Optional[int] = None,
    max_new_tokens: int = 64,
    device: str = "cuda",
) -> dict:
    """
    QA 성능 평가

    Args:
        model: ParametricQA instance
        qa_pairs: [{"question": ..., "answer": ..., "gold_doc_ids": [...]}, ...]
        top_k: selection top-k
        max_samples: 평가할 최대 샘플 수

    Returns:
        dict with em, f1, recall_at_k, mrr, selection_precision
    """
    model.eval()

    if max_samples:
        qa_pairs = qa_pairs[:max_samples]

    all_metrics = []

    for item in tqdm(qa_pairs, desc="Evaluating QA"):
        question = item["question"]
        answer = item["answer"]
        gold_doc_ids = item["gold_doc_ids"]

        # Tokenize question
        q_encoded = tokenizer(
            question,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        question_ids = q_encoded["input_ids"].to(device)
        question_mask = q_encoded["attention_mask"].to(device)

        # Document selection
        selected_ids, selected_scores = model.select_documents(
            question_ids, question_mask, k=top_k
        )
        selected_list = selected_ids[0].cpu().tolist()

        # Generate answer
        try:
            generated_ids = model.generate(
                query_ids=question_ids,
                doc_indices=selected_ids,
                query_attention_mask=question_mask,
                max_new_tokens=max_new_tokens,
            )
            prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            # Remove the prompt from generation (if echoed)
            if question in prediction:
                prediction = prediction.replace(question, "").strip()
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            prediction = ""

        # Compute metrics
        em = compute_em(prediction, answer)
        f1 = compute_f1(prediction, answer)
        recall = compute_recall_at_k(selected_list, gold_doc_ids, k=top_k)
        mrr = compute_mrr(selected_list, gold_doc_ids)
        sel_prec = compute_selection_precision(selected_list, gold_doc_ids)

        all_metrics.append({
            "em": em,
            "f1": f1,
            "recall_at_k": recall,
            "mrr": mrr,
            "selection_precision": sel_prec,
        })

    # Aggregate
    result = aggregate_metrics(all_metrics)
    result["num_samples"] = len(all_metrics)

    logger.info(f"QA Eval ({len(all_metrics)} samples): "
                f"EM={result.get('em', 0):.4f}, "
                f"F1={result.get('f1', 0):.4f}, "
                f"Recall@{top_k}={result.get('recall_at_k', 0):.4f}")

    return result


@torch.no_grad()
def evaluate_qa_with_seeds(
    model_factory,  # callable that returns model
    qa_pairs: list,
    tokenizer,
    seeds: List[int] = [42, 123, 456],
    top_k: int = 5,
    max_samples: Optional[int] = None,
    device: str = "cuda",
) -> dict:
    """
    Multiple seeds로 평가하여 mean ± std 보고

    Args:
        model_factory: seed를 받아 모델을 반환하는 함수
    """
    import numpy as np

    all_results = []

    for seed in seeds:
        torch.manual_seed(seed)
        model = model_factory(seed)

        result = evaluate_qa(
            model=model,
            qa_pairs=qa_pairs,
            tokenizer=tokenizer,
            top_k=top_k,
            max_samples=max_samples,
            device=device,
        )
        all_results.append(result)
        logger.info(f"Seed {seed}: EM={result['em']:.4f}, F1={result['f1']:.4f}")

    # Aggregate across seeds
    keys = ["em", "f1", "recall_at_k", "mrr", "selection_precision"]
    final = {}
    for key in keys:
        values = [r[key] for r in all_results if key in r]
        if values:
            final[f"{key}_mean"] = float(np.mean(values))
            final[f"{key}_std"] = float(np.std(values))

    logger.info(f"Multi-seed results: EM={final.get('em_mean', 0):.4f}±{final.get('em_std', 0):.4f}")
    return final


def generate_results_table(
    results: dict,
    method_names: list = None,
    metrics: list = None,
) -> str:
    """
    결과 테이블 생성 (Markdown 형식)

    Args:
        results: {method_name: {metric: value, ...}, ...}
        metrics: 표시할 metrics 목록
    """
    if method_names is None:
        method_names = list(results.keys())
    if metrics is None:
        metrics = ["em", "f1", "recall_at_k", "mrr", "latency_ms", "storage_mb"]

    # Header
    header = "| Method | " + " | ".join(metrics) + " |"
    separator = "|--------|" + "|".join(["--------"] * len(metrics)) + "|"

    rows = [header, separator]

    for method in method_names:
        if method not in results:
            continue
        r = results[method]
        values = []
        for m in metrics:
            if f"{m}_mean" in r and f"{m}_std" in r:
                values.append(f"{r[f'{m}_mean']:.2f}±{r[f'{m}_std']:.2f}")
            elif m in r:
                val = r[m]
                values.append(f"{val:.2f}" if isinstance(val, float) else str(val))
            else:
                values.append("-")
        row = f"| {method} | " + " | ".join(values) + " |"
        rows.append(row)

    return "\n".join(rows)
