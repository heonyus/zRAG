"""
QA 및 Evidence 평가 모듈

새 설계 (LLM-as-Memory):
- Evidence Quality: ROUGE-L, Answer Coverage, Token F1
- RAG Baseline 비교

레거시 (ParametricQA):
- EM, F1 (Answer Quality)
- Recall@K, MRR (Selection Quality)
"""

import torch
from tqdm import tqdm
from typing import Optional, List, Dict
import logging

from .metrics import (
    compute_em, compute_f1, compute_recall_at_k,
    compute_mrr, compute_selection_precision, aggregate_metrics,
)
from .evidence_metrics import (
    evaluate_evidence_quality,
    compare_with_rag,
    print_comparison_table,
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


# ============================================
# 새 설계: Evidence 평가 함수들
# ============================================

@torch.no_grad()
def evaluate_evidence(
    model,
    evidence_pairs: List[Dict],
    tokenizer,
    corpus: Optional[Dict] = None,
    max_samples: Optional[int] = None,
    max_new_tokens: int = 128,
    device: str = "cuda",
) -> Dict:
    """
    Evidence 생성 품질 평가 (LLM-as-Memory 새 설계)

    Args:
        model: ParametricMemoryLLM instance
        evidence_pairs: [{
            "query": str,
            "evidence": str (gold evidence),
            "answer": str (optional),
            "gold_doc_id": int (optional),
        }, ...]
        tokenizer: LLM tokenizer
        corpus: {doc_id: doc_text, ...} for faithfulness (optional)
        max_samples: 평가할 최대 샘플 수
        max_new_tokens: 생성 최대 토큰 수
        device: 디바이스

    Returns:
        dict with evidence quality metrics
    """
    model.eval()

    if max_samples:
        evidence_pairs = evidence_pairs[:max_samples]

    all_metrics = []
    generated_evidences = []
    gold_evidences = []
    answers = []

    for item in tqdm(evidence_pairs, desc="Evaluating Evidence"):
        query = item["query"]
        gold_evidence = item["evidence"]
        answer = item.get("answer", "")
        gold_doc_id = item.get("gold_doc_id", -1)

        # Tokenize query
        q_encoded = tokenizer(
            query,
            max_length=128,
            truncation=True,
            return_tensors="pt",
        )
        query_ids = q_encoded["input_ids"].to(device)

        # Generate evidence
        try:
            generated = model.generate_evidence(
                query_ids=query_ids,
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            logger.warning(f"Evidence generation failed: {e}")
            generated = ""

        # Source document for faithfulness
        source_document = None
        if corpus and gold_doc_id >= 0:
            source_document = corpus.get(gold_doc_id, None)

        # Compute metrics
        metrics = evaluate_evidence_quality(
            generated_evidence=generated,
            gold_evidence=gold_evidence,
            answer=answer,
            source_document=source_document,
        )
        all_metrics.append(metrics)

        # Store for RAG comparison
        generated_evidences.append(generated)
        gold_evidences.append(gold_evidence)
        answers.append(answer)

    # Aggregate
    if not all_metrics:
        return {"num_samples": 0}

    aggregated = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if key in m]
        if values:
            aggregated[key] = sum(values) / len(values)

    aggregated["num_samples"] = len(all_metrics)

    # Log results
    logger.info(
        f"Evidence Eval ({len(all_metrics)} samples): "
        f"ROUGE-L={aggregated.get('rouge_l_f1', 0):.4f}, "
        f"AnswerCov={aggregated.get('answer_coverage', 0):.4f}"
    )

    # Store raw outputs for later analysis
    aggregated["_generated_evidences"] = generated_evidences
    aggregated["_gold_evidences"] = gold_evidences
    aggregated["_answers"] = answers

    return aggregated


@torch.no_grad()
def evaluate_evidence_vs_rag(
    model,
    evidence_pairs: List[Dict],
    rag_retrieved: List[str],
    tokenizer,
    max_samples: Optional[int] = None,
    max_new_tokens: int = 128,
    device: str = "cuda",
) -> Dict:
    """
    우리 방식 vs RAG Baseline 비교 평가

    Args:
        model: ParametricMemoryLLM instance
        evidence_pairs: [{query, evidence (gold), answer}, ...]
        rag_retrieved: RAG로 검색된 텍스트들 (동일 순서)
        tokenizer: LLM tokenizer

    Returns:
        dict with comparison metrics
    """
    model.eval()

    if max_samples:
        evidence_pairs = evidence_pairs[:max_samples]
        rag_retrieved = rag_retrieved[:max_samples]

    our_evidences = []

    for item in tqdm(evidence_pairs, desc="Generating Evidence for Comparison"):
        query = item["query"]

        q_encoded = tokenizer(
            query,
            max_length=128,
            truncation=True,
            return_tensors="pt",
        )
        query_ids = q_encoded["input_ids"].to(device)

        try:
            generated = model.generate_evidence(
                query_ids=query_ids,
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            generated = ""

        our_evidences.append(generated)

    # Compare
    gold_evidences = [item["evidence"] for item in evidence_pairs]
    answers = [item.get("answer", "") for item in evidence_pairs]

    comparison = compare_with_rag(
        our_evidence=our_evidences,
        rag_retrieved=rag_retrieved,
        gold_evidence=gold_evidences,
        answers=answers,
    )

    # Print table
    print_comparison_table(comparison)

    return comparison


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
