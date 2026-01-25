"""
Evidence 품질 평가 메트릭

기존 RAG와 비교 가능한 메트릭:
- ROUGE-L: Generated Evidence vs Gold Evidence
- Answer Coverage: Evidence가 Answer를 포함하는지
- Faithfulness: Evidence가 원본 문서에 기반하는지
"""

import re
from typing import List, Optional
from collections import Counter


def normalize_text(text: str) -> str:
    """텍스트 정규화"""
    # 소문자 변환
    text = text.lower()
    # 특수문자 제거
    text = re.sub(r'[^\w\s]', '', text)
    # 여러 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_tokens(text: str) -> List[str]:
    """토큰화"""
    return normalize_text(text).split()


def compute_rouge_l(prediction: str, reference: str) -> dict:
    """
    ROUGE-L (Longest Common Subsequence) F1 계산

    Args:
        prediction: 생성된 텍스트
        reference: 정답 텍스트

    Returns:
        dict with precision, recall, f1
    """
    pred_tokens = get_tokens(prediction)
    ref_tokens = get_tokens(reference)

    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # LCS 길이 계산 (DP)
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]

    precision = lcs_length / len(pred_tokens) if pred_tokens else 0.0
    recall = lcs_length / len(ref_tokens) if ref_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_answer_coverage(evidence: str, answer: str) -> float:
    """
    Evidence가 Answer를 포함하는지 측정

    Args:
        evidence: 생성된 evidence
        answer: 정답

    Returns:
        1.0 if answer in evidence, else partial match score
    """
    evidence_norm = normalize_text(evidence)
    answer_norm = normalize_text(answer)

    if not answer_norm:
        return 1.0

    # 완전 포함
    if answer_norm in evidence_norm:
        return 1.0

    # 부분 포함 (토큰 기준)
    answer_tokens = set(get_tokens(answer))
    evidence_tokens = set(get_tokens(evidence))

    if not answer_tokens:
        return 1.0

    overlap = len(answer_tokens & evidence_tokens)
    return overlap / len(answer_tokens)


def compute_token_f1(prediction: str, reference: str) -> dict:
    """
    Token-level F1 계산 (EM/F1 스타일)

    Args:
        prediction: 생성된 텍스트
        reference: 정답 텍스트

    Returns:
        dict with precision, recall, f1
    """
    pred_tokens = Counter(get_tokens(prediction))
    ref_tokens = Counter(get_tokens(reference))

    # 교집합 (min count)
    common = sum((pred_tokens & ref_tokens).values())

    num_pred = sum(pred_tokens.values())
    num_ref = sum(ref_tokens.values())

    if num_pred == 0 or num_ref == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = common / num_pred
    recall = common / num_ref
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_faithfulness_simple(
    evidence: str,
    source_document: str,
) -> float:
    """
    간단한 Faithfulness 측정
    (Evidence의 내용이 Source Document에 기반하는지)

    Args:
        evidence: 생성된 evidence
        source_document: 원본 문서

    Returns:
        overlap ratio (0~1)
    """
    evidence_tokens = set(get_tokens(evidence))
    source_tokens = set(get_tokens(source_document))

    if not evidence_tokens:
        return 1.0

    overlap = len(evidence_tokens & source_tokens)
    return overlap / len(evidence_tokens)


def evaluate_evidence_quality(
    generated_evidence: str,
    gold_evidence: str,
    answer: str,
    source_document: Optional[str] = None,
) -> dict:
    """
    Evidence 품질 종합 평가

    Args:
        generated_evidence: 생성된 evidence
        gold_evidence: 정답 evidence
        answer: 정답 answer
        source_document: 원본 문서 (optional)

    Returns:
        dict with all metrics
    """
    # ROUGE-L
    rouge_l = compute_rouge_l(generated_evidence, gold_evidence)

    # Token F1
    token_f1 = compute_token_f1(generated_evidence, gold_evidence)

    # Answer Coverage
    answer_coverage = compute_answer_coverage(generated_evidence, answer)

    metrics = {
        "rouge_l_precision": rouge_l["precision"],
        "rouge_l_recall": rouge_l["recall"],
        "rouge_l_f1": rouge_l["f1"],
        "token_f1_precision": token_f1["precision"],
        "token_f1_recall": token_f1["recall"],
        "token_f1_f1": token_f1["f1"],
        "answer_coverage": answer_coverage,
    }

    # Faithfulness (source document 있을 때만)
    if source_document:
        faithfulness = compute_faithfulness_simple(generated_evidence, source_document)
        metrics["faithfulness"] = faithfulness

    return metrics


def evaluate_batch(
    predictions: List[dict],
) -> dict:
    """
    배치 평가 및 집계

    Args:
        predictions: [{
            "generated_evidence": str,
            "gold_evidence": str,
            "answer": str,
            "source_document": str (optional),
        }, ...]

    Returns:
        dict with aggregated metrics
    """
    all_metrics = []

    for pred in predictions:
        metrics = evaluate_evidence_quality(
            generated_evidence=pred["generated_evidence"],
            gold_evidence=pred["gold_evidence"],
            answer=pred["answer"],
            source_document=pred.get("source_document"),
        )
        all_metrics.append(metrics)

    # 집계
    if not all_metrics:
        return {}

    aggregated = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if key in m]
        if values:
            aggregated[f"avg_{key}"] = sum(values) / len(values)
            aggregated[f"max_{key}"] = max(values)
            aggregated[f"min_{key}"] = min(values)

    aggregated["num_samples"] = len(all_metrics)

    return aggregated


def compare_with_rag(
    our_evidence: List[str],
    rag_retrieved: List[str],
    gold_evidence: List[str],
    answers: List[str],
) -> dict:
    """
    우리 방식 vs RAG Retrieved Text 비교

    Args:
        our_evidence: 우리 방식으로 생성된 evidence들
        rag_retrieved: RAG로 검색된 텍스트들
        gold_evidence: 정답 evidence들
        answers: 정답들

    Returns:
        dict with comparison metrics
    """
    our_metrics = []
    rag_metrics = []

    for our, rag, gold, answer in zip(our_evidence, rag_retrieved, gold_evidence, answers):
        # Our method
        our_m = evaluate_evidence_quality(our, gold, answer)
        our_metrics.append(our_m)

        # RAG baseline
        rag_m = evaluate_evidence_quality(rag, gold, answer)
        rag_metrics.append(rag_m)

    # 집계
    def aggregate(metrics_list):
        if not metrics_list:
            return {}
        result = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list]
            result[key] = sum(values) / len(values)
        return result

    our_agg = aggregate(our_metrics)
    rag_agg = aggregate(rag_metrics)

    # 비교
    comparison = {
        "our": our_agg,
        "rag": rag_agg,
        "diff": {
            key: our_agg.get(key, 0) - rag_agg.get(key, 0)
            for key in our_agg.keys()
        },
        "num_samples": len(our_evidence),
    }

    return comparison


def print_comparison_table(comparison: dict):
    """비교 결과 테이블 출력"""
    print("\n" + "=" * 60)
    print("Evidence Quality Comparison: Our Method vs RAG")
    print("=" * 60)

    our = comparison["our"]
    rag = comparison["rag"]
    diff = comparison["diff"]

    print(f"{'Metric':<25} {'Ours':>12} {'RAG':>12} {'Diff':>12}")
    print("-" * 60)

    for key in ["rouge_l_f1", "token_f1_f1", "answer_coverage"]:
        if key in our:
            our_val = our[key]
            rag_val = rag[key]
            diff_val = diff[key]
            sign = "+" if diff_val > 0 else ""
            print(f"{key:<25} {our_val:>12.4f} {rag_val:>12.4f} {sign}{diff_val:>11.4f}")

    print("-" * 60)
    print(f"Samples: {comparison['num_samples']}")
    print("=" * 60)
