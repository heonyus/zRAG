"""
평가 지표 모듈
- Exact Match (EM)
- Token F1
- Recall@K
- ROUGE-L
- Supporting Fact F1 (HotpotQA)
"""

import re
import string
from collections import Counter
from typing import List, Union

# Optional: rouge_score library (fallback to simple implementation if not available)
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


def extract_answer_span(text: str) -> str:
    """
    모델 출력에서 답 span만 추출 (공격적 후처리)

    LLM은 종종 "Answer: ...", "The answer is ...", "Based on the evidence..."
    같은 프리앰블을 붙이므로, 실제 답 부분만 추출해야 EM이 제대로 계산됨.
    """
    if not text:
        return ""

    t = text.strip()

    # 1. 흔한 프리앰블 패턴 제거
    t = re.sub(r'^(based on the evidence provided,?\s*)', '', t, flags=re.I).strip()
    t = re.sub(r'^(based on the (given )?evidence,?\s*)', '', t, flags=re.I).strip()
    t = re.sub(r'^(according to the evidence,?\s*)', '', t, flags=re.I).strip()
    t = re.sub(r'^(the answer is|answer is|answer:)\s*', '', t, flags=re.I).strip()
    t = re.sub(r'^(답:|답은)\s*', '', t).strip()

    # 2. 첫 줄만 (HotpotQA는 단답이 많음)
    t = t.split("\n")[0].strip()

    # 3. 첫 문장만 (마침표 기준)
    t = t.split(".")[0].strip()

    # 4. 따옴표/특수문자 정리
    t = t.strip(' "\'\t.,;:!?')

    # 5. 너무 길면 앞 8단어만 (HotpotQA 답은 대부분 짧음)
    words = t.split()
    if len(words) > 8:
        t = " ".join(words[:8]).strip()

    return t


def normalize_answer(text: str) -> str:
    """답변 정규화 (EM/F1 계산 전처리)"""
    text = text.lower()
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.strip()


def compute_em(prediction: str, gold: Union[str, List[str]]) -> float:
    """
    Exact Match 계산

    Args:
        prediction: 모델 예측
        gold: 정답 (단일 또는 복수 허용)
    """
    pred_norm = normalize_answer(prediction)

    if isinstance(gold, str):
        gold_list = [gold]
    else:
        gold_list = gold

    for g in gold_list:
        if pred_norm == normalize_answer(g):
            return 1.0
    return 0.0


def compute_f1(prediction: str, gold: Union[str, List[str]]) -> float:
    """
    Token-level F1 계산

    Args:
        prediction: 모델 예측
        gold: 정답
    """
    pred_tokens = normalize_answer(prediction).split()

    if isinstance(gold, str):
        gold_list = [gold]
    else:
        gold_list = gold

    best_f1 = 0.0
    for g in gold_list:
        gold_tokens = normalize_answer(g).split()

        if not pred_tokens or not gold_tokens:
            f1 = float(pred_tokens == gold_tokens)
        else:
            common = Counter(pred_tokens) & Counter(gold_tokens)
            num_common = sum(common.values())

            if num_common == 0:
                f1 = 0.0
            else:
                precision = num_common / len(pred_tokens)
                recall = num_common / len(gold_tokens)
                f1 = 2 * precision * recall / (precision + recall)

        best_f1 = max(best_f1, f1)

    return best_f1


def compute_recall_at_k(
    selected_ids: List[int],
    gold_ids: List[int],
    k: int = None,
) -> float:
    """
    Recall@K: top-K 선택에 gold document 포함 비율

    Args:
        selected_ids: 선택된 document ids
        gold_ids: 정답 document ids
        k: top-k (None이면 selected_ids 전체)
    """
    if not gold_ids:
        return 0.0

    if k is not None:
        selected_ids = selected_ids[:k]

    selected_set = set(selected_ids)
    gold_set = set(gold_ids)

    hits = len(selected_set & gold_set)
    return hits / len(gold_set)


def compute_mrr(
    selected_ids: List[int],
    gold_ids: List[int],
) -> float:
    """
    Mean Reciprocal Rank

    Args:
        selected_ids: 선택된 document ids (순서 중요)
        gold_ids: 정답 document ids
    """
    gold_set = set(gold_ids)

    for rank, doc_id in enumerate(selected_ids, 1):
        if doc_id in gold_set:
            return 1.0 / rank

    return 0.0


def compute_selection_precision(
    selected_ids: List[int],
    gold_ids: List[int],
) -> float:
    """Selection Precision: 선택된 문서 중 relevant 비율"""
    if not selected_ids:
        return 0.0

    gold_set = set(gold_ids)
    hits = sum(1 for sid in selected_ids if sid in gold_set)
    return hits / len(selected_ids)


def compute_rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 score"""
    if ROUGE_AVAILABLE:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return scores['rougeL'].fmeasure
    else:
        # Fallback: simple LCS-based implementation
        from .evidence_metrics import compute_rouge_l as compute_rouge_l_simple
        result = compute_rouge_l_simple(prediction, reference)
        return result["f1"]


def compute_supporting_fact_f1(
    predicted_sents: List[tuple],  # [(title, sent_idx), ...]
    gold_sents: List[tuple],       # [(title, sent_idx), ...]
) -> float:
    """
    Supporting Fact F1 (HotpotQA)

    Args:
        predicted_sents: 예측된 supporting sentences
        gold_sents: 정답 supporting sentences
    """
    pred_set = set(predicted_sents)
    gold_set = set(gold_sents)

    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0

    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def compute_joint_metrics(
    answer_em: float,
    answer_f1: float,
    sp_precision: float,
    sp_recall: float,
) -> dict:
    """
    Joint metrics for HotpotQA

    Joint EM = answer_em * sp_em
    Joint F1 = answer_f1 * sp_f1
    """
    sp_f1 = 2 * sp_precision * sp_recall / (sp_precision + sp_recall + 1e-8)
    sp_em = float(sp_precision == 1.0 and sp_recall == 1.0)

    return {
        "joint_em": answer_em * sp_em,
        "joint_f1": answer_f1 * sp_f1,
        "sp_em": sp_em,
        "sp_f1": sp_f1,
    }


def aggregate_metrics(metric_list: List[dict]) -> dict:
    """
    여러 샘플의 metrics를 평균/std로 집계

    Args:
        metric_list: [{"em": 1.0, "f1": 0.8}, {"em": 0.0, "f1": 0.3}, ...]
    """
    import numpy as np

    if not metric_list:
        return {}

    keys = metric_list[0].keys()
    result = {}

    for key in keys:
        values = [m[key] for m in metric_list if key in m and m[key] is not None]
        if values:
            result[f"{key}"] = float(np.mean(values))
            result[f"{key}_std"] = float(np.std(values))

    return result
