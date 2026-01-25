"""
RAGAS 평가 모듈
- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall
"""

import torch
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def evaluate_ragas(
    predictions: List[dict],
    use_openai: bool = False,
    openai_model: str = "gpt-4o-mini",
) -> dict:
    """
    RAGAS 프레임워크로 RAG 품질 평가

    Args:
        predictions: [{
            "question": str,
            "answer": str (model prediction),
            "contexts": [str, ...] (retrieved/used contexts),
            "ground_truth": str (gold answer),
        }, ...]
        use_openai: OpenAI API 사용 여부

    Returns:
        dict with faithfulness, answer_relevancy, context_precision, context_recall
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset as HFDataset
    except ImportError:
        logger.warning("ragas not installed. Skipping RAGAS evaluation.")
        return _fallback_ragas_metrics(predictions)

    # RAGAS format으로 변환
    ragas_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for pred in predictions:
        ragas_data["question"].append(pred["question"])
        ragas_data["answer"].append(pred["answer"])
        ragas_data["contexts"].append(pred.get("contexts", [""]))
        ragas_data["ground_truth"].append(pred.get("ground_truth", ""))

    dataset = HFDataset.from_dict(ragas_data)

    # Evaluate
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    try:
        result = ragas_evaluate(
            dataset=dataset,
            metrics=metrics,
        )
        return {
            "faithfulness": result.get("faithfulness", 0.0),
            "answer_relevancy": result.get("answer_relevancy", 0.0),
            "context_precision": result.get("context_precision", 0.0),
            "context_recall": result.get("context_recall", 0.0),
        }
    except Exception as e:
        logger.warning(f"RAGAS evaluation failed: {e}")
        return _fallback_ragas_metrics(predictions)


def _fallback_ragas_metrics(predictions: List[dict]) -> dict:
    """
    RAGAS 사용 불가 시 간이 메트릭 계산

    - Faithfulness: answer에 있는 단어 중 context에도 있는 비율
    - Answer Relevancy: answer와 question의 token overlap
    """
    faithfulness_scores = []
    relevancy_scores = []

    for pred in predictions:
        answer_tokens = set(pred["answer"].lower().split())
        question_tokens = set(pred["question"].lower().split())

        # Context tokens
        contexts = pred.get("contexts", [""])
        context_tokens = set()
        for ctx in contexts:
            context_tokens.update(ctx.lower().split())

        # Faithfulness: answer tokens in context
        if answer_tokens:
            faith = len(answer_tokens & context_tokens) / len(answer_tokens)
            faithfulness_scores.append(faith)

        # Relevancy: answer-question overlap
        if answer_tokens:
            rel = len(answer_tokens & question_tokens) / len(answer_tokens)
            relevancy_scores.append(rel)

    return {
        "faithfulness": sum(faithfulness_scores) / max(len(faithfulness_scores), 1),
        "answer_relevancy": sum(relevancy_scores) / max(len(relevancy_scores), 1),
        "context_precision": 0.0,  # requires RAGAS
        "context_recall": 0.0,    # requires RAGAS
        "note": "fallback_metrics (ragas not available)",
    }


@torch.no_grad()
def prepare_ragas_predictions(
    model,
    qa_pairs: list,
    corpus: dict,
    tokenizer,
    top_k: int = 5,
    max_new_tokens: int = 64,
    max_samples: int = 100,
    device: str = "cuda",
) -> List[dict]:
    """
    모델 예측 결과를 RAGAS 입력 형식으로 준비

    Returns:
        [{question, answer, contexts, ground_truth}, ...]
    """
    model.eval()
    predictions = []

    for item in qa_pairs[:max_samples]:
        question = item["question"]
        gold_answer = item["answer"]
        gold_doc_ids = item["gold_doc_ids"]

        # Tokenize
        q_encoded = tokenizer(
            question,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        query_ids = q_encoded["input_ids"].to(device)
        query_mask = q_encoded["attention_mask"].to(device)

        # Select and generate
        selected_ids, _ = model.select_documents(query_ids, query_mask, k=top_k)
        selected_list = selected_ids[0].cpu().tolist()

        # Get context texts
        contexts = []
        for doc_id in selected_list:
            if doc_id in corpus:
                contexts.append(corpus[doc_id])

        # Generate answer
        try:
            generated_ids = model.generate(
                query_ids=query_ids,
                doc_indices=selected_ids,
                query_attention_mask=query_mask,
                max_new_tokens=max_new_tokens,
            )
            pred_answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        except Exception:
            pred_answer = ""

        predictions.append({
            "question": question,
            "answer": pred_answer,
            "contexts": contexts if contexts else [""],
            "ground_truth": gold_answer,
        })

    return predictions
