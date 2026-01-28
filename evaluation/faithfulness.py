"""
Faithfulness 평가 모듈

Evidence가 Answer를 정당화(justify)하는지 평가합니다.

방법:
1. Answer Containment: Answer가 Evidence에 포함되는지 (기본)
2. Entailment Proxy: NLI 모델로 Evidence → Answer entailment 확인 (중급)
3. LLM-as-Judge: LLM이 직접 Evidence로 Answer 도출 가능한지 판단 (고급)
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FaithfulnessResult:
    """Faithfulness 평가 결과"""
    score: float              # 종합 점수 (0-1)
    containment: float        # Answer containment 점수
    entailment: Optional[float] = None  # NLI 기반 점수
    llm_judge: Optional[float] = None   # LLM 판정 점수
    details: Optional[Dict] = None      # 상세 정보


def normalize_text(text: str) -> str:
    """텍스트 정규화 (비교용)"""
    text = text.lower().strip()
    # 연속 공백 제거
    text = re.sub(r'\s+', ' ', text)
    # 구두점 주변 공백 정리
    text = re.sub(r'\s*([.,!?;:])\s*', r'\1 ', text)
    return text


def compute_answer_containment(
    evidence: str,
    answer: str,
    normalize: bool = True,
) -> Tuple[float, Dict]:
    """
    Answer Containment 계산

    Evidence가 Answer를 포함하는지 확인 (exact match, partial match)

    Args:
        evidence: Evidence 텍스트
        answer: Answer 텍스트
        normalize: 텍스트 정규화 여부

    Returns:
        Tuple of (score, details)
    """
    if normalize:
        evidence_norm = normalize_text(evidence)
        answer_norm = normalize_text(answer)
    else:
        evidence_norm = evidence
        answer_norm = answer

    details = {
        "exact_match": False,
        "partial_match": False,
        "token_overlap": 0.0,
    }

    # 1. Exact match
    if answer_norm in evidence_norm:
        details["exact_match"] = True
        return 1.0, details

    # 2. Token-level overlap
    answer_tokens = set(answer_norm.split())
    evidence_tokens = set(evidence_norm.split())

    if not answer_tokens:
        return 0.0, details

    overlap = answer_tokens & evidence_tokens
    overlap_ratio = len(overlap) / len(answer_tokens)
    details["token_overlap"] = overlap_ratio

    # 3. Partial match (answer의 50% 이상 토큰이 evidence에 있으면)
    if overlap_ratio >= 0.5:
        details["partial_match"] = True
        return min(overlap_ratio, 1.0), details

    return overlap_ratio * 0.5, details


def compute_entailment_score(
    evidence: str,
    answer: str,
    question: str,
    model_name: str = "microsoft/deberta-v3-base-mnli-fever-anli",
) -> Tuple[float, Dict]:
    """
    NLI 모델을 사용한 Entailment 점수 계산

    Premise: Evidence
    Hypothesis: "The answer to '{question}' is '{answer}'"

    Args:
        evidence: Evidence 텍스트
        answer: Answer 텍스트
        question: 원래 질문
        model_name: NLI 모델 이름

    Returns:
        Tuple of (entailment_score, details)
    """
    try:
        from transformers import pipeline
    except ImportError:
        logger.warning("transformers not available for NLI")
        return 0.0, {"error": "transformers not available"}

    try:
        # NLI pipeline
        nli = pipeline("text-classification", model=model_name, device=0)

        # Hypothesis 구성
        hypothesis = f"The answer to '{question}' is '{answer}'"

        # NLI 예측
        result = nli(f"{evidence}", hypothesis)

        # 결과 파싱
        label = result[0]["label"].lower()
        score = result[0]["score"]

        details = {
            "label": label,
            "confidence": score,
            "hypothesis": hypothesis[:100],
        }

        if label == "entailment":
            return score, details
        elif label == "neutral":
            return score * 0.5, details
        else:  # contradiction
            return 0.0, details

    except Exception as e:
        logger.warning(f"NLI inference failed: {e}")
        return 0.0, {"error": str(e)}


def compute_llm_judge_score(
    evidence: str,
    answer: str,
    question: str,
    judge_model = None,
    judge_tokenizer = None,
) -> Tuple[float, Dict]:
    """
    LLM-as-Judge 방식으로 Faithfulness 판정

    LLM에게 "이 Evidence로 Answer를 도출할 수 있는가?" 질문

    Args:
        evidence: Evidence 텍스트
        answer: Answer 텍스트
        question: 원래 질문
        judge_model: 판정에 사용할 LLM (None이면 새로 로드)
        judge_tokenizer: 토크나이저

    Returns:
        Tuple of (score, details)
    """
    prompt = f"""You are evaluating whether an answer can be derived from the given evidence.

Evidence: {evidence}

Question: {question}
Answer: {answer}

Can this answer be directly derived or inferred from the evidence?
Respond with only "YES" or "NO"."""

    if judge_model is None:
        # 모델 로드 비용이 크므로 경고
        logger.warning("LLM judge requires model loading - consider passing pre-loaded model")
        return 0.0, {"error": "model not provided"}

    try:
        import torch

        inputs = judge_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(judge_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = judge_model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
            )

        response = judge_tokenizer.decode(outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
        response = response.strip().upper()

        details = {
            "response": response,
            "prompt_length": len(prompt),
        }

        if "YES" in response:
            return 1.0, details
        elif "NO" in response:
            return 0.0, details
        else:
            # 애매한 응답
            return 0.5, details

    except Exception as e:
        logger.warning(f"LLM judge failed: {e}")
        return 0.0, {"error": str(e)}


def compute_faithfulness(
    evidence: str,
    answer: str,
    question: str,
    use_entailment: bool = False,
    use_llm_judge: bool = False,
    entailment_model: str = None,
    judge_model = None,
    judge_tokenizer = None,
) -> FaithfulnessResult:
    """
    종합 Faithfulness 점수 계산

    Args:
        evidence: Evidence 텍스트
        answer: Answer 텍스트
        question: 원래 질문
        use_entailment: NLI 기반 평가 사용 여부
        use_llm_judge: LLM 판정 사용 여부
        entailment_model: NLI 모델 이름
        judge_model: LLM judge 모델
        judge_tokenizer: LLM judge 토크나이저

    Returns:
        FaithfulnessResult
    """
    # 1. Answer Containment (기본)
    containment, containment_details = compute_answer_containment(evidence, answer)

    scores = [containment]
    weights = [1.0]

    result = FaithfulnessResult(
        score=containment,
        containment=containment,
        details={"containment": containment_details},
    )

    # 2. Entailment (옵션)
    if use_entailment:
        entailment, ent_details = compute_entailment_score(
            evidence, answer, question,
            model_name=entailment_model or "microsoft/deberta-v3-base-mnli-fever-anli",
        )
        result.entailment = entailment
        result.details["entailment"] = ent_details
        scores.append(entailment)
        weights.append(0.5)

    # 3. LLM Judge (옵션)
    if use_llm_judge and judge_model is not None:
        llm_score, llm_details = compute_llm_judge_score(
            evidence, answer, question, judge_model, judge_tokenizer
        )
        result.llm_judge = llm_score
        result.details["llm_judge"] = llm_details
        scores.append(llm_score)
        weights.append(1.0)

    # 가중 평균
    total_weight = sum(weights)
    weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
    result.score = weighted_score

    return result


def evaluate_faithfulness_batch(
    evidences: List[str],
    answers: List[str],
    questions: List[str],
    use_entailment: bool = False,
    use_llm_judge: bool = False,
    show_progress: bool = True,
) -> Dict:
    """
    배치로 Faithfulness 평가

    Args:
        evidences: Evidence 텍스트 리스트
        answers: Answer 텍스트 리스트
        questions: 질문 리스트
        use_entailment: NLI 사용 여부
        use_llm_judge: LLM judge 사용 여부
        show_progress: 진행률 표시

    Returns:
        평균 점수 및 통계
    """
    from tqdm import tqdm

    assert len(evidences) == len(answers) == len(questions)

    results = []
    iterator = zip(evidences, answers, questions)
    if show_progress:
        iterator = tqdm(list(iterator), desc="Computing faithfulness")

    for evidence, answer, question in iterator:
        result = compute_faithfulness(
            evidence=evidence,
            answer=answer,
            question=question,
            use_entailment=use_entailment,
            use_llm_judge=use_llm_judge,
        )
        results.append(result)

    # 통계 계산
    scores = [r.score for r in results]
    containments = [r.containment for r in results]

    stats = {
        "faithfulness": sum(scores) / len(scores),
        "containment": sum(containments) / len(containments),
        "num_samples": len(results),
        "num_exact_match": sum(1 for r in results if r.details.get("containment", {}).get("exact_match", False)),
        "num_partial_match": sum(1 for r in results if r.details.get("containment", {}).get("partial_match", False)),
    }

    if use_entailment:
        entailments = [r.entailment for r in results if r.entailment is not None]
        if entailments:
            stats["entailment"] = sum(entailments) / len(entailments)

    return stats


if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.INFO)

    # 테스트 케이스
    test_cases = [
        {
            "evidence": "Paris is the capital and most populous city of France.",
            "answer": "Paris",
            "question": "What is the capital of France?",
        },
        {
            "evidence": "The Eiffel Tower is located in Paris and was built in 1889.",
            "answer": "1889",
            "question": "When was the Eiffel Tower built?",
        },
        {
            "evidence": "France is a country in Western Europe.",
            "answer": "Berlin",  # Wrong answer
            "question": "What is the capital of France?",
        },
    ]

    print("Faithfulness Evaluation Test\n")

    for i, tc in enumerate(test_cases):
        print(f"Test case {i+1}:")
        print(f"  Question: {tc['question']}")
        print(f"  Answer: {tc['answer']}")
        print(f"  Evidence: {tc['evidence'][:60]}...")

        result = compute_faithfulness(
            evidence=tc["evidence"],
            answer=tc["answer"],
            question=tc["question"],
        )

        print(f"  Score: {result.score:.3f}")
        print(f"  Containment: {result.containment:.3f}")
        print(f"  Details: {result.details}")
        print()
