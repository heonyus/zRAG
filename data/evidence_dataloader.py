"""
Evidence 생성 학습용 DataLoader

학습 데이터 형식:
{
    "query": str,
    "evidence": str,      # 생성 target
    "gold_doc_id": int,   # (선택) attention guidance용
}
"""

from typing import Optional, List

# Heavy dependencies (optional - for Dataset/DataLoader classes)
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object  # Placeholder for class definition
    DataLoader = None
    AutoTokenizer = None


class EvidenceDataset(Dataset):
    """
    Evidence 생성 학습용 Dataset

    각 샘플: (query, evidence, gold_doc_id)
    - query: 질문 텍스트
    - evidence: 생성할 근거 텍스트 (정답)
    - gold_doc_id: 관련 문서 ID (attention guidance용, optional)
    """

    def __init__(
        self,
        qa_pairs: List[dict],
        tokenizer: AutoTokenizer,
        max_query_length: int = 128,
        max_evidence_length: int = 256,
    ):
        """
        Args:
            qa_pairs: [{"query": str, "evidence": str, "gold_doc_id": int}, ...]
            tokenizer: LLM tokenizer
            max_query_length: 최대 질문 길이
            max_evidence_length: 최대 evidence 길이
        """
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_evidence_length = max_evidence_length

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        item = self.qa_pairs[idx]
        query = item["query"]
        evidence = item["evidence"]
        gold_doc_id = item.get("gold_doc_id", -1)

        # Query tokenization
        query_encoded = self.tokenizer(
            query,
            max_length=self.max_query_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Evidence tokenization (generation target)
        evidence_encoded = self.tokenizer(
            evidence,
            max_length=self.max_evidence_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "query_ids": query_encoded["input_ids"].squeeze(0),
            "query_mask": query_encoded["attention_mask"].squeeze(0),
            "evidence_ids": evidence_encoded["input_ids"].squeeze(0),
            "evidence_mask": evidence_encoded["attention_mask"].squeeze(0),
            "gold_doc_id": torch.tensor(gold_doc_id, dtype=torch.long),
        }


def collate_evidence_batch(batch: List[dict]) -> dict:
    """
    Evidence 학습용 batch collate function

    Args:
        batch: List of dataset items

    Returns:
        Batched tensors
    """
    query_ids = torch.stack([b["query_ids"] for b in batch])
    query_mask = torch.stack([b["query_mask"] for b in batch])
    evidence_ids = torch.stack([b["evidence_ids"] for b in batch])
    evidence_mask = torch.stack([b["evidence_mask"] for b in batch])
    gold_doc_ids = torch.stack([b["gold_doc_id"] for b in batch])

    return {
        "query_ids": query_ids,
        "query_mask": query_mask,
        "evidence_ids": evidence_ids,
        "evidence_mask": evidence_mask,
        "gold_doc_ids": gold_doc_ids,
    }


def create_evidence_dataloader(
    qa_pairs: List[dict],
    tokenizer: AutoTokenizer,
    batch_size: int = 2,
    max_query_length: int = 128,
    max_evidence_length: int = 256,
    shuffle: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    """
    Evidence 학습용 DataLoader 생성 헬퍼

    Args:
        qa_pairs: [{"query": str, "evidence": str, "gold_doc_id": int}, ...]
        tokenizer: LLM tokenizer
        batch_size: 배치 크기
        max_query_length: 최대 질문 길이
        max_evidence_length: 최대 evidence 길이
        shuffle: 셔플 여부
        num_workers: DataLoader worker 수

    Returns:
        DataLoader instance
    """
    dataset = EvidenceDataset(
        qa_pairs=qa_pairs,
        tokenizer=tokenizer,
        max_query_length=max_query_length,
        max_evidence_length=max_evidence_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_evidence_batch,
    )


# ============================================
# Evidence 추출 유틸리티
# ============================================

def extract_evidence_from_nq(
    document: str,
    answer: str,
    context_window: int = 200,
) -> str:
    """
    Natural Questions에서 evidence 추출

    방법: answer span 주변 context 추출

    Args:
        document: 전체 문서 텍스트
        answer: 정답 텍스트
        context_window: 정답 앞뒤로 가져올 문자 수

    Returns:
        evidence: 추출된 근거 텍스트
    """
    # answer 위치 찾기 (대소문자 무시)
    doc_lower = document.lower()
    answer_lower = answer.lower()
    answer_start = doc_lower.find(answer_lower)

    if answer_start == -1:
        # answer를 찾지 못하면 문서 앞부분 반환
        return document[:500]

    # 주변 context 추출
    start = max(0, answer_start - context_window)
    end = min(len(document), answer_start + len(answer) + context_window)

    # 문장 경계 조정 (가능하면)
    # 시작 위치를 이전 마침표 이후로
    while start > 0 and document[start] not in '.!?\n':
        start -= 1
    if start > 0:
        start += 1  # 마침표 다음부터

    # 끝 위치를 다음 마침표까지
    while end < len(document) and document[end - 1] not in '.!?\n':
        end += 1

    evidence = document[start:end].strip()
    return evidence


def extract_evidence_from_hotpotqa(
    context: List[tuple],
    supporting_facts: List[tuple],
) -> str:
    """
    HotpotQA에서 evidence 추출

    방법: supporting_facts로 지정된 문장들 연결

    Args:
        context: [(title, [sentences...]), ...] 형식
        supporting_facts: [(title, sent_idx), ...] 형식

    Returns:
        evidence: supporting facts 문장들을 연결한 텍스트
    """
    # context를 dict로 변환
    context_dict = {title: sentences for title, sentences in context}

    evidence_sentences = []
    for title, sent_idx in supporting_facts:
        if title in context_dict:
            sentences = context_dict[title]
            if sent_idx < len(sentences):
                evidence_sentences.append(sentences[sent_idx])

    if not evidence_sentences:
        # supporting facts를 찾지 못하면 첫 번째 문서의 첫 문장들
        if context:
            first_doc_sentences = context[0][1][:3]
            return " ".join(first_doc_sentences)
        return ""

    return " ".join(evidence_sentences)


def prepare_evidence_pairs_from_nq(
    dataset: dict,
    corpus: dict,
    max_samples: Optional[int] = None,
) -> List[dict]:
    """
    NQ 데이터셋에서 (query, evidence, gold_doc_id) pairs 생성

    Args:
        dataset: {"qa_pairs": [...], "corpus": {...}}
        corpus: {doc_id: doc_text, ...}
        max_samples: 최대 샘플 수

    Returns:
        List of {"query": str, "evidence": str, "gold_doc_id": int}
    """
    qa_pairs = dataset.get("qa_pairs", dataset)
    if max_samples:
        qa_pairs = qa_pairs[:max_samples]

    evidence_pairs = []
    for item in qa_pairs:
        query = item["question"]
        answer = item["answer"]
        gold_doc_ids = item.get("gold_doc_ids", [])

        # 첫 번째 gold document에서 evidence 추출
        if gold_doc_ids and gold_doc_ids[0] in corpus:
            gold_doc_id = gold_doc_ids[0]
            document = corpus[gold_doc_id]
            evidence = extract_evidence_from_nq(document, answer)
        else:
            # gold document가 없으면 answer 자체를 evidence로
            evidence = f"The answer is {answer}."
            gold_doc_id = -1

        evidence_pairs.append({
            "query": query,
            "evidence": evidence,
            "gold_doc_id": gold_doc_id,
        })

    return evidence_pairs


def prepare_evidence_pairs_from_hotpotqa(
    dataset: List[dict],
    max_samples: Optional[int] = None,
) -> List[dict]:
    """
    HotpotQA 데이터셋에서 (query, evidence, gold_doc_id) pairs 생성

    Args:
        dataset: HotpotQA 데이터셋 (supporting_facts 포함)
        max_samples: 최대 샘플 수

    Returns:
        List of {"query": str, "evidence": str, "gold_doc_id": int}
    """
    if max_samples:
        dataset = dataset[:max_samples]

    evidence_pairs = []
    for item in dataset:
        query = item["question"]
        context = item.get("context", [])
        supporting_facts = item.get("supporting_facts", [])

        evidence = extract_evidence_from_hotpotqa(context, supporting_facts)

        # HotpotQA는 여러 문서가 관련되므로 gold_doc_id는 -1로 설정
        # (또는 별도 매핑 필요)
        evidence_pairs.append({
            "query": query,
            "evidence": evidence,
            "gold_doc_id": -1,
        })

    return evidence_pairs
