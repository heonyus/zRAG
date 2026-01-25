"""
데이터 전처리 모듈
- NQ/HotpotQA를 (question, answer, gold_doc_ids, evidence) 형태로 변환
- Corpus를 document pool로 구성
"""

import json
import random
from pathlib import Path
from typing import Optional

from datasets import Dataset


def preprocess_nq(dataset, corpus_size: int = 5000, max_samples: Optional[int] = None) -> dict:
    """
    Natural Questions 전처리

    Returns:
        {
            "qa_pairs": [(question, answer, gold_doc_id, evidence), ...],
            "corpus": {doc_id: doc_text, ...},
            "doc_id_map": {original_id: new_id, ...}
        }
    """
    corpus = {}
    qa_pairs = []
    doc_id_counter = 0

    # FlashRAG format 처리
    if "question" in dataset.column_names.get("train", dataset.column_names.get("test", [])):
        return _preprocess_nq_flashrag(dataset, corpus_size, max_samples)

    # Original NQ format 처리
    return _preprocess_nq_original(dataset, corpus_size, max_samples)


def _preprocess_nq_flashrag(dataset, corpus_size: int, max_samples: Optional[int]) -> dict:
    """FlashRAG 형식 NQ 전처리"""
    corpus = {}
    qa_pairs = []
    doc_id_counter = 0
    text_to_id = {}  # deduplication

    split = "test" if "test" in dataset else "validation" if "validation" in dataset else "train"
    data = dataset[split]

    if max_samples:
        indices = list(range(min(max_samples, len(data))))
        data = data.select(indices)

    for item in data:
        question = item["question"]
        answer = item.get("golden_answers", item.get("answer", [""]))[0] \
            if isinstance(item.get("golden_answers", item.get("answer", "")), list) \
            else item.get("golden_answers", item.get("answer", ""))

        # Gold context (evidence)
        gold_contexts = item.get("golden_contexts", item.get("contexts", []))
        if not gold_contexts:
            continue

        gold_doc_ids = []
        evidence_texts = []

        for ctx in gold_contexts[:3]:  # 최대 3개 gold context
            ctx_text = ctx if isinstance(ctx, str) else ctx.get("text", "")
            if not ctx_text:
                continue

            if ctx_text in text_to_id:
                doc_id = text_to_id[ctx_text]
            else:
                doc_id = doc_id_counter
                corpus[doc_id] = ctx_text
                text_to_id[ctx_text] = doc_id
                doc_id_counter += 1

            gold_doc_ids.append(doc_id)
            evidence_texts.append(ctx_text)

        if gold_doc_ids and answer:
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "gold_doc_ids": gold_doc_ids,
                "evidence": evidence_texts[0] if evidence_texts else "",
            })

        if len(corpus) >= corpus_size:
            break

    # corpus_size까지 패딩 (distractor documents)
    if len(corpus) < corpus_size:
        corpus = _pad_corpus_from_dataset(corpus, dataset, corpus_size, text_to_id)

    return {
        "qa_pairs": qa_pairs,
        "corpus": corpus,
        "stats": {
            "num_qa": len(qa_pairs),
            "corpus_size": len(corpus),
            "avg_doc_length": sum(len(d.split()) for d in corpus.values()) / max(len(corpus), 1),
        }
    }


def _preprocess_nq_original(dataset, corpus_size: int, max_samples: Optional[int]) -> dict:
    """Original NQ format 전처리"""
    corpus = {}
    qa_pairs = []
    doc_id_counter = 0
    text_to_id = {}

    data = dataset["train"] if "train" in dataset else dataset["validation"]
    if max_samples:
        indices = list(range(min(max_samples, len(data))))
        data = data.select(indices)

    for item in data:
        question = item["question"]["text"]

        # Short answer 추출
        annotations = item["annotations"]
        if not annotations["short_answers"] or not annotations["short_answers"][0]:
            continue

        short_answer = annotations["short_answers"][0]
        if isinstance(short_answer, dict) and "text" in short_answer:
            answer = short_answer["text"][0] if short_answer["text"] else ""
        else:
            continue

        if not answer:
            continue

        # Document (Wikipedia page)
        doc_text = item["document"]["tokens"]["token"]
        doc_text = " ".join(doc_text[:512])  # 최대 512 tokens

        if doc_text not in text_to_id:
            text_to_id[doc_text] = doc_id_counter
            corpus[doc_id_counter] = doc_text
            doc_id_counter += 1

        doc_id = text_to_id[doc_text]

        qa_pairs.append({
            "question": question,
            "answer": answer,
            "gold_doc_ids": [doc_id],
            "evidence": doc_text,
        })

        if len(corpus) >= corpus_size:
            break

    return {
        "qa_pairs": qa_pairs,
        "corpus": corpus,
        "stats": {
            "num_qa": len(qa_pairs),
            "corpus_size": len(corpus),
        }
    }


def preprocess_hotpotqa(dataset, corpus_size: int = 50000, max_samples: Optional[int] = None) -> dict:
    """
    HotpotQA 전처리 (distractor setting)
    각 샘플에 10개 context paragraph (2 gold + 8 distractor)
    """
    corpus = {}
    qa_pairs = []
    doc_id_counter = 0
    text_to_id = {}

    data = dataset["validation"]  # dev set 사용 (test labels unavailable)
    if max_samples:
        indices = list(range(min(max_samples, len(data))))
        data = data.select(indices)

    for item in data:
        question = item["question"]
        answer = item["answer"]
        supporting_facts_titles = set(item["supporting_facts"]["title"])

        gold_doc_ids = []
        all_doc_ids = []

        # Context paragraphs
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            doc_text = " ".join(sentences)
            if not doc_text.strip():
                continue

            if doc_text not in text_to_id:
                text_to_id[doc_text] = doc_id_counter
                corpus[doc_id_counter] = doc_text
                doc_id_counter += 1

            doc_id = text_to_id[doc_text]
            all_doc_ids.append(doc_id)

            if title in supporting_facts_titles:
                gold_doc_ids.append(doc_id)

        if gold_doc_ids and answer:
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "gold_doc_ids": gold_doc_ids,
                "all_doc_ids": all_doc_ids,
                "supporting_facts": item["supporting_facts"],
                "type": item["type"],  # bridge or comparison
                "level": item["level"],
            })

        if len(corpus) >= corpus_size:
            break

    return {
        "qa_pairs": qa_pairs,
        "corpus": corpus,
        "stats": {
            "num_qa": len(qa_pairs),
            "corpus_size": len(corpus),
            "bridge_count": sum(1 for q in qa_pairs if q["type"] == "bridge"),
            "comparison_count": sum(1 for q in qa_pairs if q["type"] == "comparison"),
        }
    }


def build_corpus(corpus_dict: dict, save_path: Optional[str] = None) -> dict:
    """corpus dict를 저장 가능한 형태로 구성"""
    corpus_data = {
        "documents": [],
        "id_to_idx": {},
    }

    for doc_id, doc_text in corpus_dict.items():
        corpus_data["documents"].append({
            "id": doc_id,
            "text": doc_text,
            "length": len(doc_text.split()),
        })
        corpus_data["id_to_idx"][doc_id] = len(corpus_data["documents"]) - 1

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(corpus_data, f, ensure_ascii=False, indent=2)
        print(f"[Corpus] Saved to {path}: {len(corpus_data['documents'])} documents")

    return corpus_data


def _pad_corpus_from_dataset(corpus: dict, dataset, target_size: int, text_to_id: dict) -> dict:
    """corpus를 target_size까지 distractor 문서로 패딩"""
    doc_id_counter = max(corpus.keys()) + 1 if corpus else 0
    split = "train" if "train" in dataset else list(dataset.keys())[0]

    for item in dataset[split]:
        if len(corpus) >= target_size:
            break

        # 데이터셋에서 추가 context 추출
        contexts = item.get("golden_contexts", item.get("contexts", []))
        for ctx in contexts:
            ctx_text = ctx if isinstance(ctx, str) else ctx.get("text", "")
            if ctx_text and ctx_text not in text_to_id:
                corpus[doc_id_counter] = ctx_text
                text_to_id[ctx_text] = doc_id_counter
                doc_id_counter += 1

                if len(corpus) >= target_size:
                    break

    return corpus


if __name__ == "__main__":
    from download import download_dataset

    # NQ 전처리 테스트
    dataset = download_dataset("natural_questions")
    result = preprocess_nq(dataset, corpus_size=1000, max_samples=500)
    print(f"NQ Stats: {result['stats']}")

    # HotpotQA 전처리 테스트
    dataset = download_dataset("hotpot_qa")
    result = preprocess_hotpotqa(dataset, corpus_size=5000, max_samples=500)
    print(f"HotpotQA Stats: {result['stats']}")
