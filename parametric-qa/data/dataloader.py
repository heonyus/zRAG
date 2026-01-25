"""
PyTorch DataLoader 모듈
- WritePhaseDataset: z_i 학습용 (doc_id, doc_text)
- ReadPhaseDataset: QA 학습용 (question, answer, gold_doc_ids)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Optional


class WritePhaseDataset(Dataset):
    """
    Write Phase 학습용 Dataset
    각 샘플: (doc_id, doc_token_ids)
    """

    def __init__(
        self,
        corpus: dict,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_length = max_length

        # corpus를 list로 변환 (순서 보장)
        self.doc_ids = sorted(corpus.keys())
        self.doc_texts = [corpus[did] for did in self.doc_ids]

    def __len__(self):
        return len(self.doc_ids)

    def __getitem__(self, idx):
        doc_id = self.doc_ids[idx]
        doc_text = self.doc_texts[idx]

        # Tokenize
        encoded = self.tokenizer(
            doc_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "doc_id": doc_id,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


class ReadPhaseDataset(Dataset):
    """
    Read Phase 학습용 Dataset
    각 샘플: (question_ids, answer_ids, gold_doc_ids)
    """

    def __init__(
        self,
        qa_pairs: list,
        tokenizer: AutoTokenizer,
        max_query_length: int = 128,
        max_answer_length: int = 64,
    ):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_answer_length = max_answer_length

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        item = self.qa_pairs[idx]
        question = item["question"]
        answer = item["answer"]
        gold_doc_ids = item["gold_doc_ids"]

        # Tokenize question
        q_encoded = self.tokenizer(
            question,
            max_length=self.max_query_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize answer
        a_encoded = self.tokenizer(
            answer,
            max_length=self.max_answer_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "question_ids": q_encoded["input_ids"].squeeze(0),
            "question_mask": q_encoded["attention_mask"].squeeze(0),
            "answer_ids": a_encoded["input_ids"].squeeze(0),
            "answer_mask": a_encoded["attention_mask"].squeeze(0),
            "gold_doc_ids": torch.tensor(gold_doc_ids, dtype=torch.long),
        }


def collate_read_phase(batch):
    """
    Read Phase용 custom collate function
    gold_doc_ids의 길이가 다를 수 있으므로 처리
    """
    question_ids = torch.stack([b["question_ids"] for b in batch])
    question_mask = torch.stack([b["question_mask"] for b in batch])
    answer_ids = torch.stack([b["answer_ids"] for b in batch])
    answer_mask = torch.stack([b["answer_mask"] for b in batch])

    # gold_doc_ids: pad to max length in batch
    max_gold = max(len(b["gold_doc_ids"]) for b in batch)
    gold_doc_ids = torch.full((len(batch), max_gold), -1, dtype=torch.long)
    for i, b in enumerate(batch):
        gold_doc_ids[i, :len(b["gold_doc_ids"])] = b["gold_doc_ids"]

    return {
        "question_ids": question_ids,
        "question_mask": question_mask,
        "answer_ids": answer_ids,
        "answer_mask": answer_mask,
        "gold_doc_ids": gold_doc_ids,
    }


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 2,
    collate_fn=None,
) -> DataLoader:
    """범용 DataLoader 생성"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def create_write_dataloader(
    corpus: dict,
    tokenizer: AutoTokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    shuffle: bool = True,
) -> DataLoader:
    """Write Phase DataLoader 생성 헬퍼"""
    dataset = WritePhaseDataset(corpus, tokenizer, max_length)
    return get_dataloader(dataset, batch_size=batch_size, shuffle=shuffle)


def create_read_dataloader(
    qa_pairs: list,
    tokenizer: AutoTokenizer,
    batch_size: int = 2,
    max_query_length: int = 128,
    max_answer_length: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """Read Phase DataLoader 생성 헬퍼"""
    dataset = ReadPhaseDataset(qa_pairs, tokenizer, max_query_length, max_answer_length)
    return get_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_read_phase,
    )
