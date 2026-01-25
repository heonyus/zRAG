"""
Standard RAG Baseline
- BM25 또는 Dense retrieval (E5-base-v2) + LLM
- FlashRAG 표준 설정 준수
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, BitsAndBytesConfig
from tqdm import tqdm
from typing import List, Optional
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from evaluation.metrics import (
    compute_em, compute_f1, compute_recall_at_k,
    compute_mrr, aggregate_metrics,
)

logger = logging.getLogger(__name__)

class StandardRAGBaseline:
    """
    Standard RAG: Retrieve text passages + LLM generation

    Retriever options:
    - bm25: BM25 sparse retrieval
    - dense: E5-base-v2 dense retrieval
    """

    def __init__(
        self,
        llm_name: str = "Qwen/Qwen3-8B",
        retriever_type: str = "dense",
        retriever_name: str = "intfloat/e5-base-v2",
        quantization: str = "4bit",
        device: str = "cuda",
    ):
        self.device = device
        self.retriever_type = retriever_type

        # LLM
        quant_config = None
        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.llm.eval()

        # Retriever
        if retriever_type == "dense":
            self.retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_name)
            self.retriever_model = AutoModel.from_pretrained(retriever_name).to(device)
            self.retriever_model.eval()
        elif retriever_type == "bm25":
            self.bm25 = None  # Initialized in build_index

        # Index
        self.corpus = None
        self.doc_embeddings = None
        self.doc_ids = None

    @torch.no_grad()
    def build_index(self, corpus: dict, batch_size: int = 32):
        """코퍼스 인덱스 구축"""
        self.corpus = corpus
        self.doc_ids = sorted(corpus.keys())
        doc_texts = [corpus[did] for did in self.doc_ids]

        if self.retriever_type == "dense":
            self._build_dense_index(doc_texts, batch_size)
        elif self.retriever_type == "bm25":
            self._build_bm25_index(doc_texts)

        logger.info(f"Index built: {len(self.doc_ids)} documents, type={self.retriever_type}")

    def _build_dense_index(self, doc_texts: List[str], batch_size: int):
        """Dense embedding index 구축"""
        all_embeddings = []

        for i in range(0, len(doc_texts), batch_size):
            batch_texts = doc_texts[i:i+batch_size]
            # E5 format: "passage: " prefix
            batch_texts = [f"passage: {t}" for t in batch_texts]

            encoded = self.retriever_tokenizer(
                batch_texts,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.retriever_model(**encoded)
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            all_embeddings.append(embeddings.cpu())

        self.doc_embeddings = torch.cat(all_embeddings, dim=0)  # [num_docs, hidden]

    def _build_bm25_index(self, doc_texts: List[str]):
        """BM25 index 구축"""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.error("rank_bm25 not installed. pip install rank-bm25")
            raise

        tokenized_corpus = [doc.lower().split() for doc in doc_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    @torch.no_grad()
    def retrieve(self, query: str, k: int = 5) -> List[dict]:
        """Top-k 문서 검색"""
        if self.retriever_type == "dense":
            return self._dense_retrieve(query, k)
        elif self.retriever_type == "bm25":
            return self._bm25_retrieve(query, k)

    def _dense_retrieve(self, query: str, k: int) -> List[dict]:
        """Dense retrieval"""
        # E5 format
        query_text = f"query: {query}"
        encoded = self.retriever_tokenizer(
            query_text,
            max_length=128,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.retriever_model(**encoded)
        q_embedding = outputs.last_hidden_state[:, 0, :]
        q_embedding = torch.nn.functional.normalize(q_embedding, p=2, dim=-1)

        # Similarity
        scores = torch.matmul(q_embedding.cpu(), self.doc_embeddings.t()).squeeze(0)
        top_k = torch.topk(scores, k)

        results = []
        for idx, score in zip(top_k.indices.tolist(), top_k.values.tolist()):
            doc_id = self.doc_ids[idx]
            results.append({
                "doc_id": doc_id,
                "text": self.corpus[doc_id],
                "score": score,
            })
        return results

    def _bm25_retrieve(self, query: str, k: int) -> List[dict]:
        """BM25 retrieval"""
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        top_k_indices = np.argsort(scores)[-k:][::-1]

        results = []
        for idx in top_k_indices:
            doc_id = self.doc_ids[idx]
            results.append({
                "doc_id": doc_id,
                "text": self.corpus[doc_id],
                "score": float(scores[idx]),
            })
        return results

    def build_prompt(self, question: str, contexts: List[str]) -> str:
        """RAG prompt 구성"""
        context_str = "\n\n".join([f"Document {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
        return (
            f"Answer the question based on the given documents.\n\n"
            f"{context_str}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )

    @torch.no_grad()
    def generate(self, question: str, k: int = 5, max_new_tokens: int = 64) -> dict:
        """검색 + 생성"""
        # Retrieve
        retrieved = self.retrieve(question, k=k)
        contexts = [r["text"] for r in retrieved]
        retrieved_ids = [r["doc_id"] for r in retrieved]

        # Generate
        prompt = self.build_prompt(question, contexts)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=2048,
            truncation=True,
        ).to(self.device)

        outputs = self.llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        new_tokens = outputs[0][inputs["input_ids"].size(1):]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return {
            "answer": answer,
            "retrieved_ids": retrieved_ids,
            "contexts": contexts,
        }

    @torch.no_grad()
    def evaluate(
        self,
        qa_pairs: list,
        k: int = 5,
        max_samples: Optional[int] = None,
        max_new_tokens: int = 64,
    ) -> dict:
        """전체 QA pairs 평가"""
        if max_samples:
            qa_pairs = qa_pairs[:max_samples]

        all_metrics = []

        for item in tqdm(qa_pairs, desc=f"RAG ({self.retriever_type})"):
            question = item["question"]
            answer = item["answer"]
            gold_doc_ids = item["gold_doc_ids"]

            result = self.generate(question, k=k, max_new_tokens=max_new_tokens)

            em = compute_em(result["answer"], answer)
            f1 = compute_f1(result["answer"], answer)
            recall = compute_recall_at_k(result["retrieved_ids"], gold_doc_ids, k=k)
            mrr = compute_mrr(result["retrieved_ids"], gold_doc_ids)

            all_metrics.append({
                "em": em,
                "f1": f1,
                "recall_at_k": recall,
                "mrr": mrr,
            })

        result = aggregate_metrics(all_metrics)
        result["num_samples"] = len(all_metrics)
        result["method"] = f"rag_{self.retriever_type}"

        logger.info(f"RAG ({self.retriever_type}): EM={result['em']:.4f}, "
                    f"F1={result['f1']:.4f}, Recall@{k}={result['recall_at_k']:.4f}")
        return result

    # ============================================
    # Evidence 비교용 메서드 (LLM-as-Memory 비교)
    # ============================================

    @torch.no_grad()
    def get_retrieved_as_evidence(
        self,
        queries: List[str],
        k: int = 1,
        join_docs: bool = True,
    ) -> List[str]:
        """
        RAG retrieved text를 evidence로 반환 (LLM-as-Memory와 비교용)

        Args:
            queries: 질문 리스트
            k: 검색할 문서 수
            join_docs: 여러 문서를 하나로 합칠지 여부

        Returns:
            List of retrieved text (as evidence)
        """
        retrieved_texts = []

        for query in tqdm(queries, desc="Retrieving for comparison"):
            results = self.retrieve(query, k=k)
            if join_docs:
                # 여러 문서를 공백으로 연결
                text = " ".join([r["text"] for r in results])
            else:
                # 첫 번째 문서만
                text = results[0]["text"] if results else ""
            retrieved_texts.append(text)

        return retrieved_texts

    @torch.no_grad()
    def evaluate_as_evidence(
        self,
        evidence_pairs: List[dict],
        k: int = 1,
        max_samples: Optional[int] = None,
    ) -> dict:
        """
        RAG retrieved text를 evidence로 평가 (LLM-as-Memory와 동일 메트릭)

        Args:
            evidence_pairs: [{
                "query": str,
                "evidence": str (gold evidence),
                "answer": str (optional),
            }, ...]
            k: 검색할 문서 수

        Returns:
            dict with evidence quality metrics
        """
        from evaluation.evidence_metrics import evaluate_evidence_quality

        if max_samples:
            evidence_pairs = evidence_pairs[:max_samples]

        all_metrics = []

        for item in tqdm(evidence_pairs, desc=f"RAG Evidence ({self.retriever_type})"):
            query = item["query"]
            gold_evidence = item["evidence"]
            answer = item.get("answer", "")

            # Retrieve
            results = self.retrieve(query, k=k)
            retrieved_text = " ".join([r["text"] for r in results])

            # Evaluate
            metrics = evaluate_evidence_quality(
                generated_evidence=retrieved_text,
                gold_evidence=gold_evidence,
                answer=answer,
            )
            all_metrics.append(metrics)

        # Aggregate
        if not all_metrics:
            return {"num_samples": 0}

        aggregated = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            if values:
                aggregated[key] = sum(values) / len(values)

        aggregated["num_samples"] = len(all_metrics)
        aggregated["method"] = f"rag_{self.retriever_type}"

        logger.info(
            f"RAG Evidence ({self.retriever_type}): "
            f"ROUGE-L={aggregated.get('rouge_l_f1', 0):.4f}, "
            f"AnswerCov={aggregated.get('answer_coverage', 0):.4f}"
        )

        return aggregated
