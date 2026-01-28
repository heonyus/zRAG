"""
Contriever RAG Baseline

Facebook Contriever를 사용한 Dense Retrieval + LLM

Contriever는 unsupervised contrastive learning으로 학습된 retriever로,
E5와 함께 대표적인 dense retriever baseline입니다.

참고: https://github.com/facebookresearch/contriever
"""

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Optional
from tqdm import tqdm
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from baselines.standard_rag import StandardRAGBaseline

logger = logging.getLogger(__name__)


class ContrieverRetriever:
    """
    Contriever 기반 Dense Retriever

    특징:
    - Mean pooling (CLS 대신)
    - Unsupervised contrastive learning
    - No query/passage prefix needed
    """

    def __init__(
        self,
        model_name: str = "facebook/contriever",
        device: str = "cuda",
    ):
        self.device = device
        self.model_name = model_name

        logger.info(f"Loading Contriever: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

        # Index
        self.doc_embeddings = None
        self.doc_ids = None
        self.corpus = None

        logger.info("Contriever loaded")

    def mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean pooling (Contriever 방식)"""
        # attention_mask를 확장하여 token_embeddings와 동일한 shape으로
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # 마스크된 토큰의 합
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        # 마스크된 토큰 수
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> torch.Tensor:
        """
        텍스트를 임베딩으로 변환

        Args:
            texts: 텍스트 리스트
            batch_size: 배치 크기
            show_progress: 진행률 표시

        Returns:
            [num_texts, hidden_size] 텐서
        """
        all_embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding")

        for i in iterator:
            batch_texts = texts[i:i+batch_size]

            encoded = self.tokenizer(
                batch_texts,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**encoded)

            # Mean pooling
            embeddings = self.mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def build_index(self, corpus: Dict[str, str], batch_size: int = 32):
        """
        Corpus 인덱스 구축

        Args:
            corpus: {doc_id: text} 딕셔너리
            batch_size: 배치 크기
        """
        self.corpus = corpus
        self.doc_ids = sorted(corpus.keys())
        doc_texts = [corpus[did] for did in self.doc_ids]

        logger.info(f"Building Contriever index for {len(self.doc_ids)} documents...")

        self.doc_embeddings = self.encode(doc_texts, batch_size=batch_size, show_progress=True)

        logger.info(f"Index built: {self.doc_embeddings.shape}")

    @torch.no_grad()
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Top-k 문서 검색

        Args:
            query: 질문
            k: 검색할 문서 수

        Returns:
            [{doc_id, text, score}, ...] 리스트
        """
        # Query 임베딩
        q_embedding = self.encode([query])  # [1, hidden]

        # 코사인 유사도
        scores = torch.matmul(q_embedding, self.doc_embeddings.t()).squeeze(0)

        # Top-k
        top_k = torch.topk(scores, min(k, len(self.doc_ids)))

        results = []
        for idx, score in zip(top_k.indices.tolist(), top_k.values.tolist()):
            doc_id = self.doc_ids[idx]
            results.append({
                "doc_id": doc_id,
                "text": self.corpus[doc_id],
                "score": score,
            })

        return results


class ContrieverRAGBaseline(StandardRAGBaseline):
    """
    Contriever를 사용한 RAG Baseline

    StandardRAGBaseline을 상속받아 retriever만 Contriever로 교체
    """

    def __init__(
        self,
        llm_name: str = "Qwen/Qwen3-8B",
        retriever_name: str = "facebook/contriever",
        quantization: str = "4bit",
        device: str = "cuda",
    ):
        # Parent init 호출하지 않고 직접 초기화
        # (StandardRAGBaseline의 init이 retriever를 다르게 초기화하므로)

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.device = device
        self.retriever_type = "contriever"

        # LLM
        quant_config = None
        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        logger.info(f"Loading LLM: {llm_name}")

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

        # Contriever Retriever
        self.retriever = ContrieverRetriever(
            model_name=retriever_name,
            device=device,
        )

        # Index
        self.corpus = None
        self.doc_ids = None

    def build_index(self, corpus: Dict[str, str], batch_size: int = 32):
        """Contriever 인덱스 구축"""
        self.corpus = corpus
        self.doc_ids = sorted(corpus.keys())
        self.retriever.build_index(corpus, batch_size=batch_size)
        logger.info(f"Contriever index built: {len(self.doc_ids)} documents")

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Contriever로 검색"""
        return self.retriever.retrieve(query, k)

    # build_prompt, generate, evaluate는 StandardRAGBaseline에서 상속


def create_contriever_baseline(
    llm_name: str = "Qwen/Qwen3-8B",
    quantization: str = "4bit",
    device: str = "cuda",
) -> ContrieverRAGBaseline:
    """Contriever baseline 생성 헬퍼"""
    return ContrieverRAGBaseline(
        llm_name=llm_name,
        quantization=quantization,
        device=device,
    )


if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.INFO)

    print("Testing ContrieverRetriever...")

    # Retriever만 테스트
    retriever = ContrieverRetriever()

    # 샘플 corpus
    corpus = {
        "doc1": "Paris is the capital of France. It is known for the Eiffel Tower.",
        "doc2": "Berlin is the capital of Germany. It has a famous wall.",
        "doc3": "Tokyo is the capital of Japan. It is a very large city.",
        "doc4": "The Eiffel Tower was built in 1889 for the World's Fair.",
    }

    # 인덱스 구축
    retriever.build_index(corpus)

    # 검색 테스트
    query = "What is the capital of France?"
    results = retriever.retrieve(query, k=2)

    print(f"\nQuery: {query}")
    print("Results:")
    for r in results:
        print(f"  [{r['score']:.4f}] {r['doc_id']}: {r['text'][:50]}...")
