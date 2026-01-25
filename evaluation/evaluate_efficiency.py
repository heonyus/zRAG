"""
Efficiency 평가 모듈
- Latency (ms/query)
- Memory (GPU VRAM)
- Storage (MB/1K docs)
- FLOPs 추정
"""

import time
import torch
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_efficiency(
    model,
    tokenizer,
    num_queries: int = 100,
    top_k: int = 5,
    max_new_tokens: int = 64,
    warmup_queries: int = 10,
    device: str = "cuda",
) -> dict:
    """
    Efficiency 종합 평가

    Returns:
        dict with latency_ms, memory_gb, storage_mb_per_1k, throughput
    """
    model.eval()
    results = {}

    # ==========================================
    # 1. Latency 측정
    # ==========================================
    latency_results = measure_latency(
        model, tokenizer, num_queries, top_k,
        max_new_tokens, warmup_queries, device
    )
    results.update(latency_results)

    # ==========================================
    # 2. Memory 측정
    # ==========================================
    memory_results = measure_memory(model, device)
    results.update(memory_results)

    # ==========================================
    # 3. Storage 측정
    # ==========================================
    storage_results = measure_storage(model)
    results.update(storage_results)

    logger.info(f"Efficiency: latency={results['latency_mean_ms']:.1f}ms, "
                f"memory={results['vram_gb']:.2f}GB, "
                f"storage={results['storage_mb_per_1k']:.2f}MB/1K")

    return results


def measure_latency(
    model,
    tokenizer,
    num_queries: int = 100,
    top_k: int = 5,
    max_new_tokens: int = 64,
    warmup_queries: int = 10,
    device: str = "cuda",
) -> dict:
    """Inference latency 측정"""
    model.eval()

    # Dummy queries
    dummy_text = "What is the capital of France?"
    encoded = tokenizer(
        dummy_text,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    query_ids = encoded["input_ids"].to(device)
    query_mask = encoded["attention_mask"].to(device)

    # Warmup
    for _ in range(warmup_queries):
        selected_ids, _ = model.select_documents(query_ids, query_mask, k=top_k)
        _ = model.generate(
            query_ids=query_ids,
            doc_indices=selected_ids,
            query_attention_mask=query_mask,
            max_new_tokens=max_new_tokens,
        )

    # Measure
    selection_times = []
    generation_times = []
    total_times = []

    if device == "cuda":
        torch.cuda.synchronize()

    for _ in range(num_queries):
        # Selection latency
        start = time.perf_counter()
        if device == "cuda":
            torch.cuda.synchronize()
        selected_ids, _ = model.select_documents(query_ids, query_mask, k=top_k)
        if device == "cuda":
            torch.cuda.synchronize()
        selection_time = (time.perf_counter() - start) * 1000
        selection_times.append(selection_time)

        # Generation latency
        start = time.perf_counter()
        _ = model.generate(
            query_ids=query_ids,
            doc_indices=selected_ids,
            query_attention_mask=query_mask,
            max_new_tokens=max_new_tokens,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        generation_time = (time.perf_counter() - start) * 1000
        generation_times.append(generation_time)

        total_times.append(selection_time + generation_time)

    return {
        "latency_mean_ms": float(np.mean(total_times)),
        "latency_std_ms": float(np.std(total_times)),
        "latency_p50_ms": float(np.percentile(total_times, 50)),
        "latency_p95_ms": float(np.percentile(total_times, 95)),
        "selection_latency_ms": float(np.mean(selection_times)),
        "generation_latency_ms": float(np.mean(generation_times)),
        "throughput_qps": 1000.0 / float(np.mean(total_times)),
    }


def measure_memory(model, device: str = "cuda") -> dict:
    """GPU VRAM 사용량 측정"""
    if device != "cuda" or not torch.cuda.is_available():
        return {"vram_gb": 0.0, "vram_allocated_gb": 0.0}

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Current allocation
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)

    return {
        "vram_gb": allocated,
        "vram_reserved_gb": reserved,
        "vram_peak_gb": peak,
    }


def measure_storage(model) -> dict:
    """z_i 저장 크기 측정"""
    num_docs = model.doc_vectors.size(0)
    z_size_bytes = model.doc_vectors.nelement() * model.doc_vectors.element_size()
    z_size_mb = z_size_bytes / (1024 ** 2)

    # Per 1K docs
    storage_per_1k = z_size_mb / (num_docs / 1000) if num_docs > 0 else 0

    # Projection layer size
    proj_size_bytes = sum(
        p.nelement() * p.element_size()
        for p in model.z_to_embedding.parameters()
    )
    proj_size_mb = proj_size_bytes / (1024 ** 2)

    # LoRA weight size
    lora_size_bytes = sum(
        p.nelement() * p.element_size()
        for name, p in model.llm.named_parameters()
        if p.requires_grad
    )
    lora_size_mb = lora_size_bytes / (1024 ** 2)

    return {
        "storage_mb_per_1k": storage_per_1k,
        "z_total_mb": z_size_mb,
        "z_per_doc_bytes": z_size_bytes / max(num_docs, 1),
        "projection_mb": proj_size_mb,
        "lora_mb": lora_size_mb,
        "total_model_mb": z_size_mb + proj_size_mb + lora_size_mb,
        "num_docs": num_docs,
        "compression_ratio": _estimate_compression_ratio(model),
    }


def _estimate_compression_ratio(model) -> float:
    """압축률 추정: 원본 텍스트 대비 z_i 크기"""
    # 가정: 평균 문서 길이 512 tokens × 2 bytes/token = 1024 bytes
    avg_doc_bytes = 512 * 2  # 대략적인 원본 크기

    z_per_doc = (model.doc_vectors.size(1) * model.doc_vectors.size(2) *
                 model.doc_vectors.element_size())

    if z_per_doc == 0:
        return 0.0

    return avg_doc_bytes / z_per_doc


def compare_with_rag(
    pqa_results: dict,
    rag_corpus_size: int = 50000,
    rag_embedding_dim: int = 768,
    rag_avg_doc_tokens: int = 100,
) -> dict:
    """Standard RAG와의 efficiency 비교"""
    # RAG storage: embeddings + text
    rag_embedding_storage = rag_corpus_size * rag_embedding_dim * 4 / (1024 ** 2)  # fp32
    rag_text_storage = rag_corpus_size * rag_avg_doc_tokens * 4 / (1024 ** 2)  # 4 bytes/token avg
    rag_total_storage_mb = rag_embedding_storage + rag_text_storage

    # PQA storage
    pqa_storage_mb = pqa_results.get("z_total_mb", 0) + pqa_results.get("projection_mb", 0)

    # Comparison
    return {
        "rag_storage_mb": rag_total_storage_mb,
        "pqa_storage_mb": pqa_storage_mb,
        "storage_reduction_x": rag_total_storage_mb / max(pqa_storage_mb, 1e-6),
        "rag_storage_per_1k": rag_total_storage_mb / (rag_corpus_size / 1000),
        "pqa_storage_per_1k": pqa_results.get("storage_mb_per_1k", 0),
    }
