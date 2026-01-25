"""
Write Phase 평가
- Reconstruction Perplexity
- ROUGE-L (재생성 품질)
- z_i 정보량 분석
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional
import logging

from .metrics import compute_rouge_l

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_reconstruction(
    model,
    corpus: dict,
    tokenizer,
    num_samples: int = 100,
    max_length: int = 512,
    device: str = "cuda",
) -> dict:
    """
    Write Phase 평가: z_i → D_i reconstruction 품질

    Returns:
        dict with avg_perplexity, avg_loss, avg_rouge_l, per_doc_results
    """
    model.eval()

    doc_ids = sorted(corpus.keys())[:num_samples]
    results = []

    for doc_id in tqdm(doc_ids, desc="Evaluating reconstruction"):
        doc_text = corpus[doc_id]

        # Tokenize
        encoded = tokenizer(
            doc_text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        doc_id_tensor = torch.tensor([doc_id], device=device)

        # Forward: reconstruction loss
        result = model.write_phase_forward(
            doc_ids=doc_id_tensor,
            doc_input_ids=input_ids,
            doc_attention_mask=attention_mask,
        )

        # Generation: z_i → text
        generated_text = generate_from_z(model, doc_id, tokenizer, max_length, device)

        # ROUGE-L
        rouge_l = compute_rouge_l(generated_text, doc_text) if generated_text else 0.0

        results.append({
            "doc_id": doc_id,
            "loss": result["loss"].item(),
            "perplexity": result["perplexity"],
            "rouge_l": rouge_l,
            "generated_preview": generated_text[:200] if generated_text else "",
        })

    # Aggregate
    avg_loss = sum(r["loss"] for r in results) / len(results)
    avg_ppl = sum(r["perplexity"] for r in results) / len(results)
    avg_rouge = sum(r["rouge_l"] for r in results) / len(results)

    summary = {
        "avg_loss": avg_loss,
        "avg_perplexity": avg_ppl,
        "avg_rouge_l": avg_rouge,
        "num_samples": len(results),
        "ppl_below_10": sum(1 for r in results if r["perplexity"] < 10) / len(results),
    }

    logger.info(f"Reconstruction Eval: PPL={avg_ppl:.2f}, "
                f"ROUGE-L={avg_rouge:.4f}, "
                f"PPL<10: {summary['ppl_below_10']*100:.1f}%")

    return summary


@torch.no_grad()
def generate_from_z(
    model,
    doc_id: int,
    tokenizer,
    max_length: int = 256,
    device: str = "cuda",
) -> str:
    """z_i만으로 텍스트 생성 (reconstruction quality 확인용)"""
    model.eval()

    # z_i embeddings
    z_i = model.doc_vectors[doc_id:doc_id+1]  # [1, m_tokens, z_dim]
    z_embeds = model.z_to_embedding(z_i)  # [1, m_tokens, hidden_size]

    # Generate autoregressively
    try:
        generated_ids = model.llm.generate(
            inputs_embeds=z_embeds,
            max_new_tokens=max_length,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    except Exception as e:
        logger.warning(f"Generation failed for doc {doc_id}: {e}")
        generated_text = ""

    return generated_text


@torch.no_grad()
def evaluate_z_quality(
    model,
    corpus: dict,
    num_samples: int = 50,
    device: str = "cuda",
) -> dict:
    """
    z_i 벡터 품질 분석

    - z_i 간 거리 분포
    - z_i norm 분포
    - z_i 유사도 matrix
    """
    import numpy as np

    model.eval()

    doc_ids = sorted(corpus.keys())[:num_samples]
    z_vectors = model.doc_vectors[doc_ids].detach().cpu()  # [N, m, z_dim]

    # Mean pooling
    z_mean = z_vectors.mean(dim=1).numpy()  # [N, z_dim]

    # Norms
    norms = np.linalg.norm(z_mean, axis=1)

    # Pairwise cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(z_mean)

    # Upper triangle (exclude diagonal)
    upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]

    return {
        "z_norm_mean": float(norms.mean()),
        "z_norm_std": float(norms.std()),
        "pairwise_sim_mean": float(upper_tri.mean()),
        "pairwise_sim_std": float(upper_tri.std()),
        "pairwise_sim_max": float(upper_tri.max()),
        "pairwise_sim_min": float(upper_tri.min()),
    }
