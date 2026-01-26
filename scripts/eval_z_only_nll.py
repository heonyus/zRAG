"""
Z-Only NLL Evaluation
- doc 입력 없이 z만으로 문서를 예측하는 NLL 측정
- 현재 학습된 z_pool/projection이 실제로 문서 정보를 담고 있는지 확인

판정 기준:
- z-only NLL이 8~11: z가 문서 정보를 담지 않음 (unconditional prior 수준)
- z-only NLL이 2~4: z가 문서 정보를 담고 있음 (conditioning 작동)
"""

import sys
sys.path.insert(0, "/home/lhe339/data/zRAG")

import torch
import torch.nn.functional as F
from torch.amp import autocast
from pathlib import Path
from omegaconf import OmegaConf
from datasets import load_dataset
from models.write_phase_model import WritePhaseModel, ZPoolManager
from training.train_write_phase import prepare_corpus


def compute_z_only_nll(model, z_i, doc_ids):
    """
    z만으로 doc을 예측하는 NLL 계산

    Args:
        model: WritePhaseModel
        z_i: [m_tokens, z_dim] learned z vector
        doc_ids: [1, doc_len] document token ids

    Returns:
        nll: negative log likelihood (per token)
    """
    # z를 embedding space로 projection
    alpha_clamped = torch.clamp(model.alpha, min=0.5)
    z_embed = alpha_clamped * model.z_to_embedding(z_i)  # [m_tokens, hidden]
    z_embed = z_embed.unsqueeze(0)  # [1, m_tokens, hidden]

    m_tokens = z_embed.shape[1]
    doc_len = doc_ids.shape[1]

    # z_embed만 입력으로 사용 (doc_embed 없음!)
    inputs_embeds = z_embed  # [1, m_tokens, hidden]

    # attention mask: z tokens만
    attention_mask = torch.ones(1, m_tokens, device=z_embed.device)

    # LLM forward
    outputs = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
    )

    # logits: [1, m_tokens, vocab_size]
    # z의 마지막 토큰 → doc[0] 예측
    # z[-2] → doc[0], z[-1] → doc[1], ...
    # 실제로는 m_tokens개의 logits으로 m_tokens개의 doc 토큰만 예측 가능
    # 전체 doc을 예측하려면 autoregressive하게 해야 하지만,
    # 여기서는 "z가 doc 시작을 예측할 수 있는가"를 측정

    # 간단한 방법: z의 마지막 위치에서 doc 첫 토큰 예측
    last_logit = outputs.logits[0, -1, :]  # [vocab_size]
    first_doc_token = doc_ids[0, 0]  # scalar

    nll_first = F.cross_entropy(last_logit.unsqueeze(0), first_doc_token.unsqueeze(0))

    # 더 정확한 방법: autoregressive로 전체 doc NLL 측정
    # doc 토큰을 하나씩 붙여가며 loss 계산
    total_nll = 0.0
    num_tokens = min(doc_len, 50)  # 처음 50토큰만 (속도 위해)

    current_embeds = z_embed.clone()

    for i in range(num_tokens):
        # forward
        outputs = model.llm(
            inputs_embeds=current_embeds,
            use_cache=False,
        )

        # 마지막 위치에서 다음 토큰 예측
        logits = outputs.logits[0, -1, :]  # [vocab_size]
        target = doc_ids[0, i]

        nll = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
        total_nll += nll.item()

        # 다음 토큰 embedding 추가
        next_embed = model.llm.get_input_embeddings()(doc_ids[0, i:i+1]).unsqueeze(0)  # [1, 1, hidden]
        current_embeds = torch.cat([current_embeds, next_embed], dim=1)

    avg_nll = total_nll / num_tokens

    return {
        "nll_first_token": nll_first.item(),
        "nll_avg_50": avg_nll,
        "num_tokens": num_tokens,
    }


def main():
    print("=" * 60)
    print("Z-Only NLL Evaluation")
    print("=" * 60)
    print("목적: z만으로 문서를 예측하는 NLL 측정")
    print("      (doc 컨텍스트 없이 순수하게 z의 정보량 확인)")

    # 1. Load config & model
    config_path = "/home/lhe339/data/zRAG/configs/phase1_write.yaml"
    config = OmegaConf.load(config_path)

    print("\n[1] Loading model...")
    model = WritePhaseModel(
        llm_name=config.model.llm_name,
        m_tokens=config.memory.m_tokens,
        z_dim=config.memory.z_dim,
        quantization=config.model.get("quantization", "4bit"),
    )

    # Load trained projection
    proj_path = Path(config.logging.save_dir) / "projection.pt"
    if proj_path.exists():
        model.load_projection(proj_path)
        print(f"  Loaded projection from {proj_path}")

    # Load z_pool
    z_pool_path = Path(config.logging.save_dir) / "z_pool.pt"
    z_pool = ZPoolManager(m_tokens=config.memory.m_tokens, z_dim=config.memory.z_dim)
    z_pool.load(z_pool_path)
    print(f"  Loaded z_pool: {len(z_pool.doc_ids)} documents")
    print(f"  Alpha value: {model.alpha.item():.4f}")

    # Load corpus
    dataset_name = config.data.get("dataset", "hotpot_qa")
    num_docs = config.data.get("num_docs", 10)
    print(f"  Loading dataset: {dataset_name}")
    dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki")
    corpus = prepare_corpus(dataset, max_docs=num_docs, dataset_name=dataset_name)

    # 2. Prepare test data
    print("\n[2] Preparing test data...")
    doc_ids_list = z_pool.doc_ids[:5]

    tokenizer = model.tokenizer
    tokenized_docs = {}
    for doc_id in doc_ids_list:
        text = corpus[doc_id]
        encoded = tokenizer(
            text,
            return_tensors="pt",
            max_length=config.data.get("max_doc_length", 512),
            truncation=True,
            padding=False,
        )
        tokenized_docs[doc_id] = encoded["input_ids"].cuda()

    print(f"  Prepared {len(doc_ids_list)} documents")

    # 3. Z-Only NLL Evaluation
    print("\n" + "=" * 60)
    print("[3] Z-ONLY NLL EVALUATION")
    print("=" * 60)

    model.eval()

    results_first_token = []  # 첫 토큰 NLL (pure z-only)
    results_z_only = []       # 50 토큰 평균 (teacher forcing 포함)
    results_with_doc = []     # doc context 있을 때

    with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
        for doc_id in doc_ids_list:
            z_i = z_pool.get_z(doc_id).to(model.device)
            doc_ids = tokenized_docs[doc_id]

            # Z-only NLL
            z_only_result = compute_z_only_nll(model, z_i, doc_ids)
            results_first_token.append(z_only_result["nll_first_token"])
            results_z_only.append(z_only_result["nll_avg_50"])

            # With-doc NLL (기존 forward 방식)
            attention_mask = torch.ones_like(doc_ids)
            out_with_doc = model(z_i, doc_ids, attention_mask)
            loss_with_doc = out_with_doc["loss"].item()
            results_with_doc.append(loss_with_doc)

            print(f"  {doc_id}:")
            print(f"    ★ nll_first_token (PURE z-only): {z_only_result['nll_first_token']:.4f}")
            print(f"    nll_avg_50 (teacher forcing):    {z_only_result['nll_avg_50']:.4f}")
            print(f"    with-doc NLL:                    {loss_with_doc:.4f}")

    # Summary
    avg_first_token = sum(results_first_token) / len(results_first_token)
    avg_z_only = sum(results_z_only) / len(results_z_only)
    avg_with_doc = sum(results_with_doc) / len(results_with_doc)

    print("\n" + "-" * 40)
    print("[Summary]")
    print(f"  ★ avg nll_first_token (PURE z-only): {avg_first_token:.4f}")
    print(f"  avg nll_avg_50 (teacher forcing):    {avg_z_only:.4f}")
    print(f"  avg with-doc NLL:                    {avg_with_doc:.4f}")
    print(f"\n  first_token vs with-doc gap: {avg_first_token - avg_with_doc:+.4f}")

    # 4. Diagnosis (based on PURE z-only = first_token)
    print("\n" + "=" * 60)
    print("[4] DIAGNOSIS (based on nll_first_token)")
    print("=" * 60)

    if avg_first_token > 10:
        print("🔴 z-only NLL이 매우 높음 (>10)")
        print("   → z가 문서 정보를 담고 있지 않음 (unconditional prior 수준)")
        print("   → z-only objective로 Phase 1 재학습 필요")
    elif avg_first_token > 7:
        print("🟡 z-only NLL이 높은 편 (7~10)")
        print("   → z가 약간의 정보를 담지만 부족함")
        print("   → z-only objective + capacity 증가 필요")
    elif avg_first_token > 4:
        print("🟢 z-only NLL이 합리적 (4~7)")
        print("   → z가 문서 정보를 일부 담고 있음")
        print("   → fine-tuning으로 개선 가능")
    else:
        print("🟢 z-only NLL이 낮음 (<4)")
        print("   → z가 문서 정보를 잘 담고 있음!")

    # 5. Baseline comparison (random z)
    print("\n" + "=" * 60)
    print("[5] BASELINE: Random Z (first_token comparison)")
    print("=" * 60)

    results_random_first = []
    with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
        for doc_id in doc_ids_list:
            # Random z (same shape as learned z)
            z_random = torch.randn_like(z_pool.get_z(doc_id)).to(model.device)
            z_random = z_random * 0.1  # scale down
            doc_ids = tokenized_docs[doc_id]

            random_result = compute_z_only_nll(model, z_random, doc_ids)
            results_random_first.append(random_result["nll_first_token"])

    avg_random_first = sum(results_random_first) / len(results_random_first)

    print(f"  ★ avg random-z first_token NLL:  {avg_random_first:.4f}")
    print(f"  ★ avg learned-z first_token NLL: {avg_first_token:.4f}")
    print(f"  improvement: {avg_random_first - avg_first_token:+.4f} ({(avg_random_first - avg_first_token) / avg_random_first * 100:+.1f}%)")

    if avg_first_token < avg_random_first - 0.5:
        print("\n  ✓ 학습된 z가 random z보다 유의미하게 좋음")
    else:
        print("\n  ⚠️ 학습된 z가 random z와 큰 차이 없음")

    # 6. Z Permutation Test (first_token)
    print("\n" + "=" * 60)
    print("[6] Z PERMUTATION TEST (first_token)")
    print("=" * 60)
    print("목적: doc_i에 z_j를 넣었을 때 first_token NLL이 증가하는지 확인")

    results_matched_first = []
    results_mismatched_first = []

    with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
        for i, doc_id in enumerate(doc_ids_list):
            other_doc_id = doc_ids_list[(i + 1) % len(doc_ids_list)]

            z_matched = z_pool.get_z(doc_id).to(model.device)
            z_mismatched = z_pool.get_z(other_doc_id).to(model.device)
            doc_ids = tokenized_docs[doc_id]

            matched_result = compute_z_only_nll(model, z_matched, doc_ids)
            mismatched_result = compute_z_only_nll(model, z_mismatched, doc_ids)

            results_matched_first.append(matched_result["nll_first_token"])
            results_mismatched_first.append(mismatched_result["nll_first_token"])

            print(f"  {doc_id}: matched={matched_result['nll_first_token']:.4f}, mismatched={mismatched_result['nll_first_token']:.4f}, diff={mismatched_result['nll_first_token'] - matched_result['nll_first_token']:+.4f}")

    avg_matched_first = sum(results_matched_first) / len(results_matched_first)
    avg_mismatched_first = sum(results_mismatched_first) / len(results_mismatched_first)

    print(f"\n  ★ avg matched first_token NLL:    {avg_matched_first:.4f}")
    print(f"  ★ avg mismatched first_token NLL: {avg_mismatched_first:.4f}")
    print(f"  gap: {avg_mismatched_first - avg_matched_first:+.4f}")

    if avg_mismatched_first > avg_matched_first + 0.5:
        print("\n  ✓ z가 문서별 정보를 담고 있음 (mismatched에서 NLL 증가)")
    else:
        print("\n  ⚠️ z가 문서별 정보를 담지 않음 (mismatched와 차이 없음)")


if __name__ == "__main__":
    main()
