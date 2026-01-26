"""
Z Ranking Test
- 각 z_i에 대해 모든 문서 D_j의 NLL을 계산
- 정답 문서 D_i가 가장 낮은 NLL을 갖는지 확인 (top-1 accuracy)

결과 해석:
- top-1 accuracy = 10% (random baseline for 10 docs)
- top-1 accuracy >> 10%: z가 문서 content를 담고 있음
- top-1 accuracy ~ 10%: z가 content를 담지 않음 (그냥 LLM-compatible prefix)
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


def compute_doc_nll(model, z_i, doc_ids, max_tokens=50):
    """
    z_i가 주어졌을 때 doc의 NLL 계산 (teacher forcing)

    Args:
        model: WritePhaseModel
        z_i: [m_tokens, z_dim] z vector
        doc_ids: [1, doc_len] document token ids
        max_tokens: 최대 몇 토큰까지 계산할지

    Returns:
        nll: average NLL over tokens
    """
    # z를 embedding space로 projection
    alpha_clamped = torch.clamp(model.alpha, min=0.5)
    z_embed = alpha_clamped * model.z_to_embedding(z_i)  # [m_tokens, hidden]
    z_embed = z_embed.unsqueeze(0)  # [1, m_tokens, hidden]

    doc_len = doc_ids.shape[1]
    num_tokens = min(doc_len, max_tokens)

    total_nll = 0.0
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

        # 다음 토큰 embedding 추가 (teacher forcing)
        next_embed = model.llm.get_input_embeddings()(doc_ids[0, i:i+1]).unsqueeze(0)
        current_embeds = torch.cat([current_embeds, next_embed], dim=1)

    return total_nll / num_tokens


def main():
    print("=" * 60)
    print("Z Ranking Test")
    print("=" * 60)
    print("목적: z_i가 주어졌을 때 정답 문서 D_i를 찾을 수 있는가?")
    print("      (top-1 accuracy 측정)")

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

    # 2. Prepare test data - 모든 문서 tokenize
    print("\n[2] Preparing test data...")
    doc_ids_list = z_pool.doc_ids  # 모든 문서
    num_test_docs = len(doc_ids_list)

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

    print(f"  Prepared {num_test_docs} documents")
    print(f"  Random baseline: {1/num_test_docs:.1%} (top-1)")

    # 3. Ranking Test
    print("\n" + "=" * 60)
    print("[3] RANKING TEST")
    print("=" * 60)
    print(f"각 z_i에 대해 {num_test_docs}개 문서의 NLL 계산 후 ranking")
    print(f"(NLL이 낮을수록 z_i가 해당 문서를 잘 예측한다는 의미)")

    model.eval()

    correct_top1 = 0
    correct_top3 = 0
    all_ranks = []
    all_nll_matrices = []  # 전체 NLL matrix 저장

    with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
        for i, query_doc_id in enumerate(doc_ids_list):
            z_i = z_pool.get_z(query_doc_id).to(model.device)

            # 모든 문서에 대해 NLL 계산
            nlls = {}
            for candidate_doc_id in doc_ids_list:
                doc_ids = tokenized_docs[candidate_doc_id]
                nll = compute_doc_nll(model, z_i, doc_ids, max_tokens=50)
                nlls[candidate_doc_id] = nll

            all_nll_matrices.append(nlls)

            # NLL 기준 정렬 (낮을수록 좋음)
            sorted_docs = sorted(nlls.items(), key=lambda x: x[1])

            # 정답 문서의 rank 찾기
            rank = -1
            for r, (doc_id, nll) in enumerate(sorted_docs):
                if doc_id == query_doc_id:
                    rank = r + 1  # 1-indexed
                    break

            all_ranks.append(rank)

            if rank == 1:
                correct_top1 += 1
                rank_symbol = "✓"
            elif rank <= 3:
                correct_top3 += 1
                rank_symbol = "△"
            else:
                rank_symbol = "✗"

            # 결과 출력 (상세)
            correct_nll = nlls[query_doc_id]
            best_nll = sorted_docs[0][1]
            best_doc = sorted_docs[0][0]
            nll_gap = correct_nll - best_nll

            print(f"\n  [{rank_symbol}] z_{i} ({query_doc_id}):")
            print(f"      rank = {rank}/{num_test_docs}")
            print(f"      correct doc NLL = {correct_nll:.4f}")
            print(f"      best doc NLL    = {best_nll:.4f} ({best_doc})")
            print(f"      gap (correct - best) = {nll_gap:+.4f}")

            # top-5 ranking 출력
            top5_str = " > ".join([f"{d}({n:.2f})" for d, n in sorted_docs[:5]])
            print(f"      ranking: {top5_str}")

    # 4. Summary
    print("\n" + "=" * 60)
    print("[4] SUMMARY")
    print("=" * 60)

    top1_acc = correct_top1 / num_test_docs
    top3_acc = correct_top3 / num_test_docs
    avg_rank = sum(all_ranks) / len(all_ranks)
    random_top1 = 1 / num_test_docs
    random_top3 = min(3, num_test_docs) / num_test_docs
    random_avg_rank = (num_test_docs + 1) / 2

    print(f"  Total documents: {num_test_docs}")
    print(f"\n  {'Metric':<25} {'Actual':>10} {'Random':>10} {'Ratio':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Top-1 Accuracy':<25} {top1_acc:>9.1%} {random_top1:>9.1%} {top1_acc/random_top1:>9.1f}x")
    print(f"  {'Top-3 Accuracy':<25} {top3_acc:>9.1%} {random_top3:>9.1%} {top3_acc/random_top3:>9.1f}x")
    print(f"  {'Average Rank':<25} {avg_rank:>10.2f} {random_avg_rank:>10.2f} {random_avg_rank/avg_rank:>9.1f}x")

    print(f"\n  ★ Top-1 Accuracy: {top1_acc:.1%} ({correct_top1}/{num_test_docs})")
    print(f"  ★ Top-3 Accuracy: {top3_acc:.1%} ({correct_top3}/{num_test_docs})")
    print(f"  ★ Average Rank:   {avg_rank:.2f} (random would be {random_avg_rank:.1f})")
    print(f"\n  Rank distribution: {all_ranks}")

    # NLL 통계
    print("\n  --- NLL Statistics ---")
    all_correct_nlls = []
    all_incorrect_nlls = []
    for i, query_doc_id in enumerate(doc_ids_list):
        nlls = all_nll_matrices[i]
        for doc_id, nll in nlls.items():
            if doc_id == query_doc_id:
                all_correct_nlls.append(nll)
            else:
                all_incorrect_nlls.append(nll)

    avg_correct_nll = sum(all_correct_nlls) / len(all_correct_nlls)
    avg_incorrect_nll = sum(all_incorrect_nlls) / len(all_incorrect_nlls)
    nll_separation = avg_incorrect_nll - avg_correct_nll

    print(f"  avg NLL (correct doc):   {avg_correct_nll:.4f}")
    print(f"  avg NLL (incorrect doc): {avg_incorrect_nll:.4f}")
    print(f"  separation gap:          {nll_separation:+.4f}")

    if nll_separation > 0.5:
        print(f"  → 정답 문서의 NLL이 유의미하게 낮음 (good separation)")
    else:
        print(f"  → 정답/오답 문서의 NLL 차이가 작음 (poor separation)")

    # 5. Diagnosis
    print("\n" + "=" * 60)
    print("[5] DIAGNOSIS")
    print("=" * 60)

    random_baseline = 1 / num_test_docs
    improvement_ratio = top1_acc / random_baseline if random_baseline > 0 else 0

    print(f"\n  [핵심 지표]")
    print(f"  - Top-1 Accuracy: {top1_acc:.1%} (random: {random_baseline:.1%})")
    print(f"  - Improvement:    {improvement_ratio:.1f}x over random")
    print(f"  - NLL Separation: {nll_separation:+.4f}")

    print(f"\n  [판정]")
    if top1_acc >= 0.8:  # 80%+
        print("  🟢🟢 z가 문서 content를 매우 잘 담고 있음!")
        print(f"      top-1 accuracy {top1_acc:.1%} (거의 완벽)")
        print("      → Phase 2로 진행 가능")
    elif top1_acc > random_baseline * 5:  # 5x better than random
        print("  🟢 z가 문서 content를 담고 있음!")
        print(f"      top-1 accuracy {top1_acc:.1%} >> random {random_baseline:.1%} ({improvement_ratio:.1f}x)")
        print("      → z-only objective로 더 개선 가능")
    elif top1_acc > random_baseline * 2:  # 2x better than random
        print("  🟡 z가 약간의 content 정보를 담음")
        print(f"      top-1 accuracy {top1_acc:.1%} > random {random_baseline:.1%} ({improvement_ratio:.1f}x)")
        print("      → z-only objective 필요")
    elif top1_acc > random_baseline:  # slightly better
        print("  🟠 z가 content를 거의 담지 않음")
        print(f"      top-1 accuracy {top1_acc:.1%} ~ random {random_baseline:.1%} ({improvement_ratio:.1f}x)")
        print("      → 현재 objective로는 content encoding 어려움")
        print("      → z-only objective로 Phase 1 재설계 필요")
    else:
        print("  🔴 z가 content를 전혀 담지 않음")
        print(f"      top-1 accuracy {top1_acc:.1%} ≤ random {random_baseline:.1%}")
        print("      → 현재 objective로는 content encoding 불가")
        print("      → z-only objective로 Phase 1 재설계 필수")

    print(f"\n  [다음 단계]")
    if top1_acc >= 0.5:
        print("  1. Phase 1은 어느 정도 성공 - capacity/epoch 조절로 추가 개선")
        print("  2. 또는 z-only objective로 전환하여 더 강한 encoding 시도")
    else:
        print("  1. z-only objective로 Phase 1 재학습 (최우선)")
        print("  2. 짧은 문서(64-128 tokens)로 시작")
        print("  3. train-test mismatch 해소 필수")


if __name__ == "__main__":
    main()
