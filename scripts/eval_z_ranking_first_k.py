"""
Z Ranking Test - First-K Tokens (K=16 default)
- First-token ranking의 개선 버전
- 첫 토큰만이 아니라 처음 K개 토큰의 평균 NLL로 ranking
- bfloat16 양자화 문제 회피를 위해 fp32로 NLL 계산

장점:
- 단일 토큰 편향(easy first token) 완화
- 더 안정적인 ranking 지표
- teacher forcing 없이 순수 z-only conditioning 측정
"""

import sys
sys.path.insert(0, "/home/lhe339/data/zRAG")

import torch
import torch.nn.functional as F
from pathlib import Path
from omegaconf import OmegaConf
from datasets import load_dataset
from models.write_phase_model import WritePhaseModel, ZPoolManager
from training.train_write_phase import prepare_corpus


def compute_first_k_nll(model, z_i, doc_ids, k=16):
    """
    z_i가 주어졌을 때 doc의 처음 K개 토큰 NLL 계산 (순수 z-only)

    Args:
        model: WritePhaseModel
        z_i: [m_tokens, z_dim] z vector
        doc_ids: [1, doc_len] document token ids
        k: 처음 몇 개 토큰을 볼지 (default=16)

    Returns:
        dict: {
            "nll_avg_k": 처음 K개 토큰 평균 NLL,
            "nll_first": 첫 토큰 NLL,
            "nlls": 개별 토큰 NLL 리스트,
        }
    """
    # z를 embedding space로 projection
    alpha_clamped = torch.clamp(model.alpha, min=0.5)
    z_embed = alpha_clamped * model.z_to_embedding(z_i)  # [m_tokens, hidden]
    z_embed = z_embed.unsqueeze(0)  # [1, m_tokens, hidden]

    # LLM의 dtype에 맞추기 (bfloat16)
    model_dtype = next(model.llm.parameters()).dtype
    z_embed = z_embed.to(dtype=model_dtype)

    doc_len = doc_ids.shape[1]
    num_tokens = min(doc_len, k)

    nlls = []
    current_embeds = z_embed.clone()

    for i in range(num_tokens):
        # forward (autoregressive)
        outputs = model.llm(
            inputs_embeds=current_embeds,
            use_cache=False,
        )

        # 마지막 위치에서 다음 토큰 예측
        # FP32로 NLL 계산 (bfloat16 양자화 문제 회피)
        logits = outputs.logits[0, -1, :].float()  # [vocab_size] -> fp32
        target = doc_ids[0, i]

        nll = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
        nlls.append(nll.item())

        # 다음 토큰 embedding 추가 (teacher forcing으로 ground truth 사용)
        next_embed = model.llm.get_input_embeddings()(doc_ids[0, i:i+1]).unsqueeze(0)
        next_embed = next_embed.to(dtype=model_dtype)
        current_embeds = torch.cat([current_embeds, next_embed], dim=1)

    return {
        "nll_avg_k": sum(nlls) / len(nlls),
        "nll_first": nlls[0] if nlls else float("inf"),
        "nlls": nlls,
        "k": num_tokens,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=16, help="Number of tokens to evaluate")
    args = parser.parse_args()

    K = args.k

    print("=" * 60)
    print(f"Z Ranking Test - First-{K} Tokens")
    print("=" * 60)
    print(f"목적: z_i가 주어졌을 때 처음 {K}개 토큰의 평균 NLL로 문서 ranking")
    print("      (first-token 편향 완화, fp32로 정밀 계산)")

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
    doc_ids_list = z_pool.doc_ids
    num_test_docs = len(doc_ids_list)

    tokenizer = model.tokenizer
    tokenized_docs = {}
    first_k_tokens = {}  # 각 문서의 first K tokens 저장

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

        # first K tokens 저장 (디버깅용)
        first_k = encoded["input_ids"][0, :K].tolist()
        first_k_strs = [tokenizer.decode([t]) for t in first_k]
        first_k_tokens[doc_id] = first_k_strs

    print(f"  Prepared {num_test_docs} documents")
    print(f"  Random baseline: {1/num_test_docs:.1%} (top-1)")
    print(f"  Evaluation: first-{K} tokens")

    # 3. First-K Token Distribution
    print("\n" + "=" * 60)
    print(f"[3] FIRST-{K} TOKEN DISTRIBUTION")
    print("=" * 60)

    for doc_id in doc_ids_list:
        tokens_str = " ".join([f"'{t}'" for t in first_k_tokens[doc_id][:5]])
        print(f"  {doc_id}: {tokens_str} ...")

    # 4. Ranking Test
    print("\n" + "=" * 60)
    print(f"[4] RANKING TEST (First-{K} Tokens)")
    print("=" * 60)
    print(f"각 z_i에 대해 {num_test_docs}개 문서의 first-{K} NLL 계산 후 ranking")
    print(f"(NLL이 낮을수록 z_i가 해당 문서를 잘 예측한다는 의미)")

    model.eval()

    correct_top1 = 0
    correct_top3 = 0
    all_ranks = []
    nll_matrix = {}  # (query_doc, candidate_doc) -> nll
    best_doc_counts = {doc_id: 0 for doc_id in doc_ids_list}

    with torch.no_grad():
        for i, query_doc_id in enumerate(doc_ids_list):
            z_i = z_pool.get_z(query_doc_id).to(model.device)

            # 모든 문서에 대해 first-K NLL 계산
            nlls = {}
            for candidate_doc_id in doc_ids_list:
                doc_ids = tokenized_docs[candidate_doc_id]
                result = compute_first_k_nll(model, z_i, doc_ids, k=K)
                nlls[candidate_doc_id] = result["nll_avg_k"]
                nll_matrix[(query_doc_id, candidate_doc_id)] = result["nll_avg_k"]

            # NLL 기준 정렬 (낮을수록 좋음)
            sorted_docs = sorted(nlls.items(), key=lambda x: x[1])
            best_doc = sorted_docs[0][0]
            best_doc_counts[best_doc] += 1

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

            # 결과 출력
            correct_nll = nlls[query_doc_id]
            best_nll = sorted_docs[0][1]
            nll_gap = correct_nll - best_nll

            print(f"\n  [{rank_symbol}] z_{i} ({query_doc_id}):")
            print(f"      rank = {rank}/{num_test_docs}")
            print(f"      correct doc NLL = {correct_nll:.4f}")
            print(f"      best doc NLL    = {best_nll:.4f} ({best_doc})")
            print(f"      gap (correct - best) = {nll_gap:+.4f}")

            # top-3 ranking 출력
            top3_str = " > ".join([f"{d}({n:.3f})" for d, n in sorted_docs[:3]])
            print(f"      top-3: {top3_str}")

    # 5. Summary
    print("\n" + "=" * 60)
    print("[5] SUMMARY")
    print("=" * 60)

    top1_acc = correct_top1 / num_test_docs
    top3_acc = (correct_top1 + correct_top3) / num_test_docs
    avg_rank = sum(all_ranks) / len(all_ranks)
    random_top1 = 1 / num_test_docs
    random_top3 = min(3, num_test_docs) / num_test_docs
    random_avg_rank = (num_test_docs + 1) / 2

    print(f"  Total documents: {num_test_docs}")
    print(f"  Evaluation: first-{K} tokens (fp32)")

    print(f"\n  {'Metric':<25} {'Actual':>10} {'Random':>10} {'Ratio':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Top-1 Accuracy':<25} {top1_acc:>9.1%} {random_top1:>9.1%} {top1_acc/random_top1:>9.1f}x")
    print(f"  {'Top-3 Accuracy':<25} {top3_acc:>9.1%} {random_top3:>9.1%} {top3_acc/random_top3:>9.1f}x")
    print(f"  {'Average Rank':<25} {avg_rank:>10.2f} {random_avg_rank:>10.2f} {random_avg_rank/avg_rank:>9.1f}x")

    print(f"\n  ★ Top-1 Accuracy: {top1_acc:.1%} ({correct_top1}/{num_test_docs})")
    print(f"\n  Rank distribution: {all_ranks}")

    # NLL 통계
    all_correct_nlls = []
    all_incorrect_nlls = []
    for query_doc_id in doc_ids_list:
        for candidate_doc_id in doc_ids_list:
            nll = nll_matrix[(query_doc_id, candidate_doc_id)]
            if query_doc_id == candidate_doc_id:
                all_correct_nlls.append(nll)
            else:
                all_incorrect_nlls.append(nll)

    avg_correct_nll = sum(all_correct_nlls) / len(all_correct_nlls)
    avg_incorrect_nll = sum(all_incorrect_nlls) / len(all_incorrect_nlls)
    nll_separation = avg_incorrect_nll - avg_correct_nll

    print(f"\n  --- NLL Statistics ---")
    print(f"  avg NLL (correct doc):   {avg_correct_nll:.4f}")
    print(f"  avg NLL (incorrect doc): {avg_incorrect_nll:.4f}")
    print(f"  separation gap:          {nll_separation:+.4f}")

    # 6. Attractor Analysis
    print("\n" + "=" * 60)
    print("[6] ATTRACTOR ANALYSIS")
    print("=" * 60)
    print("(어떤 문서가 가장 자주 best로 선택되는지)")

    for doc_id in sorted(best_doc_counts, key=best_doc_counts.get, reverse=True):
        count = best_doc_counts[doc_id]
        is_attractor = " ← ATTRACTOR" if count >= num_test_docs // 2 else ""
        print(f"  {doc_id}: selected {count}/{num_test_docs} times{is_attractor}")

    # Column averages (easy doc detection)
    col_avgs = {}
    for candidate_doc_id in doc_ids_list:
        col_avgs[candidate_doc_id] = sum(nll_matrix[(q, candidate_doc_id)] for q in doc_ids_list) / num_test_docs

    print("\n  --- Column Averages (Easy Doc Detection) ---")
    for doc_id in sorted(col_avgs, key=col_avgs.get):
        is_easiest = " ← EASIEST" if doc_id == min(col_avgs, key=col_avgs.get) else ""
        first_tokens = " ".join(first_k_tokens[doc_id][:3])
        print(f"  {doc_id}: avg_nll={col_avgs[doc_id]:.4f} (first='{first_tokens}'){is_easiest}")

    # 7. NLL Matrix
    print("\n" + "=" * 60)
    print(f"[7] NLL MATRIX (first-{K} avg)")
    print("=" * 60)
    print("    각 셀: NLL(doc_j | z_i)")
    print("    [X.X] = 대각선 (정답)")
    print("    *X.X* = 해당 row에서 최솟값 (best)")

    # Header
    header = "         " + " ".join([f"d{d[-1]:>5}" for d in doc_ids_list])
    print(header)
    print("         " + "-" * (7 * num_test_docs))

    for query_doc_id in doc_ids_list:
        best_in_row = min(doc_ids_list, key=lambda d: nll_matrix[(query_doc_id, d)])

        row = f"  z_{query_doc_id[-1]}  |"
        for candidate_doc_id in doc_ids_list:
            nll = nll_matrix[(query_doc_id, candidate_doc_id)]
            if query_doc_id == candidate_doc_id:
                row += f" [{nll:4.1f}]"  # 대각선 강조
            elif candidate_doc_id == best_in_row:
                row += f" *{nll:4.1f}*"  # best 강조
            else:
                row += f"  {nll:4.1f} "
        print(row)

    # 8. Diagnosis
    print("\n" + "=" * 60)
    print("[8] DIAGNOSIS")
    print("=" * 60)

    print(f"\n  [핵심 지표]")
    print(f"  - Top-1 Accuracy: {top1_acc:.1%} (random: {random_top1:.1%})")
    print(f"  - Improvement:    {top1_acc/random_top1:.1f}x over random")
    print(f"  - NLL Separation: {nll_separation:+.4f}")

    print(f"\n  [판정]")
    if top1_acc >= 0.8:
        print("  🟢🟢 z가 문서 content를 매우 잘 담고 있음!")
        print("      → Phase 2로 진행 가능")
    elif top1_acc > random_top1 * 5:
        print("  🟢 z가 문서 content를 담고 있음!")
        print(f"      top-1 accuracy {top1_acc:.1%} >> random {random_top1:.1%}")
        print("      → z-only objective로 더 개선 가능")
    elif top1_acc > random_top1 * 2:
        print("  🟡 z가 약간의 content 정보를 담음")
        print(f"      top-1 accuracy {top1_acc:.1%} > random {random_top1:.1%}")
        print("      → z-only objective 필요")
    elif top1_acc > random_top1:
        print("  🟠 z가 content를 거의 담지 않음")
        print("      → z-only objective로 Phase 1 재설계 필요")
    else:
        print("  🔴 z가 content를 전혀 담지 않음")
        print("      → 근본적인 objective 재설계 필수")

    # 9. Comparison with first-token
    print("\n" + "=" * 60)
    print("[9] COMPARISON: First-K vs First-Token")
    print("=" * 60)
    print(f"  [First-{K} Tokens (이 스크립트)]")
    print(f"    Top-1 Accuracy:  {top1_acc:.1%}")
    print(f"    NLL Separation:  {nll_separation:+.4f}")
    print()
    print("  [해석 가이드]")
    print(f"  - First-{K}가 First-1보다 나쁘면:")
    print("      → z가 첫 토큰만 잘 예측 (약한 conditioning)")
    print(f"  - First-{K}가 비슷하거나 더 좋으면:")
    print("      → z가 여러 토큰에 걸쳐 일관된 conditioning 제공")
    print("  - Attractor가 줄었으면:")
    print(f"      → first-token 편향이 완화됨")

    # ========================================
    # FINAL SUMMARY (복사/붙여넣기용)
    # ========================================
    # Find attractor (most frequently selected as best)
    max_selected = max(best_doc_counts.values())
    attractor_docs = [d for d, c in best_doc_counts.items() if c == max_selected]
    attractor_str = ", ".join(attractor_docs) if attractor_docs else "none"

    # Find easiest doc
    easiest_doc = min(col_avgs, key=col_avgs.get)
    easiest_nll = col_avgs[easiest_doc]

    print("\n")
    print("=" * 60)
    print("★★★ FINAL SUMMARY (복사용) ★★★")
    print("=" * 60)
    print(f"""
┌─────────────────────────────────────────────────────────┐
│  FIRST-{K} RANKING RESULTS                               │
│  ────────────────────                                   │
│  Top-1 Accuracy:   {top1_acc:6.1%}  (random: {random_top1:5.1%}, {top1_acc/random_top1:.1f}x)       │
│  Top-3 Accuracy:   {top3_acc:6.1%}  (random: {random_top3:5.1%}, {top3_acc/random_top3:.1f}x)       │
│  Average Rank:     {avg_rank:6.2f}  (random: {random_avg_rank:5.2f})              │
│                                                         │
│  NLL STATISTICS                                         │
│  ──────────────                                         │
│  avg NLL (correct):   {avg_correct_nll:7.4f}                          │
│  avg NLL (incorrect): {avg_incorrect_nll:7.4f}                          │
│  separation gap:      {nll_separation:+7.4f}                          │
│                                                         │
│  ATTRACTOR / EASY DOC                                   │
│  ────────────────────                                   │
│  attractor:  {attractor_str:12} (selected {max_selected}/{num_test_docs} times)        │
│  easiest:    {easiest_doc:12} (avg_nll={easiest_nll:.4f})             │
│                                                         │
│  VERDICT: {"🟢 z contains content info" if top1_acc > random_top1 * 2 else "🟡 weak signal" if top1_acc > random_top1 else "🔴 no signal":40}│
└─────────────────────────────────────────────────────────┘
""")

    # One-liner for easy copy
    print("▶ ONE-LINER:")
    print(f"  K={K} | Top1={top1_acc:.1%} | Sep={nll_separation:+.4f} | Attractor={attractor_str}({max_selected}/{num_test_docs})")


if __name__ == "__main__":
    main()
