"""
Z Ranking Test (First-Token Only)
- teacher forcing 없이 순수 first-token NLL만으로 ranking
- doc_6 attractor 현상이 first-token에서도 나타나는지 확인

목적: teacher forcing 편향 제거 후 순수 z conditioning 품질 측정
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


def compute_first_token_nll(model, z_i, doc_ids):
    """
    z만으로 doc의 첫 번째 토큰을 예측하는 NLL 계산
    (teacher forcing 없는 pure z-only metric)
    """
    # z를 embedding space로 projection
    alpha_clamped = torch.clamp(model.alpha, min=0.5)
    z_embed = alpha_clamped * model.z_to_embedding(z_i)  # [m_tokens, hidden]
    z_embed = z_embed.unsqueeze(0)  # [1, m_tokens, hidden]

    # z_embed만 입력으로 forward
    outputs = model.llm(
        inputs_embeds=z_embed,
        use_cache=False,
    )

    # 마지막 z 토큰 위치에서 첫 번째 doc 토큰 예측
    last_logit = outputs.logits[0, -1, :]  # [vocab_size]
    first_doc_token = doc_ids[0, 0]  # scalar

    nll = F.cross_entropy(last_logit.unsqueeze(0), first_doc_token.unsqueeze(0))
    return nll.item()


def main():
    print("=" * 60)
    print("Z Ranking Test (FIRST-TOKEN ONLY)")
    print("=" * 60)
    print("목적: teacher forcing 없이 순수 first-token NLL로 ranking")
    print("      doc_6 attractor가 first-token에서도 나타나는지 확인")

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
    first_tokens = {}  # 각 문서의 첫 토큰 저장

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
        first_tokens[doc_id] = tokenizer.decode(encoded["input_ids"][0, 0])

    print(f"  Prepared {num_test_docs} documents")
    print(f"  Random baseline: {1/num_test_docs:.1%} (top-1)")

    # 각 문서의 첫 토큰 출력
    print(f"\n  --- First tokens of each document ---")
    for doc_id in doc_ids_list:
        first_tok = first_tokens[doc_id]
        print(f"    {doc_id}: '{first_tok}' (id={tokenized_docs[doc_id][0, 0].item()})")

    # 3. First-Token Ranking Test
    print("\n" + "=" * 60)
    print("[3] FIRST-TOKEN RANKING TEST")
    print("=" * 60)
    print("각 z_i에 대해 10개 문서의 first-token NLL 계산 후 ranking")
    print("(pure z-only, NO teacher forcing)")

    model.eval()

    correct_top1 = 0
    correct_top3 = 0
    all_ranks = []
    nll_matrix = {}  # {(z_i, doc_j): nll}

    with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
        for i, query_doc_id in enumerate(doc_ids_list):
            z_i = z_pool.get_z(query_doc_id).to(model.device)

            # 모든 문서에 대해 first-token NLL 계산
            nlls = {}
            for candidate_doc_id in doc_ids_list:
                doc_ids = tokenized_docs[candidate_doc_id]
                nll = compute_first_token_nll(model, z_i, doc_ids)
                nlls[candidate_doc_id] = nll
                nll_matrix[(query_doc_id, candidate_doc_id)] = nll

            # NLL 기준 정렬 (낮을수록 좋음)
            sorted_docs = sorted(nlls.items(), key=lambda x: x[1])

            # 정답 문서의 rank 찾기
            rank = -1
            for r, (doc_id, nll) in enumerate(sorted_docs):
                if doc_id == query_doc_id:
                    rank = r + 1
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
            best_doc = sorted_docs[0][0]
            nll_gap = correct_nll - best_nll

            print(f"\n  [{rank_symbol}] z_{i} ({query_doc_id}):")
            print(f"      rank = {rank}/{num_test_docs}")
            print(f"      correct doc first-token NLL = {correct_nll:.4f}")
            print(f"      best doc first-token NLL    = {best_nll:.4f} ({best_doc})")
            print(f"      gap (correct - best) = {nll_gap:+.4f}")

            # top-5 ranking
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
    print(f"\n  Rank distribution: {all_ranks}")

    # NLL 통계
    print("\n  --- First-Token NLL Statistics ---")
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

    print(f"  avg first-token NLL (correct doc):   {avg_correct_nll:.4f}")
    print(f"  avg first-token NLL (incorrect doc): {avg_incorrect_nll:.4f}")
    print(f"  separation gap:                      {nll_separation:+.4f}")

    # 5. Doc Attractor 분석
    print("\n" + "=" * 60)
    print("[5] DOC ATTRACTOR ANALYSIS")
    print("=" * 60)
    print("어떤 문서가 가장 자주 'best doc'으로 선택되는지 분석")

    best_doc_counts = {}
    for query_doc_id in doc_ids_list:
        # 이 z에서 best doc 찾기
        best_doc = min(doc_ids_list, key=lambda d: nll_matrix[(query_doc_id, d)])
        best_doc_counts[best_doc] = best_doc_counts.get(best_doc, 0) + 1

    print(f"\n  Best doc frequency:")
    for doc_id, count in sorted(best_doc_counts.items(), key=lambda x: -x[1]):
        pct = count / num_test_docs * 100
        bar = "█" * count
        is_attractor = " ← ATTRACTOR!" if count > num_test_docs * 0.5 else ""
        print(f"    {doc_id}: {count:2d} ({pct:4.0f}%) {bar}{is_attractor}")

    # 문서별 평균 NLL (어떤 z를 넣든 이 문서가 쉬운지)
    print(f"\n  Average first-token NLL per document (across all z):")
    doc_avg_nlls = {}
    for candidate_doc_id in doc_ids_list:
        nlls_for_doc = [nll_matrix[(q, candidate_doc_id)] for q in doc_ids_list]
        doc_avg_nlls[candidate_doc_id] = sum(nlls_for_doc) / len(nlls_for_doc)

    for doc_id, avg_nll in sorted(doc_avg_nlls.items(), key=lambda x: x[1]):
        first_tok = first_tokens[doc_id]
        is_easy = " ← EASY DOC" if avg_nll == min(doc_avg_nlls.values()) else ""
        print(f"    {doc_id}: {avg_nll:.4f} (first='{first_tok}'){is_easy}")

    # 6. 10x10 NLL Matrix 출력
    print("\n" + "=" * 60)
    print("[6] NLL MATRIX (z_i row, doc_j col)")
    print("=" * 60)
    print("    각 셀: NLL(doc_j | z_i)")
    print("    [X.X] = 대각선 (정답)")
    print("    *X.X* = 해당 row에서 최솟값 (best)")

    # Header
    header = "         " + " ".join([f"d{d[-1]:>5}" for d in doc_ids_list])
    print(header)
    print("         " + "-" * (7 * num_test_docs))

    for query_doc_id in doc_ids_list:
        # 이 row에서 best doc 찾기
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

    # Row/Column 평균
    print("\n  --- Row/Column Averages ---")
    print("  (Row avg = 해당 z가 전체 문서에 대해 얼마나 낮은 NLL을 주는지)")
    print("  (Col avg = 해당 문서가 모든 z에 대해 얼마나 쉬운지)")

    row_avgs = {}
    col_avgs = {}
    for query_doc_id in doc_ids_list:
        row_avgs[query_doc_id] = sum(nll_matrix[(query_doc_id, d)] for d in doc_ids_list) / num_test_docs
    for candidate_doc_id in doc_ids_list:
        col_avgs[candidate_doc_id] = sum(nll_matrix[(q, candidate_doc_id)] for q in doc_ids_list) / num_test_docs

    print(f"\n  Row averages (z → all docs):")
    for doc_id in sorted(row_avgs, key=row_avgs.get):
        print(f"    z_{doc_id[-1]}: {row_avgs[doc_id]:.4f}")

    print(f"\n  Column averages (all z → doc):")
    for doc_id in sorted(col_avgs, key=col_avgs.get):
        is_easiest = " ← EASIEST" if doc_id == min(col_avgs, key=col_avgs.get) else ""
        print(f"    {doc_id}: {col_avgs[doc_id]:.4f}{is_easiest}")

    # 7. Diagnosis
    print("\n" + "=" * 60)
    print("[7] DIAGNOSIS")
    print("=" * 60)

    # Attractor 존재 여부
    max_attractor_count = max(best_doc_counts.values())
    attractor_doc = max(best_doc_counts, key=best_doc_counts.get)

    print(f"\n  [Attractor 분석]")
    if max_attractor_count > num_test_docs * 0.5:
        print(f"  🔴 {attractor_doc}가 {max_attractor_count}/{num_test_docs}회 best로 선택됨")
        print(f"     → 강한 attractor 존재 (z 분화 실패 또는 쉬운 문서)")

        # attractor doc의 특성 분석
        attractor_avg_nll = doc_avg_nlls[attractor_doc]
        other_avg_nll = sum(v for k, v in doc_avg_nlls.items() if k != attractor_doc) / (num_test_docs - 1)
        print(f"     {attractor_doc} avg NLL: {attractor_avg_nll:.4f}")
        print(f"     Other docs avg NLL: {other_avg_nll:.4f}")
        print(f"     Gap: {other_avg_nll - attractor_avg_nll:+.4f}")

        if attractor_avg_nll < other_avg_nll - 0.5:
            print(f"     → {attractor_doc}가 구조적으로 '쉬운 문서'임 (첫 토큰이 일반적)")
        else:
            print(f"     → z들이 충분히 분화되지 않았을 가능성")
    else:
        print(f"  🟢 특정 문서로의 강한 attractor 없음")

    print(f"\n  [분리력 분석]")
    if nll_separation > 1.0:
        print(f"  🟢 separation gap {nll_separation:.4f} - 좋은 분리력")
    elif nll_separation > 0.3:
        print(f"  🟡 separation gap {nll_separation:.4f} - 약한 분리력")
    else:
        print(f"  🔴 separation gap {nll_separation:.4f} - 분리력 거의 없음")

    print(f"\n  [결론]")
    if top1_acc >= 0.5 and nll_separation > 0.5:
        print("  → first-token에서도 어느 정도 분리됨, objective 개선으로 향상 가능")
    elif top1_acc > random_top1:
        print("  → z에 약간의 정보 있으나 분리 부족")
        print("  → z-only objective + contrastive loss 필요")
    else:
        print("  → z가 문서 분리에 실패")
        print("  → 근본적인 objective 재설계 필요")

    # 8. 비교 (Teacher Forcing vs First-Token)
    print("\n" + "=" * 60)
    print("[8] COMPARISON: First-Token vs Teacher-Forcing")
    print("=" * 60)
    print("  이전 teacher-forcing(50토큰) 결과와 비교:")
    print("  (teacher-forcing 결과는 eval_z_ranking.py에서 확인)")
    print()
    print(f"  [First-Token Only (이 스크립트)]")
    print(f"    Top-1 Accuracy:  {top1_acc:.1%}")
    print(f"    Top-3 Accuracy:  {top3_acc:.1%}")
    print(f"    Average Rank:    {avg_rank:.2f}")
    print(f"    NLL Separation:  {nll_separation:+.4f}")
    print()
    print("  [해석 가이드]")
    print("  - First-Token 결과가 Teacher-Forcing보다 나쁘면:")
    print("      → z가 순수 conditioning으로는 약함")
    print("      → teacher forcing이 ranking을 도운 것")
    print("  - First-Token 결과가 비슷하면:")
    print("      → z 자체의 conditioning 품질 문제")
    print("  - doc_6 attractor가 first-token에서 더 강하면:")
    print("      → doc_6의 첫 토큰이 모델의 default output과 가까움")


if __name__ == "__main__":
    main()
