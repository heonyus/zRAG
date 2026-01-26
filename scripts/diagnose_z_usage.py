"""
z Usage Diagnostic Script
- z=0 ablation: z를 제거해도 loss/생성이 동일한지 확인
- z permutation: doc_i에 z_j를 넣어도 loss가 동일한지 확인

결과 해석:
- z=0에서 loss 차이 없음 → z가 conditioning으로 작동하지 않음
- permutation에서 loss 차이 없음 → z가 문서 식별 정보를 담지 않음
"""

import sys
sys.path.insert(0, "/home/lhe339/data/zRAG")

import torch
from torch.amp import autocast
from pathlib import Path
from omegaconf import OmegaConf
from datasets import load_dataset
from models.write_phase_model import WritePhaseModel, ZPoolManager
from training.train_write_phase import prepare_corpus

def main():
    print("=" * 60)
    print("Z Usage Diagnostic: z=0 Ablation + Permutation Test")
    print("=" * 60)

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

    # Load corpus for reference
    dataset_name = config.data.get("dataset", "hotpot_qa")
    num_docs = config.data.get("num_docs", 10)
    print(f"  Loading dataset: {dataset_name}")
    dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki")
    corpus = prepare_corpus(dataset, max_docs=num_docs, dataset_name=dataset_name)

    # 2. Prepare test data
    print("\n[2] Preparing test data...")
    doc_ids = z_pool.doc_ids[:5]  # Test on first 5 docs

    tokenizer = model.tokenizer
    tokenized_docs = {}
    for doc_id in doc_ids:
        text = corpus[doc_id]
        encoded = tokenizer(
            text,
            return_tensors="pt",
            max_length=config.data.get("max_doc_length", 512),
            truncation=True,
            padding=False,
        )
        tokenized_docs[doc_id] = {
            "input_ids": encoded["input_ids"].cuda(),
            "attention_mask": encoded["attention_mask"].cuda(),
        }

    print(f"  Prepared {len(doc_ids)} documents for testing")

    # 3. z=0 Ablation Test
    print("\n" + "=" * 60)
    print("[3] Z=0 ABLATION TEST")
    print("=" * 60)
    print("목적: z를 0으로 바꿔도 loss가 비슷하면 z가 무시되고 있음")

    model.eval()

    results_normal = []
    results_zero = []

    with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
        for doc_id in doc_ids:
            z_i = z_pool.get_z(doc_id).to(model.device)
            z_zero = torch.zeros_like(z_i)

            doc_data = tokenized_docs[doc_id]

            # Normal z
            out_normal = model(z_i, doc_data["input_ids"], doc_data["attention_mask"])
            loss_normal = out_normal["loss"].item()

            # Zero z
            out_zero = model(z_zero, doc_data["input_ids"], doc_data["attention_mask"])
            loss_zero = out_zero["loss"].item()

            results_normal.append(loss_normal)
            results_zero.append(loss_zero)

            print(f"  {doc_id}: loss(z)={loss_normal:.4f}, loss(z=0)={loss_zero:.4f}, diff={loss_zero - loss_normal:+.4f}")

    avg_normal = sum(results_normal) / len(results_normal)
    avg_zero = sum(results_zero) / len(results_zero)

    print(f"\n  [Summary] avg_loss(z)={avg_normal:.4f}, avg_loss(z=0)={avg_zero:.4f}")
    print(f"  [Summary] Difference: {avg_zero - avg_normal:+.4f} ({(avg_zero - avg_normal) / avg_normal * 100:+.1f}%)")

    if abs(avg_zero - avg_normal) < 0.1:
        print("\n  ⚠️  결론: z=0과 차이가 거의 없음 → z가 conditioning으로 작동하지 않음")
    else:
        print(f"\n  ✓ 결론: z=0에서 loss가 {avg_zero - avg_normal:+.4f} 증가 → z가 일부 사용됨")

    # 4. Generation comparison
    print("\n" + "-" * 40)
    print("[3.1] Generation Comparison (z vs z=0)")
    print("-" * 40)

    test_doc_id = doc_ids[0]
    z_i = z_pool.get_z(test_doc_id).to(model.device)
    z_zero = torch.zeros_like(z_i)

    gen_normal = model.generate_from_z(z_i, max_new_tokens=100, do_sample=False)
    gen_zero = model.generate_from_z(z_zero, max_new_tokens=100, do_sample=False)

    print(f"\n  Original: {corpus[test_doc_id][:150]}...")
    print(f"\n  Gen(z):   {gen_normal[:150]}...")
    print(f"\n  Gen(z=0): {gen_zero[:150]}...")

    if gen_normal[:50] == gen_zero[:50]:
        print("\n  ⚠️  생성 결과가 동일함 → z가 생성에 영향을 주지 않음")
    else:
        print("\n  ✓ 생성 결과가 다름 → z가 생성에 영향을 줌")

    # 5. Permutation Test
    print("\n" + "=" * 60)
    print("[4] Z PERMUTATION TEST")
    print("=" * 60)
    print("목적: doc_i에 z_j를 넣어도 loss가 비슷하면 z가 문서 식별 정보가 아님")

    results_matched = []
    results_mismatched = []

    with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
        for i, doc_id in enumerate(doc_ids):
            # Get z for different doc (circular)
            other_doc_id = doc_ids[(i + 1) % len(doc_ids)]

            z_matched = z_pool.get_z(doc_id).to(model.device)
            z_mismatched = z_pool.get_z(other_doc_id).to(model.device)

            doc_data = tokenized_docs[doc_id]

            # Matched z (correct)
            out_matched = model(z_matched, doc_data["input_ids"], doc_data["attention_mask"])
            loss_matched = out_matched["loss"].item()

            # Mismatched z (wrong doc's z)
            out_mismatched = model(z_mismatched, doc_data["input_ids"], doc_data["attention_mask"])
            loss_mismatched = out_mismatched["loss"].item()

            results_matched.append(loss_matched)
            results_mismatched.append(loss_mismatched)

            print(f"  {doc_id}: loss(z_i)={loss_matched:.4f}, loss(z_j)={loss_mismatched:.4f}, diff={loss_mismatched - loss_matched:+.4f}")

    avg_matched = sum(results_matched) / len(results_matched)
    avg_mismatched = sum(results_mismatched) / len(results_mismatched)

    print(f"\n  [Summary] avg_loss(matched)={avg_matched:.4f}, avg_loss(mismatched)={avg_mismatched:.4f}")
    print(f"  [Summary] Difference: {avg_mismatched - avg_matched:+.4f} ({(avg_mismatched - avg_matched) / avg_matched * 100:+.1f}%)")

    if abs(avg_mismatched - avg_matched) < 0.1:
        print("\n  ⚠️  결론: 잘못된 z를 넣어도 차이 없음 → z가 문서 식별 정보를 담지 않음")
    else:
        print(f"\n  ✓ 결론: 잘못된 z에서 loss가 {avg_mismatched - avg_matched:+.4f} 증가 → z가 문서별 정보를 담음")

    # Final diagnosis
    print("\n" + "=" * 60)
    print("[5] FINAL DIAGNOSIS")
    print("=" * 60)

    z0_diff = avg_zero - avg_normal
    perm_diff = avg_mismatched - avg_matched

    if abs(z0_diff) < 0.1 and abs(perm_diff) < 0.1:
        print("🔴 z가 완전히 무시됨 (conditioning 실패)")
        print("   → z-only objective로 Phase 1 재설계 필요")
    elif abs(z0_diff) < 0.1:
        print("🟡 z가 있긴 하지만 0과 구분 안됨")
        print("   → projection 또는 alpha 문제일 수 있음")
    elif abs(perm_diff) < 0.1:
        print("🟡 z가 conditioning이지만 문서별 정보가 아님")
        print("   → z가 generic signal로만 작동 중")
    else:
        print("🟢 z가 문서별 conditioning으로 작동 중")
        print(f"   z=0 diff: {z0_diff:+.4f}, perm diff: {perm_diff:+.4f}")
        print("   → keyword overlap 0% 문제는 capacity/objective 조정 필요")


if __name__ == "__main__":
    main()
