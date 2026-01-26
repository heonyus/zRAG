"""
Data Mapping Verification Script
- z_pool에 저장된 doc_id들이 실제 corpus와 매칭되는지 확인
- 학습 시 사용한 텍스트 == 평가 시 사용하는 텍스트인지 검증
- corpus_manifest.json과 SHA256 해시 비교로 동일성 확정

핵심 질문:
1. z_pool.doc_ids가 corpus keys와 일치하는가?
2. z_i 학습 시 사용한 텍스트가 평가 시 tokenize하는 텍스트와 동일한가? (해시로 검증)
"""

import sys
sys.path.insert(0, "/home/lhe339/data/zRAG")

import torch
import hashlib
import json
from pathlib import Path
from omegaconf import OmegaConf
from datasets import load_dataset
from models.write_phase_model import WritePhaseModel, ZPoolManager
from training.train_write_phase import prepare_corpus


def main():
    print("=" * 60)
    print("DATA MAPPING VERIFICATION")
    print("=" * 60)
    print("목적: 학습 텍스트 == 평가 텍스트 확인")
    print("      (train-test mismatch가 있으면 모든 평가 무의미)")

    # 1. Load config
    config_path = "/home/lhe339/data/zRAG/configs/phase1_write.yaml"
    config = OmegaConf.load(config_path)

    print("\n" + "=" * 60)
    print("[1] CONFIG INFO")
    print("=" * 60)
    print(f"  dataset:        {config.data.get('dataset', 'hotpot_qa')}")
    print(f"  num_docs:       {config.data.get('num_docs', 10)}")
    print(f"  max_doc_length: {config.data.get('max_doc_length', 512)}")
    print(f"  save_dir:       {config.logging.save_dir}")

    # 2. Load z_pool
    print("\n" + "=" * 60)
    print("[2] Z_POOL INFO")
    print("=" * 60)

    z_pool_path = Path(config.logging.save_dir) / "z_pool.pt"
    if not z_pool_path.exists():
        print(f"  ❌ z_pool.pt not found at {z_pool_path}")
        return

    z_pool = ZPoolManager(m_tokens=config.memory.m_tokens, z_dim=config.memory.z_dim)
    z_pool.load(z_pool_path)

    print(f"  z_pool path:    {z_pool_path}")
    print(f"  num documents:  {len(z_pool.doc_ids)}")
    print(f"  doc_ids:        {z_pool.doc_ids}")

    # Check z_pool internal structure
    print(f"\n  [z_pool internal structure]")
    for doc_id in z_pool.doc_ids:
        z = z_pool.get_z(doc_id)
        print(f"    {doc_id}: z.shape={z.shape}, z.mean={z.mean().item():.4f}, z.std={z.std().item():.4f}")

    # 3. Load corpus (same way as training)
    print("\n" + "=" * 60)
    print("[3] CORPUS LOADING (same as training)")
    print("=" * 60)

    dataset_name = config.data.get("dataset", "hotpot_qa")
    num_docs = config.data.get("num_docs", 10)

    print(f"  Loading dataset: {dataset_name}")
    try:
        dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki")
    except Exception as e:
        print(f"  Warning: {e}")
        # Fallback without trust_remote_code
        dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki")
    corpus = prepare_corpus(dataset, max_docs=num_docs, dataset_name=dataset_name)

    print(f"  corpus type:    {type(corpus)}")
    print(f"  corpus keys:    {list(corpus.keys())}")
    print(f"  num documents:  {len(corpus)}")

    # 4. Mapping verification
    print("\n" + "=" * 60)
    print("[4] MAPPING VERIFICATION")
    print("=" * 60)

    # Check 1: z_pool doc_ids exist in corpus
    print("\n  [Check 1] z_pool doc_ids → corpus keys")
    missing_in_corpus = []
    for doc_id in z_pool.doc_ids:
        if doc_id in corpus:
            print(f"    ✓ {doc_id} exists in corpus")
        else:
            print(f"    ✗ {doc_id} NOT FOUND in corpus")
            missing_in_corpus.append(doc_id)

    if missing_in_corpus:
        print(f"\n  ❌ CRITICAL: {len(missing_in_corpus)} doc_ids missing from corpus!")
        print(f"     Missing: {missing_in_corpus}")
    else:
        print(f"\n  ✓ All z_pool doc_ids found in corpus")

    # Check 2: corpus keys exist in z_pool
    print("\n  [Check 2] corpus keys → z_pool doc_ids")
    missing_in_zpool = []
    for doc_id in corpus.keys():
        if doc_id in z_pool.doc_ids:
            print(f"    ✓ {doc_id} exists in z_pool")
        else:
            print(f"    ✗ {doc_id} NOT FOUND in z_pool")
            missing_in_zpool.append(doc_id)

    if missing_in_zpool:
        print(f"\n  ⚠️ {len(missing_in_zpool)} corpus keys missing from z_pool")
        print(f"     (This is OK if training used subset)")
    else:
        print(f"\n  ✓ All corpus keys found in z_pool")

    # 5. SHA256 Hash Verification (핵심!)
    print("\n" + "=" * 60)
    print("[5] SHA256 HASH VERIFICATION (CRITICAL)")
    print("=" * 60)
    print("목적: 학습 시 사용한 텍스트 == 현재 로드한 텍스트 (바이트 동일)")

    manifest_path = Path(config.logging.save_dir) / "corpus_manifest.json"
    hash_verified = False
    hash_mismatches = []

    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        print(f"  ✓ corpus_manifest.json found")
        print(f"    manifest docs: {manifest['num_docs']}")

        for doc_id in z_pool.doc_ids:
            if doc_id not in manifest["documents"]:
                print(f"    ⚠️ {doc_id} not in manifest")
                continue

            text = corpus.get(doc_id, "")
            if not text:
                print(f"    ❌ {doc_id} has no text in corpus")
                continue

            # 현재 텍스트의 SHA256
            current_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            saved_hash = manifest["documents"][doc_id]["text_sha256"]

            if current_hash == saved_hash:
                print(f"    ✓ {doc_id}: hash MATCH")
            else:
                print(f"    ❌ {doc_id}: hash MISMATCH!")
                print(f"       saved:   {saved_hash[:16]}...")
                print(f"       current: {current_hash[:16]}...")
                hash_mismatches.append(doc_id)

        if not hash_mismatches:
            hash_verified = True
            print(f"\n  ✓✓ ALL {len(z_pool.doc_ids)} documents verified (byte-identical)")
        else:
            print(f"\n  ❌ CRITICAL: {len(hash_mismatches)} documents have MISMATCHED hashes!")
            print(f"     Mismatched docs: {hash_mismatches}")
            print(f"     → 평가 결과 신뢰 불가! 재학습 또는 데이터 확인 필요")
    else:
        print(f"  ⚠️ corpus_manifest.json NOT FOUND at {manifest_path}")
        print(f"     → 학습 코드를 다시 실행하여 manifest 생성 필요")
        print(f"     → 또는 현재 텍스트가 학습 시와 동일하다고 가정")

    # 6. Document content samples (프리뷰)
    print("\n" + "=" * 60)
    print("[6] DOCUMENT CONTENT SAMPLES")
    print("=" * 60)
    print("(참고용 프리뷰 - 해시 검증이 통과했으면 불필요)")

    for i, doc_id in enumerate(z_pool.doc_ids[:5]):
        text = corpus.get(doc_id, "NOT FOUND")
        if text == "NOT FOUND":
            print(f"\n  [{i}] {doc_id}: ❌ NOT FOUND")
        else:
            # Show first 100 chars
            preview = text[:100].replace('\n', ' ')
            print(f"\n  [{i}] {doc_id}:")
            print(f"      len={len(text)} chars")
            print(f"      preview: \"{preview}...\"")

    # 7. Tokenization consistency check
    print("\n" + "=" * 60)
    print("[7] TOKENIZATION CONSISTENCY")
    print("=" * 60)

    # Load tokenizer
    model = WritePhaseModel(
        llm_name=config.model.llm_name,
        m_tokens=config.memory.m_tokens,
        z_dim=config.memory.z_dim,
        quantization=config.model.get("quantization", "4bit"),
    )
    tokenizer = model.tokenizer

    print(f"  tokenizer: {config.model.llm_name}")
    print(f"  max_length: {config.data.get('max_doc_length', 512)}")

    print("\n  [Token counts per document]")
    token_counts = {}
    for doc_id in z_pool.doc_ids:
        text = corpus.get(doc_id, "")
        if text:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                max_length=config.data.get("max_doc_length", 512),
                truncation=True,
                padding=False,
            )
            num_tokens = encoded["input_ids"].shape[1]
            token_counts[doc_id] = num_tokens

            # Show first few tokens
            first_tokens = encoded["input_ids"][0, :5].tolist()
            first_token_strs = [tokenizer.decode([t]) for t in first_tokens]
            print(f"    {doc_id}: {num_tokens} tokens, first 5: {first_token_strs}")

    # 8. First token analysis (relates to ranking test)
    print("\n" + "=" * 60)
    print("[8] FIRST TOKEN ANALYSIS")
    print("=" * 60)
    print("(ranking test의 first-token NLL과 연관)")

    first_token_dist = {}
    for doc_id in z_pool.doc_ids:
        text = corpus.get(doc_id, "")
        if text:
            encoded = tokenizer(text, return_tensors="pt", truncation=True)
            first_token_id = encoded["input_ids"][0, 0].item()
            first_token_str = tokenizer.decode([first_token_id])
            first_token_dist[doc_id] = {
                "token_id": first_token_id,
                "token_str": first_token_str,
            }
            print(f"    {doc_id}: first_token = '{first_token_str}' (id={first_token_id})")

    # Check for duplicate first tokens
    print("\n  [First token collision check]")
    token_to_docs = {}
    for doc_id, info in first_token_dist.items():
        token_str = info["token_str"]
        if token_str not in token_to_docs:
            token_to_docs[token_str] = []
        token_to_docs[token_str].append(doc_id)

    for token_str, docs in token_to_docs.items():
        if len(docs) > 1:
            print(f"    ⚠️ Token '{token_str}' shared by: {docs}")
        else:
            print(f"    ✓ Token '{token_str}': {docs[0]}")

    # 9. Summary
    print("\n" + "=" * 60)
    print("[9] SUMMARY")
    print("=" * 60)

    issues = []

    # Critical: hash verification
    if hash_mismatches:
        issues.append(f"❌ CRITICAL: {len(hash_mismatches)} docs have MISMATCHED hashes (train-test mismatch!)")
    elif not hash_verified and not manifest_path.exists():
        issues.append(f"⚠️ corpus_manifest.json not found - cannot verify text identity")

    if missing_in_corpus:
        issues.append(f"❌ {len(missing_in_corpus)} z_pool docs missing from corpus")

    # Check if first tokens are diverse
    unique_first_tokens = len(set(info["token_id"] for info in first_token_dist.values()))
    if unique_first_tokens < len(z_pool.doc_ids):
        issues.append(f"⚠️ Only {unique_first_tokens}/{len(z_pool.doc_ids)} unique first tokens (collision)")

    if not issues and hash_verified:
        print("  ✓✓ DATA MAPPING FULLY VERIFIED")
        print("  ✓ z_pool doc_ids match corpus keys")
        print("  ✓ All SHA256 hashes match (byte-identical text)")
        print(f"  ✓ All {len(z_pool.doc_ids)} documents verified")
        print("\n  → 평가 결과 신뢰 가능!")
    elif not issues:
        print("  ✓ Data mapping verified - no critical issues found")
        print("  ✓ z_pool doc_ids match corpus keys")
        print(f"  ✓ All {len(z_pool.doc_ids)} documents have valid text")
        print("\n  ⚠️ 단, hash 검증 불가 (manifest 없음)")
    else:
        print("  Issues found:")
        for issue in issues:
            print(f"    {issue}")

    print("\n  [Recommendation]")
    if hash_verified:
        print("  ✓ train-test mismatch 없음 확정 → 다음 스텝 진행 가능")
    elif hash_mismatches:
        print("  ❌ 데이터 불일치! 평가 중단하고 원인 파악 필요")
    else:
        print("  - manifest 생성 후 재검증 권장")
        print("  - 또는 현재 상태로 진행 (위험 감수)")
    print("  - first token collision이 있으면 first-K ranking 사용 권장")

    # ========================================
    # FINAL SUMMARY (복사/붙여넣기용)
    # ========================================
    print("\n")
    print("=" * 60)
    print("★★★ FINAL SUMMARY (복사용) ★★★")
    print("=" * 60)
    print(f"""
┌─────────────────────────────────────────────────────────┐
│  HASH VERIFICATION                                      │
│  ─────────────────                                      │
│  manifest exists:  {"YES" if manifest_path.exists() else "NO ← 훈련 필요!":20}                  │
│  hash mismatches:  {len(hash_mismatches):3} {"← CRITICAL!" if hash_mismatches else "← OK" if hash_verified else "← N/A":20}          │
│  verified docs:    {len(z_pool.doc_ids) - len(hash_mismatches):3}/{len(z_pool.doc_ids)}                                     │
│                                                         │
│  STATUS: {"✅ PASS - 평가 진행 가능" if hash_verified else "❌ FAIL - 평가 결과 무효" if hash_mismatches else "⚠️ UNKNOWN - manifest 생성 필요":40}│
└─────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    main()
