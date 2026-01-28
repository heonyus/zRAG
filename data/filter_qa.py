"""
Phase 2 데이터 필터링 모듈

Phase 1에서 학습한 corpus에 정답이 존재하는 QA만 선별하여
공정한 비교를 보장합니다.

문제점 (필터링 없이):
- BM25/Contriever는 "정답 없는 corpus"에서 억울하게 짐
- zRAG는 hallucination evidence로 올라옴

해결책:
- Phase 2 eval query가 Phase 1 corpus 안에서만 경쟁하도록 필터링
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_phase1_doc_ids(checkpoint_path: str) -> Set[str]:
    """
    Phase 1 checkpoint에서 학습된 문서 ID 목록 로드

    Args:
        checkpoint_path: z_pool_epoch50.pt 또는 z_pool.pt 경로

    Returns:
        Set of document IDs that were trained in Phase 1
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # z_vectors dict에서 doc_id 추출
    if "z_vectors" in checkpoint:
        doc_ids = set(checkpoint["z_vectors"].keys())
    elif "doc_ids" in checkpoint:
        doc_ids = set(checkpoint["doc_ids"])
    else:
        raise ValueError(f"Cannot find doc_ids in checkpoint: {checkpoint_path}")

    logger.info(f"Loaded {len(doc_ids)} doc_ids from Phase 1 checkpoint")
    return doc_ids


def load_corpus_manifest(manifest_path: str) -> Dict[str, dict]:
    """
    corpus_manifest.json 로드

    Args:
        manifest_path: corpus_manifest.json 경로

    Returns:
        Dict mapping doc_id to document metadata
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    return manifest.get("documents", {})


def filter_qa_for_phase1_corpus(
    qa_pairs: List[Dict],
    phase1_doc_ids: Set[str],
    gold_doc_key: str = "gold_doc_ids",
    strict: bool = True,
) -> Tuple[List[Dict], Dict]:
    """
    Phase 1 corpus에 정답이 존재하는 QA만 선별

    Args:
        qa_pairs: QA 데이터 리스트 [{question, answer, gold_doc_ids, ...}, ...]
        phase1_doc_ids: Phase 1에서 학습한 문서 ID 집합
        gold_doc_key: gold document ID가 저장된 키 이름
        strict: True면 모든 gold_doc이 phase1에 있어야 함, False면 하나라도 있으면 OK

    Returns:
        Tuple of (filtered_qa_pairs, statistics)
    """
    filtered = []
    stats = {
        "total": len(qa_pairs),
        "filtered": 0,
        "removed": 0,
        "partial_match": 0,  # 일부만 매칭된 경우
        "no_gold_docs": 0,   # gold_doc_ids가 없는 경우
    }

    for qa in tqdm(qa_pairs, desc="Filtering QA pairs"):
        gold_docs = qa.get(gold_doc_key, [])

        # gold_doc_ids가 없는 경우 → 제거 (Phase 2에서는 gold 필수)
        if not gold_docs:
            stats["no_gold_docs"] += 1
            stats["removed"] += 1
            continue

        # gold_docs 중 phase1 corpus에 있는 것 찾기
        matching_docs = [d for d in gold_docs if d in phase1_doc_ids]

        if strict:
            # 모든 gold_docs가 phase1에 있어야 함
            if len(matching_docs) == len(gold_docs):
                filtered.append(qa)
                stats["filtered"] += 1
            elif len(matching_docs) > 0:
                stats["partial_match"] += 1
                stats["removed"] += 1
            else:
                stats["removed"] += 1
        else:
            # 하나라도 매칭되면 OK
            if len(matching_docs) > 0:
                # 매칭된 gold_docs만 유지
                qa_filtered = qa.copy()
                qa_filtered[gold_doc_key] = matching_docs
                filtered.append(qa_filtered)
                stats["filtered"] += 1
                if len(matching_docs) < len(gold_docs):
                    stats["partial_match"] += 1
            else:
                stats["removed"] += 1

    logger.info(f"QA Filtering Results:")
    logger.info(f"  Total: {stats['total']}")
    pct = stats['filtered'] / stats['total'] * 100 if stats['total'] > 0 else 0.0
    logger.info(f"  Filtered (kept): {stats['filtered']} ({pct:.1f}%)")
    logger.info(f"  Removed: {stats['removed']}")
    logger.info(f"  Partial matches: {stats['partial_match']}")
    logger.info(f"  No gold docs: {stats['no_gold_docs']}")

    return filtered, stats


def get_overlapping_qa(
    dataset_name: str,
    phase1_checkpoint_path: str,
    split: str = "validation",
    max_samples: Optional[int] = None,
) -> Tuple[List[Dict], Set[str], Dict]:
    """
    Phase 1 corpus와 겹치는 QA 데이터 로드

    Args:
        dataset_name: "natural_questions" 또는 "hotpot_qa"
        phase1_checkpoint_path: Phase 1 checkpoint 경로
        split: 데이터셋 split ("train", "validation", "test")
        max_samples: 최대 샘플 수

    Returns:
        Tuple of (filtered_qa_pairs, phase1_doc_ids, statistics)
    """
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from data.download import download_dataset
    from data.preprocess import preprocess_nq, preprocess_hotpotqa

    # Phase 1 doc_ids 로드
    phase1_doc_ids = load_phase1_doc_ids(phase1_checkpoint_path)

    # 데이터셋 로드
    logger.info(f"Loading {dataset_name} ({split})...")
    dataset = download_dataset(dataset_name)

    # 전처리 (corpus 구축 포함)
    if dataset_name == "natural_questions":
        processed = preprocess_nq(dataset, corpus_size=50000, max_samples=max_samples or 10000)
    elif dataset_name == "hotpot_qa":
        processed = preprocess_hotpotqa(dataset, corpus_size=50000, max_samples=max_samples or 10000)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    qa_pairs = processed["qa_pairs"]
    corpus = processed["corpus"]

    logger.info(f"Loaded {len(qa_pairs)} QA pairs, {len(corpus)} documents")

    # 필터링
    filtered_qa, stats = filter_qa_for_phase1_corpus(
        qa_pairs=qa_pairs,
        phase1_doc_ids=phase1_doc_ids,
        strict=False,  # 하나라도 매칭되면 OK
    )

    return filtered_qa, phase1_doc_ids, stats


def analyze_corpus_overlap(
    phase1_checkpoint_path: str,
    dataset_name: str = "natural_questions",
) -> Dict:
    """
    Phase 1 corpus와 데이터셋 간 overlap 분석

    Returns:
        Analysis statistics
    """
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from data.download import download_dataset

    phase1_doc_ids = load_phase1_doc_ids(phase1_checkpoint_path)
    dataset = download_dataset(dataset_name)

    # 데이터셋의 모든 문서 ID 수집
    all_doc_ids = set()
    qa_with_overlap = 0
    total_qa = 0

    if dataset_name == "hotpot_qa":
        for item in tqdm(dataset["train"], desc="Analyzing train"):
            total_qa += 1
            # HotpotQA context에서 title 추출
            if "context" in item:
                for title, _ in item["context"]:
                    all_doc_ids.add(f"hotpot_{title}")
                    if f"hotpot_{title}" in phase1_doc_ids:
                        qa_with_overlap += 1
                        break

    analysis = {
        "phase1_docs": len(phase1_doc_ids),
        "dataset_docs": len(all_doc_ids),
        "overlap_docs": len(phase1_doc_ids & all_doc_ids),
        "total_qa": total_qa,
        "qa_with_overlap": qa_with_overlap,
        "overlap_rate": qa_with_overlap / total_qa if total_qa > 0 else 0,
    }

    logger.info(f"Corpus Overlap Analysis:")
    logger.info(f"  Phase 1 docs: {analysis['phase1_docs']}")
    logger.info(f"  Dataset docs: {analysis['dataset_docs']}")
    logger.info(f"  Overlapping docs: {analysis['overlap_docs']}")
    logger.info(f"  QA with overlap: {analysis['qa_with_overlap']}/{analysis['total_qa']} ({analysis['overlap_rate']*100:.1f}%)")

    return analysis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter QA pairs for Phase 1 corpus")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/phase1_final/z_pool_epoch50.pt",
        help="Phase 1 checkpoint path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="hotpot_qa",
        choices=["natural_questions", "hotpot_qa"],
        help="Dataset name",
    )
    parser.add_argument(
        "--analyze_only",
        action="store_true",
        help="Only analyze overlap, don't filter",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.analyze_only:
        analyze_corpus_overlap(args.checkpoint, args.dataset)
    else:
        filtered, doc_ids, stats = get_overlapping_qa(
            args.dataset,
            args.checkpoint,
            max_samples=1000,
        )
        print(f"\nFiltered QA pairs: {len(filtered)}")
        if filtered:
            print(f"Sample: {filtered[0]}")
