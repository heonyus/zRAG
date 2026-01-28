"""
Corpus Builder for Phase 2

핵심 전략: QA 먼저, Corpus 나중
- 기존: 랜덤 200 docs → 5000 QA 중 26개만 매칭 (0.5%)
- 신규: 1000+ QA 선택 → 그 QA의 supporting docs로 corpus 구성 → 100% 매칭

사용법:
    python data/corpus_builder.py --num_qa 1000 --num_docs 200 --output_dir checkpoints/phase2_corpus
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
from dataclasses import dataclass
import argparse

import torch


@dataclass
class CorpusStats:
    """Corpus 빌드 통계"""
    num_qa: int
    num_docs: int
    avg_qa_per_doc: float
    bridge_count: int
    comparison_count: int
    hard_count: int
    medium_count: int
    easy_count: int
    coverage: float  # QA가 corpus에 gold doc 있는 비율


def load_hotpotqa(split: str = "train"):
    """HotpotQA 로드"""
    from datasets import load_dataset

    print(f"Loading HotpotQA {split} split...")
    dataset = load_dataset("hotpot_qa", "distractor", split=split)
    print(f"  Loaded {len(dataset)} samples")
    return dataset


def extract_supporting_docs(item: dict) -> List[Tuple[str, str]]:
    """
    HotpotQA 샘플에서 supporting documents 추출

    Returns:
        List of (title, text) tuples for supporting documents
    """
    supporting_titles = set()

    # supporting_facts에서 title 추출
    sf = item.get("supporting_facts", {})
    if isinstance(sf, dict):
        supporting_titles = set(sf.get("title", []))
    elif isinstance(sf, list):
        for pair in sf:
            if isinstance(pair, (list, tuple)) and len(pair) >= 1:
                supporting_titles.add(pair[0])

    # context에서 해당 title의 text 추출
    docs = []
    for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
        if title in supporting_titles:
            text = " ".join(sentences)
            if text.strip():
                docs.append((title, text))

    return docs


def build_corpus_from_qa(
    dataset,
    num_qa: int = 1000,
    num_docs: int = 200,
    balance_types: bool = True,
    seed: int = 42,
) -> Tuple[Dict[str, str], List[dict], CorpusStats]:
    """
    QA 기반 corpus 구축 (핵심 함수)

    전략:
    1. 각 문서별로 어떤 QA가 참조하는지 역인덱스 구축
    2. 가장 많은 QA를 커버하는 문서 우선 선택 (greedy set cover)
    3. num_docs 문서 선택 후, 해당 문서에 gold doc이 있는 QA만 필터

    Args:
        dataset: HotpotQA dataset
        num_qa: 목표 QA 수 (실제로는 corpus 커버리지에 따라 더 적을 수 있음)
        num_docs: corpus 문서 수
        balance_types: bridge/comparison 균형 맞추기
        seed: 랜덤 시드

    Returns:
        (corpus, qa_pairs, stats)
    """
    random.seed(seed)

    print(f"Building corpus: target {num_docs} docs, {num_qa} QA pairs")

    # 1. 모든 문서와 QA 관계 수집
    doc_to_text = {}  # title -> text
    doc_to_qas = defaultdict(list)  # title -> list of qa indices
    qa_to_docs = defaultdict(set)  # qa_idx -> set of titles

    all_qas = []

    for idx, item in enumerate(dataset):
        question = item["question"]
        answer = item["answer"]
        qtype = item.get("type", "bridge")
        level = item.get("level", "medium")

        # supporting docs 추출
        supporting_docs = extract_supporting_docs(item)

        if not supporting_docs:
            continue

        # 역인덱스 구축
        for title, text in supporting_docs:
            doc_to_text[title] = text
            doc_to_qas[title].append(idx)
            qa_to_docs[idx].add(title)

        all_qas.append({
            "idx": idx,
            "question": question,
            "answer": answer,
            "type": qtype,
            "level": level,
            "supporting_titles": [d[0] for d in supporting_docs],
            "supporting_facts": item.get("supporting_facts", {}),
            "context": item["context"],
        })

        # 충분한 QA 수집
        if len(all_qas) >= num_qa * 5:  # 필터링 대비 5배 수집
            break

    print(f"  Collected {len(all_qas)} QA pairs, {len(doc_to_text)} unique documents")

    # 2. Greedy set cover: 가장 많은 QA를 커버하는 문서 우선 선택
    selected_docs = []
    covered_qas = set()
    remaining_docs = set(doc_to_text.keys())

    while len(selected_docs) < num_docs and remaining_docs:
        # 아직 커버 안 된 QA를 가장 많이 커버하는 문서 선택
        best_doc = None
        best_coverage = -1

        for doc in remaining_docs:
            # 이 문서가 커버하는 아직 커버 안 된 QA 수
            uncovered = sum(1 for qa_idx in doc_to_qas[doc] if qa_idx not in covered_qas)
            if uncovered > best_coverage:
                best_coverage = uncovered
                best_doc = doc

        if best_doc is None or best_coverage == 0:
            # 더 이상 커버할 QA가 없음
            break

        selected_docs.append(best_doc)
        remaining_docs.remove(best_doc)

        # 이 문서가 커버하는 QA 추가
        for qa_idx in doc_to_qas[best_doc]:
            covered_qas.add(qa_idx)

        if len(selected_docs) % 50 == 0:
            print(f"    Selected {len(selected_docs)} docs, covering {len(covered_qas)} QA pairs")

    print(f"  Selected {len(selected_docs)} documents covering {len(covered_qas)} QA pairs")

    # 3. Corpus 구성 (doc_id = doc_0, doc_1, ...)
    corpus = {}
    title_to_doc_id = {}

    for i, title in enumerate(selected_docs):
        doc_id = f"doc_{i}"
        corpus[doc_id] = doc_to_text[title]
        title_to_doc_id[title] = doc_id

    # 4. QA 필터링: corpus에 모든 gold doc이 있는 QA만 선택
    filtered_qas = []

    for qa in all_qas:
        # 이 QA의 supporting docs가 corpus에 있는지 확인
        gold_doc_ids = []
        for title in qa["supporting_titles"]:
            if title in title_to_doc_id:
                gold_doc_ids.append(title_to_doc_id[title])

        # 최소 1개 gold doc이 corpus에 있어야 함
        if gold_doc_ids:
            qa["gold_doc_ids"] = gold_doc_ids
            qa["gold_titles_in_corpus"] = [t for t in qa["supporting_titles"] if t in title_to_doc_id]
            filtered_qas.append(qa)

        if len(filtered_qas) >= num_qa:
            break

    # 5. Type 균형 맞추기 (선택적)
    if balance_types and len(filtered_qas) > num_qa // 2:
        bridge_qas = [q for q in filtered_qas if q["type"] == "bridge"]
        comparison_qas = [q for q in filtered_qas if q["type"] == "comparison"]

        target_each = num_qa // 2
        random.shuffle(bridge_qas)
        random.shuffle(comparison_qas)

        balanced = bridge_qas[:target_each] + comparison_qas[:target_each]
        if len(balanced) >= num_qa * 0.8:  # 80% 이상이면 balanced 사용
            filtered_qas = balanced
            print(f"  Balanced types: {len(bridge_qas[:target_each])} bridge, {len(comparison_qas[:target_each])} comparison")

    # Evidence (gold docs concat) 추가
    for qa in filtered_qas:
        evidence_parts = []
        for doc_id in qa["gold_doc_ids"][:2]:  # 최대 2개
            evidence_parts.append(corpus.get(doc_id, ""))
        qa["evidence"] = " ".join(evidence_parts)

    # 6. 통계 계산
    type_counter = Counter(q["type"] for q in filtered_qas)
    level_counter = Counter(q["level"] for q in filtered_qas)

    stats = CorpusStats(
        num_qa=len(filtered_qas),
        num_docs=len(corpus),
        avg_qa_per_doc=len(filtered_qas) / max(len(corpus), 1),
        bridge_count=type_counter.get("bridge", 0),
        comparison_count=type_counter.get("comparison", 0),
        hard_count=level_counter.get("hard", 0),
        medium_count=level_counter.get("medium", 0),
        easy_count=level_counter.get("easy", 0),
        coverage=len(filtered_qas) / len(all_qas) if all_qas else 0,
    )

    print(f"\n  Final corpus: {stats.num_docs} documents")
    print(f"  Final QA pairs: {stats.num_qa}")
    print(f"  Bridge: {stats.bridge_count}, Comparison: {stats.comparison_count}")
    print(f"  Hard: {stats.hard_count}, Medium: {stats.medium_count}, Easy: {stats.easy_count}")
    print(f"  Coverage: {stats.coverage:.2%}")

    return corpus, filtered_qas, stats


def save_corpus(
    corpus: Dict[str, str],
    qa_pairs: List[dict],
    stats: CorpusStats,
    output_dir: str,
):
    """Corpus 및 QA 저장"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Corpus 저장 (Phase 1 학습용)
    corpus_path = output_dir / "corpus.json"
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"Saved corpus: {corpus_path} ({len(corpus)} docs)")

    # 2. QA pairs 저장 (Phase 2 학습/평가용)
    qa_path = output_dir / "qa_pairs.json"

    # evidence와 context는 용량이 크므로 별도 처리
    qa_export = []
    for qa in qa_pairs:
        qa_export.append({
            "question": qa["question"],
            "answer": qa["answer"],
            "type": qa["type"],
            "level": qa["level"],
            "gold_doc_ids": qa["gold_doc_ids"],
            "gold_titles": qa.get("gold_titles_in_corpus", qa.get("supporting_titles", [])),
            "evidence": qa.get("evidence", ""),
        })

    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_export, f, ensure_ascii=False, indent=2)
    print(f"Saved QA pairs: {qa_path} ({len(qa_export)} pairs)")

    # 3. 통계 저장
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "num_qa": stats.num_qa,
            "num_docs": stats.num_docs,
            "avg_qa_per_doc": stats.avg_qa_per_doc,
            "bridge_count": stats.bridge_count,
            "comparison_count": stats.comparison_count,
            "hard_count": stats.hard_count,
            "medium_count": stats.medium_count,
            "easy_count": stats.easy_count,
            "coverage": stats.coverage,
        }, f, indent=2)
    print(f"Saved stats: {stats_path}")

    # 4. Train/Val 분할
    random.seed(42)
    indices = list(range(len(qa_export)))
    random.shuffle(indices)

    val_size = min(200, len(indices) // 5)  # 20% 또는 최대 200개
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_qa = [qa_export[i] for i in train_indices]
    val_qa = [qa_export[i] for i in val_indices]

    train_path = output_dir / "qa_train.json"
    val_path = output_dir / "qa_val.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_qa, f, ensure_ascii=False, indent=2)

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_qa, f, ensure_ascii=False, indent=2)

    print(f"Saved train/val split: {len(train_qa)} train, {len(val_qa)} val")

    return {
        "corpus_path": str(corpus_path),
        "qa_path": str(qa_path),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "stats_path": str(stats_path),
    }


def verify_corpus(output_dir: str):
    """저장된 corpus 검증"""
    output_dir = Path(output_dir)

    print("\n" + "=" * 60)
    print("Verifying saved corpus...")
    print("=" * 60)

    # Corpus 로드
    with open(output_dir / "corpus.json", "r") as f:
        corpus = json.load(f)

    # QA 로드
    with open(output_dir / "qa_pairs.json", "r") as f:
        qa_pairs = json.load(f)

    # 검증: 모든 QA의 gold_doc_ids가 corpus에 있는지
    corpus_doc_ids = set(corpus.keys())
    coverage_count = 0
    missing_docs = []

    for qa in qa_pairs:
        gold_in_corpus = [d for d in qa["gold_doc_ids"] if d in corpus_doc_ids]
        if gold_in_corpus:
            coverage_count += 1
        else:
            missing_docs.append((qa["question"][:50], qa["gold_doc_ids"]))

    coverage = coverage_count / len(qa_pairs)

    print(f"Corpus: {len(corpus)} documents")
    print(f"QA pairs: {len(qa_pairs)}")
    print(f"QA with gold in corpus: {coverage_count}/{len(qa_pairs)} ({coverage:.1%})")

    if missing_docs:
        print(f"\nWARNING: {len(missing_docs)} QA pairs have no gold docs in corpus!")
        for q, docs in missing_docs[:5]:
            print(f"  - Q: {q}... | gold_docs: {docs}")
    else:
        print("\n✓ All QA pairs have gold documents in corpus")

    # 문서당 QA 분포
    doc_qa_count = Counter()
    for qa in qa_pairs:
        for doc_id in qa["gold_doc_ids"]:
            doc_qa_count[doc_id] += 1

    most_common = doc_qa_count.most_common(5)
    least_common = doc_qa_count.most_common()[-5:] if len(doc_qa_count) >= 5 else doc_qa_count.most_common()

    print(f"\nMost referenced docs: {most_common}")
    print(f"Least referenced docs: {least_common}")

    return coverage == 1.0


def main():
    parser = argparse.ArgumentParser(description="Build corpus from QA pairs")
    parser.add_argument("--num_qa", type=int, default=1000, help="Target number of QA pairs")
    parser.add_argument("--num_docs", type=int, default=200, help="Number of documents in corpus")
    parser.add_argument("--split", type=str, default="train", help="HotpotQA split to use")
    parser.add_argument("--output_dir", type=str, default="checkpoints/phase2_corpus", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--balance_types", action="store_true", help="Balance bridge/comparison types")
    parser.add_argument("--verify_only", action="store_true", help="Only verify existing corpus")

    args = parser.parse_args()

    if args.verify_only:
        verify_corpus(args.output_dir)
        return

    # 데이터 로드
    dataset = load_hotpotqa(split=args.split)

    # Corpus 구축
    corpus, qa_pairs, stats = build_corpus_from_qa(
        dataset=dataset,
        num_qa=args.num_qa,
        num_docs=args.num_docs,
        balance_types=args.balance_types,
        seed=args.seed,
    )

    # 저장
    paths = save_corpus(corpus, qa_pairs, stats, args.output_dir)

    # 검증
    verify_corpus(args.output_dir)

    print("\n" + "=" * 60)
    print("Corpus build complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Phase 1 (Write): Train z vectors on new corpus")
    print(f"     python training/train_write_phase.py --corpus {paths['corpus_path']}")
    print(f"  2. Phase 2 (Read): Train evidence generation")
    print(f"     python training/train_read_phase.py --qa_train {paths['train_path']}")


if __name__ == "__main__":
    main()
