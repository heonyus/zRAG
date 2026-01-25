"""
데이터셋 다운로드 모듈
- Natural Questions (NQ)
- HotpotQA
- Wikipedia corpus (FlashRAG 표준)
"""

import os
from pathlib import Path
from datasets import load_dataset


def download_dataset(dataset_name: str, save_dir: str = "./data/raw") -> dict:
    """HuggingFace datasets에서 데이터셋 다운로드"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if dataset_name == "natural_questions":
        return _download_nq(save_path)
    elif dataset_name == "hotpot_qa":
        return _download_hotpotqa(save_path)
    elif dataset_name == "triviaqa":
        return _download_triviaqa(save_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _download_nq(save_path: Path) -> dict:
    """Natural Questions 다운로드 (FlashRAG 전처리 버전 우선 시도)"""
    cache_file = save_path / "nq_dataset.arrow"

    # FlashRAG preprocessed version (추천)
    try:
        dataset = load_dataset(
            "RUC-NLPIR/FlashRAG_datasets",
            "nq",
            trust_remote_code=True,
        )
        print(f"[NQ] FlashRAG version loaded: "
              f"train={len(dataset['train'])}, "
              f"dev={len(dataset.get('dev', []))}, "
              f"test={len(dataset.get('test', []))}")
        return dataset
    except Exception as e:
        print(f"[NQ] FlashRAG version failed ({e}), trying original...")

    # Fallback: original NQ
    dataset = load_dataset("google-research-datasets/natural_questions", "default")
    print(f"[NQ] Original version loaded: {len(dataset['train'])} train, "
          f"{len(dataset['validation'])} validation")
    return dataset


def _download_hotpotqa(save_path: Path) -> dict:
    """HotpotQA 다운로드 (distractor setting)"""
    dataset = load_dataset("hotpot_qa", "distractor")
    print(f"[HotpotQA] Loaded: train={len(dataset['train'])}, "
          f"validation={len(dataset['validation'])}")
    return dataset


def _download_triviaqa(save_path: Path) -> dict:
    """TriviaQA 다운로드"""
    dataset = load_dataset("trivia_qa", "rc")
    print(f"[TriviaQA] Loaded: train={len(dataset['train'])}, "
          f"validation={len(dataset['validation'])}")
    return dataset


def download_corpus(corpus_type: str = "wikipedia", save_dir: str = "./data/corpus") -> Path:
    """검색 코퍼스 다운로드"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if corpus_type == "wikipedia":
        return _download_wiki_corpus(save_path)
    else:
        raise ValueError(f"Unknown corpus type: {corpus_type}")


def _download_wiki_corpus(save_path: Path) -> Path:
    """Wikipedia December 2018 dump (FlashRAG 표준 100-word passages)"""
    corpus_file = save_path / "wiki_corpus.jsonl"

    if corpus_file.exists():
        print(f"[Corpus] Already exists: {corpus_file}")
        return corpus_file

    # FlashRAG의 pre-built Wikipedia corpus 사용
    try:
        corpus = load_dataset(
            "RUC-NLPIR/FlashRAG_datasets",
            "wiki18_100w",
            trust_remote_code=True,
        )
        print(f"[Corpus] FlashRAG Wikipedia loaded: {len(corpus['train'])} passages")
        return corpus
    except Exception as e:
        print(f"[Corpus] FlashRAG corpus failed ({e})")
        print("[Corpus] Please download manually from FlashRAG repository")
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="natural_questions",
                        choices=["natural_questions", "hotpot_qa", "triviaqa"])
    parser.add_argument("--save_dir", type=str, default="./data/raw")
    parser.add_argument("--download_corpus", action="store_true")
    args = parser.parse_args()

    download_dataset(args.dataset, args.save_dir)
    if args.download_corpus:
        download_corpus(save_dir="./data/corpus")
