"""
Phase 2 Track A: Evidence → Answer 분리 평가

zRAG vs RAG (BM25, Dense, Contriever) 비교 실험

파이프라인:
1. Phase 1 checkpoint 로드 (학습된 z vectors)
2. QA 데이터 로드 + Phase 1 corpus 필터링 (누수 방지)
3. Stage 1: Evidence 생성
   - zRAG: Query + Z prefix → Evidence 텍스트 생성
   - Baselines: Query → Retrieve Top-K → Evidence 텍스트 (concat)
4. Cost matching: Evidence 토큰 상한 적용
5. Stage 1 평가: Evidence 품질 (ROUGE-L, Coverage, Faithfulness)
6. Stage 2: Answer 생성 (완전 고정 Reader)
7. Stage 2 평가: Answer 정확도 (EM, F1)
8. Efficiency 측정
9. 결과 저장 및 비교 테이블 생성
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

# Path setup
sys.path.append(str(Path(__file__).parent.parent))

from data.filter_qa import load_phase1_doc_ids, filter_qa_for_phase1_corpus
from data.download import download_dataset
from data.preprocess import preprocess_nq, preprocess_hotpotqa
from training.train_write_phase import prepare_corpus
from experiments.fixed_reader import FixedReader, create_reader_from_config
from evaluation.metrics import compute_em, compute_f1, aggregate_metrics, extract_answer_span
from evaluation.evidence_metrics import evaluate_evidence_quality
from evaluation.faithfulness import evaluate_faithfulness_batch
from baselines.standard_rag import StandardRAGBaseline
from baselines.contriever_rag import ContrieverRAGBaseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# JSONL 기반 샘플 단위 캐시 (crash-safe)
# ============================================================

def append_jsonl(path: Path, record: dict):
    """JSONL 파일에 레코드 추가 (flush + fsync로 안전 저장)"""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_jsonl_cache(path: Path) -> Dict[int, dict]:
    """JSONL 캐시 로드 → {idx: record} dict 반환"""
    cache = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    idx = record.get("idx")
                    if idx is not None:
                        cache[idx] = record
                except json.JSONDecodeError:
                    continue  # 깨진 라인은 무시
    return cache


def atomic_json_save(path: Path, data: dict):
    """Atomic JSON 저장 (tmp → rename)"""
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def atomic_csv_save(path: Path, rows: List[List[str]], header: List[str]):
    """Atomic CSV 저장 (tmp → rename)"""
    import csv
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def load_config(config_path: str) -> OmegaConf:
    """Config 로드"""
    with open(config_path, "r") as f:
        config = OmegaConf.create(yaml.safe_load(f))
    return config


def load_phase1_checkpoint(checkpoint_dir: str) -> Dict:
    """
    Phase 1/2 checkpoint 로드 (여러 형식 지원)

    Supported formats:
    - Phase 1: z_pool_epoch50.pt / z_pool.pt with {"z_vectors": {doc_id: tensor, ...}}
    - Phase 2: best.pt / final.pt with {"memory_pool": tensor, "doc_ids": [...]}

    Returns:
        Dict with z_vectors, alpha, projection, doc_ids
    """
    checkpoint_dir = Path(checkpoint_dir)

    # 체크포인트 파일 우선순위
    checkpoint_candidates = [
        checkpoint_dir / "best.pt",           # Phase 2 best
        checkpoint_dir / "final.pt",          # Phase 2 final
        checkpoint_dir / "z_pool_epoch50.pt", # Phase 1
        checkpoint_dir / "z_pool.pt",         # Phase 1 final
    ]

    z_pool_path = None
    for candidate in checkpoint_candidates:
        if candidate.exists():
            z_pool_path = candidate
            break

    if z_pool_path is None:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    logger.info(f"Loading checkpoint: {z_pool_path}")

    ckpt = torch.load(z_pool_path, map_location="cpu")

    # Phase 2 format: memory_pool (tensor) + doc_ids (list)
    if "memory_pool" in ckpt:
        memory_pool = ckpt["memory_pool"]  # [num_docs, m_tokens, z_dim]
        doc_ids = ckpt.get("doc_ids", [f"doc_{i}" for i in range(memory_pool.shape[0])])

        # Convert tensor to dict format for compatibility
        z_vectors = {doc_id: memory_pool[i] for i, doc_id in enumerate(doc_ids)}

        result = {
            "z_vectors": z_vectors,
            "alpha": ckpt.get("alpha", 1.0),
            "projection": ckpt.get("z_to_embedding", None),
            "doc_ids": doc_ids,
        }
        logger.info(f"  Loaded Phase 2 checkpoint ({len(doc_ids)} docs)")

    # Phase 1 format: z_vectors (dict)
    elif "z_vectors" in ckpt:
        result = {
            "z_vectors": ckpt["z_vectors"],
            "alpha": ckpt.get("alpha", 1.0),
            "projection": ckpt.get("z_to_embedding", None),
            "doc_ids": list(ckpt["z_vectors"].keys()),
        }
        logger.info(f"  Loaded Phase 1 checkpoint ({len(result['doc_ids'])} docs)")

    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {list(ckpt.keys())}")

    logger.info(f"  Alpha: {result['alpha']:.4f}")
    logger.info(f"  Projection: {'found' if result['projection'] else 'not found'}")

    return result


def load_corpus_and_qa_from_phase2(
    corpus_dir: str = "checkpoints/phase2_corpus",
) -> Tuple[Dict[str, str], List[Dict]]:
    """
    Phase 2 corpus_builder.py로 생성한 corpus와 QA를 직접 로드

    이 방식이 정확합니다:
    - corpus.json: Phase 1/2 학습에 사용된 정확한 문서들
    - qa_val.json: gold_doc_ids가 100% corpus 내에 존재

    Args:
        corpus_dir: phase2_corpus 디렉토리 경로

    Returns:
        Tuple of (corpus dict, qa_pairs list)
    """
    corpus_dir = Path(corpus_dir)
    corpus_path = corpus_dir / "corpus.json"
    qa_val_path = corpus_dir / "qa_val.json"

    if not corpus_path.exists():
        raise FileNotFoundError(f"corpus.json not found: {corpus_path}")
    if not qa_val_path.exists():
        raise FileNotFoundError(f"qa_val.json not found: {qa_val_path}")

    logger.info(f"Loading Phase 2 corpus from {corpus_dir}...")

    # Corpus 로드
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    logger.info(f"  Loaded corpus: {len(corpus)} documents")

    # QA 로드 (validation set)
    with open(qa_val_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    # qa_data는 {"data": [...]} 형식일 수 있음
    if isinstance(qa_data, dict) and "data" in qa_data:
        qa_list = qa_data["data"]
    elif isinstance(qa_data, list):
        qa_list = qa_data
    else:
        raise ValueError(f"Unknown qa_val.json format: {type(qa_data)}")

    # QA pairs 변환 (Track A 형식으로)
    qa_pairs = []
    for item in qa_list:
        gold_doc_ids = item.get("gold_doc_ids", [])
        # gold_doc_ids가 문자열인 경우 리스트로 변환
        if isinstance(gold_doc_ids, str):
            gold_doc_ids = [gold_doc_ids]

        # context 구성 (gold docs의 텍스트)
        context_parts = []
        for doc_id in gold_doc_ids:
            if doc_id in corpus:
                context_parts.append(corpus[doc_id])
        context = " ".join(context_parts)

        qa_pairs.append({
            "question": item["question"],
            "answer": item["answer"],
            "gold_doc_ids": gold_doc_ids,
            "gold_titles": item.get("gold_titles", []),
            "context": context,
            "evidence": item.get("evidence", context),  # Phase 2 학습에서 사용한 evidence
        })

    # 검증: gold_doc_ids가 corpus에 있는지
    qa_with_gold = sum(1 for qa in qa_pairs if any(d in corpus for d in qa["gold_doc_ids"]))
    logger.info(f"  Loaded {len(qa_pairs)} QA pairs (validation)")
    logger.info(f"  QA with gold in corpus: {qa_with_gold} ({qa_with_gold/len(qa_pairs)*100:.1f}%)")

    # 처음 3개 샘플 출력
    for i, qa in enumerate(qa_pairs[:3]):
        logger.info(f"  [Sample {i}] Q: {qa['question'][:50]}...")
        logger.info(f"    gold_doc_ids: {qa['gold_doc_ids']}")
        logger.info(f"    answer: {qa['answer']}")

    return corpus, qa_pairs


def load_corpus_for_phase1(
    phase1_doc_ids: List[str],
    dataset_name: str = "hotpot_qa",
    num_docs: int = 200,
    corpus_dir: str = None,
) -> Tuple[Dict[str, str], List[Dict]]:
    """
    Corpus와 QA 로드 (Phase 2 corpus 우선 사용)

    Args:
        phase1_doc_ids: Phase 1에서 학습한 문서 ID 목록
        dataset_name: 데이터셋 이름
        num_docs: 문서 수
        corpus_dir: Phase 2 corpus 디렉토리 (있으면 우선 사용)

    Returns:
        Tuple of (corpus dict, qa_pairs list)
    """
    # Phase 2 corpus가 있으면 우선 사용 (100% gold doc coverage 보장)
    if corpus_dir is not None:
        corpus_dir_path = Path(corpus_dir)
        if (corpus_dir_path / "corpus.json").exists():
            return load_corpus_and_qa_from_phase2(corpus_dir)

    # Fallback: 기존 방식 (corpus 재구성) - 권장하지 않음
    logger.warning("=" * 60)
    logger.warning("WARNING: Using legacy corpus rebuild method!")
    logger.warning("This may cause gold doc mismatch (low coverage)")
    logger.warning("Recommend: use corpus_dir from phase2_corpus")
    logger.warning("=" * 60)

    logger.info(f"Rebuilding Phase 1 corpus from {dataset_name}...")

    dataset = download_dataset(dataset_name)

    # Phase 1과 동일한 prepare_corpus 사용
    corpus = prepare_corpus(dataset, max_docs=num_docs, dataset_name=dataset_name)

    # Phase 1 doc_ids와 일치 확인
    corpus_doc_ids = set(corpus.keys())
    phase1_set = set(phase1_doc_ids)
    matching = corpus_doc_ids & phase1_set

    logger.info(f"  Rebuilt corpus: {len(corpus)} documents")
    logger.info(f"  Matching Phase 1: {len(matching)}/{len(phase1_doc_ids)}")

    # corpus의 title → doc_id 매핑 생성
    title_to_doc_id = {}
    for doc_id, doc_text in corpus.items():
        # doc_text 첫 줄이 title
        first_line = doc_text.split("\n")[0].strip()
        title_to_doc_id[first_line] = doc_id
        # 정규화된 버전도 추가 (소문자, 공백 정리)
        title_to_doc_id[first_line.lower().strip()] = doc_id

    logger.info(f"  Built title->doc_id mapping: {len(title_to_doc_id)} entries")

    # QA pairs 추출 (HotpotQA용)
    qa_pairs = []
    qa_with_gold = 0

    if dataset_name == "hotpot_qa":
        if hasattr(dataset, "keys") and "train" in dataset.keys():
            data = dataset["train"]
        else:
            data = dataset

        for idx, item in enumerate(data):
            if "question" not in item or "answer" not in item:
                continue

            # gold_doc_ids 추출 (supporting_facts에서)
            gold_doc_ids = []
            gold_titles = set()

            if "supporting_facts" in item:
                sf = item["supporting_facts"]
                # HotpotQA format: supporting_facts = {"title": [...], "sent_id": [...]}
                if isinstance(sf, dict):
                    titles = sf.get("title", [])
                    gold_titles = set(titles)
                # 또는 list of [title, sent_id] pairs
                elif isinstance(sf, list):
                    for pair in sf:
                        if isinstance(pair, (list, tuple)) and len(pair) >= 1:
                            gold_titles.add(pair[0])

            # gold_titles를 corpus doc_id로 변환
            for title in gold_titles:
                # exact match
                if title in title_to_doc_id:
                    gold_doc_ids.append(title_to_doc_id[title])
                # 소문자 match
                elif title.lower().strip() in title_to_doc_id:
                    gold_doc_ids.append(title_to_doc_id[title.lower().strip()])

            gold_doc_ids = list(set(gold_doc_ids))  # 중복 제거

            # 디버그: 처음 3개 샘플 출력
            if idx < 3:
                logger.info(f"  [Sample {idx}] Q: {item['question'][:50]}...")
                logger.info(f"    gold_titles: {gold_titles}")
                logger.info(f"    gold_doc_ids: {gold_doc_ids}")

            qa_pairs.append({
                "question": item["question"],
                "answer": item["answer"],
                "gold_doc_ids": gold_doc_ids,
                "gold_titles": list(gold_titles),
                "context": " ".join([corpus.get(d, "") for d in gold_doc_ids[:2]]),
            })

            if gold_doc_ids:
                qa_with_gold += 1

            if len(qa_pairs) >= 5000:  # 충분한 QA 수집
                break

    logger.info(f"  Extracted {len(qa_pairs)} QA pairs")
    logger.info(f"  QA with gold in corpus: {qa_with_gold} ({qa_with_gold/len(qa_pairs)*100:.1f}%)")

    return corpus, qa_pairs


def truncate_evidence(text: str, max_tokens: int, tokenizer) -> str:
    """Evidence를 max_tokens로 truncate"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.decode(tokens, skip_special_tokens=True)


def generate_zrag_evidence_jsonl(
    qa_pairs: List[Dict],
    config: OmegaConf,
    cache_dir: Path,
    device: str = "cuda",
    start_idx: int = 0,
    end_idx: int = None,
) -> List[str]:
    """
    zRAG로 Evidence 생성 (JSONL 샘플 단위 캐시 - crash-safe)

    각 샘플 완료 시 즉시 JSONL에 저장 (flush+fsync)
    VM 끊겨도 완료된 샘플은 보존됨

    Args:
        qa_pairs: QA 데이터
        config: 설정
        cache_dir: 캐시 디렉토리
        device: 디바이스
        start_idx: 시작 인덱스 (sharding용)
        end_idx: 종료 인덱스 (sharding용, None이면 끝까지)

    Returns:
        생성된 Evidence 텍스트 리스트 (전체 qa_pairs 길이, 미완료는 빈 문자열)
    """
    from models.parametric_memory_llm import ParametricMemoryLLM

    cache_path = cache_dir / "evidence_zrag.jsonl"

    # 기존 캐시 로드 → 완료된 idx set
    cache = load_jsonl_cache(cache_path)
    done_indices = set(cache.keys())
    logger.info(f"zRAG cache loaded: {len(done_indices)} samples completed")

    # 범위 결정
    if end_idx is None:
        end_idx = len(qa_pairs)

    # 처리할 인덱스 목록 (아직 안 한 것만)
    todo_indices = [i for i in range(start_idx, end_idx) if i not in done_indices]

    if not todo_indices:
        logger.info(f"zRAG evidence already complete for range [{start_idx}, {end_idx})")
        # 캐시에서 결과 조립
        evidences = [""] * len(qa_pairs)
        for idx, record in cache.items():
            if 0 <= idx < len(qa_pairs):
                evidences[idx] = record.get("evidence", "")
        return evidences

    logger.info(f"Generating zRAG evidence: {len(todo_indices)} remaining (range [{start_idx}, {end_idx}))")

    # 모델 초기화
    model = ParametricMemoryLLM(
        llm_name=config.model.llm_name,
        num_docs=config.memory.num_docs,
        z_dim=config.memory.z_dim,
        m_tokens=config.memory.m_tokens,
        quantization=config.model.quantization,
        device=device,
    )

    # Phase 1/2 checkpoint 로드
    checkpoint_dir = Path(config.phase1.checkpoint_dir)
    checkpoint_candidates = [
        checkpoint_dir / "best.pt",
        checkpoint_dir / "final.pt",
        checkpoint_dir / "z_pool_epoch50.pt",
        checkpoint_dir / "z_pool.pt",
    ]

    z_pool_path = None
    for candidate in checkpoint_candidates:
        if candidate.exists():
            z_pool_path = candidate
            break

    if z_pool_path is None:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    projection_path = checkpoint_dir / "projection.pt"
    logger.info(f"Loading checkpoint from: {z_pool_path}")

    # LoRA 경로 결정
    lora_path = None
    lora_enabled = config.model.get("lora", {}).get("enabled", False)
    if lora_enabled:
        explicit_lora_path = config.model.get("lora", {}).get("path", None)
        if explicit_lora_path:
            lora_path = explicit_lora_path
        else:
            auto_lora_path = str(z_pool_path) + "_lora"
            if Path(auto_lora_path).exists():
                lora_path = auto_lora_path
        logger.info(f"LoRA enabled: {lora_enabled}, path: {lora_path}")
    else:
        logger.info("LoRA disabled in config")

    model.load_from_phase1(
        z_pool_path=str(z_pool_path),
        projection_path=str(projection_path) if projection_path.exists() else None,
        lora_path=lora_path,
    )
    model.eval()

    top_k_docs = config.get("zrag", {}).get("top_k_docs", None)
    logger.info(f"zRAG config: top_k_docs={top_k_docs}")

    # Evidence 생성 (샘플 단위 저장)
    for idx in tqdm(todo_indices, desc="zRAG Evidence"):
        qa = qa_pairs[idx]
        query = qa["question"]

        query_ids = model.tokenizer(
            query,
            return_tensors="pt",
            max_length=128,
            truncation=True,
        )["input_ids"].to(device)

        evidence = model.generate_evidence(
            query_ids=query_ids,
            max_new_tokens=config.evidence_budget.max_tokens,
            top_k_docs=top_k_docs,
            debug=(idx < 3),
        )

        # 즉시 JSONL에 저장 (crash-safe)
        record = {
            "idx": idx,
            "query": query,
            "gold_doc_ids": qa.get("gold_doc_ids", []),
            "evidence": evidence,
        }
        append_jsonl(cache_path, record)
        cache[idx] = record

    # 결과 조립
    evidences = [""] * len(qa_pairs)
    for idx, record in cache.items():
        if 0 <= idx < len(qa_pairs):
            evidences[idx] = record.get("evidence", "")

    logger.info(f"zRAG evidence complete: {sum(1 for e in evidences if e)}/{len(qa_pairs)}")
    return evidences


# Legacy wrapper for backward compatibility
def generate_zrag_evidence(
    qa_pairs: List[Dict],
    phase1_ckpt: Dict,
    config: OmegaConf,
    device: str = "cuda",
    existing_evidences: List[str] = None,
    cache_path: Path = None,
    all_evidences: Dict = None,
    save_every: int = 10,
) -> List[str]:
    """Legacy wrapper - redirects to JSONL version"""
    cache_dir = cache_path.parent if cache_path else Path(config.phase1.checkpoint_dir).parent / "phase2_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return generate_zrag_evidence_jsonl(qa_pairs, config, cache_dir, device)


def generate_baseline_evidence_jsonl(
    qa_pairs: List[Dict],
    corpus: Dict[str, str],
    baseline_type: str,
    config: OmegaConf,
    cache_dir: Path,
    device: str = "cuda",
) -> List[str]:
    """
    Baseline RAG로 Evidence 생성 (JSONL 샘플 단위 캐시 - crash-safe)

    각 샘플 완료 시 즉시 JSONL에 저장 (flush+fsync)
    VM 끊겨도 완료된 샘플은 보존됨

    Args:
        qa_pairs: QA 데이터
        corpus: 문서 corpus
        baseline_type: "bm25", "dense_e5", "contriever"
        config: 설정
        cache_dir: 캐시 디렉토리
        device: 디바이스

    Returns:
        Evidence 텍스트 리스트
    """
    cache_path = cache_dir / f"evidence_{baseline_type}.jsonl"

    # 기존 캐시 로드 → 완료된 idx set
    cache = load_jsonl_cache(cache_path)
    done_indices = set(cache.keys())
    logger.info(f"{baseline_type} cache loaded: {len(done_indices)} samples completed")

    # 처리할 인덱스 목록 (아직 안 한 것만)
    todo_indices = [i for i in range(len(qa_pairs)) if i not in done_indices]

    if not todo_indices:
        logger.info(f"{baseline_type} evidence already complete ({len(done_indices)}/{len(qa_pairs)})")
        # 캐시에서 결과 조립
        evidences = [""] * len(qa_pairs)
        for idx, record in cache.items():
            if 0 <= idx < len(qa_pairs):
                evidences[idx] = record.get("evidence", "")
        return evidences

    logger.info(f"Generating {baseline_type} evidence: {len(todo_indices)} remaining")

    top_k = config.evidence_budget.top_k

    # Retriever 초기화
    if baseline_type == "bm25":
        rag = StandardRAGBaseline(
            llm_name=config.model.llm_name,
            retriever_type="bm25",
            quantization=config.model.quantization,
            device=device,
        )
    elif baseline_type == "dense_e5":
        rag = StandardRAGBaseline(
            llm_name=config.model.llm_name,
            retriever_type="dense",
            retriever_name="intfloat/e5-base-v2",
            quantization=config.model.quantization,
            device=device,
        )
    elif baseline_type == "contriever":
        rag = ContrieverRAGBaseline(
            llm_name=config.model.llm_name,
            quantization=config.model.quantization,
            device=device,
        )
    else:
        raise ValueError(f"Unknown baseline: {baseline_type}")

    # 인덱스 구축
    rag.build_index(corpus)

    # Evidence 생성 (샘플 단위 저장)
    for idx in tqdm(todo_indices, desc=f"{baseline_type} Evidence"):
        qa = qa_pairs[idx]
        query = qa["question"]
        retrieved = rag.retrieve(query, k=top_k)

        # Top-K 문서 concat
        evidence = " ".join([r["text"] for r in retrieved])

        # 즉시 JSONL에 저장 (crash-safe)
        record = {
            "idx": idx,
            "query": query,
            "gold_doc_ids": qa.get("gold_doc_ids", []),
            "evidence": evidence,
            "retrieved_doc_ids": [r.get("doc_id", "") for r in retrieved],
        }
        append_jsonl(cache_path, record)
        cache[idx] = record

    # 결과 조립
    evidences = [""] * len(qa_pairs)
    for idx, record in cache.items():
        if 0 <= idx < len(qa_pairs):
            evidences[idx] = record.get("evidence", "")

    logger.info(f"{baseline_type} evidence complete: {sum(1 for e in evidences if e)}/{len(qa_pairs)}")
    return evidences


# Legacy wrapper for backward compatibility
def generate_baseline_evidence(
    qa_pairs: List[Dict],
    corpus: Dict[str, str],
    baseline_type: str,
    config: OmegaConf,
    device: str = "cuda",
    existing_evidences: List[str] = None,
    cache_path: Path = None,
    all_evidences: Dict = None,
    method_name: str = None,
    save_every: int = 10,
) -> List[str]:
    """Legacy wrapper - redirects to JSONL version"""
    cache_dir = cache_path.parent if cache_path else Path(config.phase1.checkpoint_dir).parent / "phase2_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return generate_baseline_evidence_jsonl(qa_pairs, corpus, baseline_type, config, cache_dir, device)


def generate_answers_jsonl(
    qa_pairs: List[Dict],
    evidences: List[str],
    method_name: str,
    cache_dir: Path,
    reader,  # FixedReader instance
) -> List[str]:
    """
    Answer 생성 (JSONL 샘플 단위 캐시 - crash-safe)

    각 샘플 완료 시 즉시 JSONL에 저장 (flush+fsync)
    VM 끊겨도 완료된 샘플은 보존됨

    Args:
        qa_pairs: QA 데이터
        evidences: Evidence 텍스트 리스트
        method_name: 메소드 이름 (zRAG, bm25, dense_e5, contriever)
        cache_dir: 캐시 디렉토리
        reader: FixedReader 인스턴스

    Returns:
        생성된 Answer 텍스트 리스트
    """
    cache_path = cache_dir / f"answer_{method_name}.jsonl"

    # 기존 캐시 로드 → 완료된 idx set
    cache = load_jsonl_cache(cache_path)
    done_indices = set(cache.keys())
    logger.info(f"{method_name} answer cache loaded: {len(done_indices)} samples completed")

    # 처리할 인덱스 목록 (아직 안 한 것만)
    todo_indices = [i for i in range(len(qa_pairs)) if i not in done_indices]

    if not todo_indices:
        logger.info(f"{method_name} answers already complete ({len(done_indices)}/{len(qa_pairs)})")
        # 캐시에서 결과 조립
        answers = [""] * len(qa_pairs)
        for idx, record in cache.items():
            if 0 <= idx < len(qa_pairs):
                answers[idx] = record.get("answer", "")
        return answers

    logger.info(f"Generating {method_name} answers: {len(todo_indices)} remaining")

    # Answer 생성 (샘플 단위 저장)
    for idx in tqdm(todo_indices, desc=f"{method_name} Answers"):
        qa = qa_pairs[idx]
        evidence = evidences[idx]
        query = qa["question"]

        answer = reader.generate_answer(query, evidence)

        # 즉시 JSONL에 저장 (crash-safe)
        record = {
            "idx": idx,
            "query": query,
            "evidence_snippet": evidence[:200] if len(evidence) > 200 else evidence,
            "answer": answer,
            "gold_answer": qa.get("answer", ""),
        }
        append_jsonl(cache_path, record)
        cache[idx] = record

    # 결과 조립
    answers = [""] * len(qa_pairs)
    for idx, record in cache.items():
        if 0 <= idx < len(qa_pairs):
            answers[idx] = record.get("answer", "")

    logger.info(f"{method_name} answers complete: {sum(1 for a in answers if a)}/{len(qa_pairs)}")
    return answers


def evaluate_stage1(
    evidences: Dict[str, List[str]],
    qa_pairs: List[Dict],
    config: OmegaConf,
) -> Dict[str, Dict]:
    """
    Stage 1 평가: Evidence 품질

    Args:
        evidences: {method: [evidence, ...]}
        qa_pairs: QA 데이터 (gold evidence 포함)

    Returns:
        {method: {rouge_l, coverage, faithfulness}}
    """
    logger.info("Evaluating Stage 1 (Evidence quality)...")

    results = {}

    for method, evidence_list in evidences.items():
        logger.info(f"  Evaluating {method}...")

        # Gold evidence (있으면)
        gold_evidences = [qa.get("evidence", qa.get("context", "")) for qa in qa_pairs]
        answers_list = [qa.get("answer", "") for qa in qa_pairs]
        questions = [qa["question"] for qa in qa_pairs]

        # Evidence metrics (ROUGE-L, etc.) - 샘플별로 계산 후 평균
        rouge_l_scores = []
        coverage_scores = []

        for gen_ev, gold_ev, ans in zip(evidence_list, gold_evidences, answers_list):
            metrics = evaluate_evidence_quality(
                generated_evidence=gen_ev,
                gold_evidence=gold_ev,
                answer=ans,
            )
            rouge_l_scores.append(metrics.get("rouge_l_f1", 0.0))
            coverage_scores.append(metrics.get("answer_coverage", 0.0))

        avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0
        avg_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0

        # Faithfulness
        faithfulness_stats = evaluate_faithfulness_batch(
            evidences=evidence_list,
            answers=answers_list,
            questions=questions,
            show_progress=False,
        )

        results[method] = {
            "rouge_l": avg_rouge_l,
            "answer_coverage": avg_coverage,
            "faithfulness": faithfulness_stats.get("faithfulness", 0.0),
            "containment": faithfulness_stats.get("containment", 0.0),
        }

    return results


def evaluate_stage2(
    answers: Dict[str, List[str]],
    qa_pairs: List[Dict],
) -> Dict[str, Dict]:
    """
    Stage 2 평가: Answer 정확도

    Args:
        answers: {method: [answer, ...]}
        qa_pairs: QA 데이터 (gold answer 포함)

    Returns:
        {method: {em, f1}}
    """
    logger.info("Evaluating Stage 2 (Answer accuracy)...")

    results = {}
    gold_answers = [qa.get("answer", "") for qa in qa_pairs]

    for method, answer_list in answers.items():
        logger.info(f"  Evaluating {method}...")

        all_metrics = []
        for i, (pred, gold) in enumerate(zip(answer_list, gold_answers)):
            # 모델 출력에서 답 span만 추출 (EM=0 문제 해결)
            pred_extracted = extract_answer_span(pred)
            em = compute_em(pred_extracted, gold)
            f1 = compute_f1(pred_extracted, gold)
            all_metrics.append({"em": em, "f1": f1})

            # 디버그: 첫 3개 샘플만 로그 출력 (extract_answer_span 동작 확인)
            if i < 3:
                logger.info(f"    [DBG {i}] raw: {pred[:80]}...")
                logger.info(f"    [DBG {i}] extracted: '{pred_extracted}'")
                logger.info(f"    [DBG {i}] gold: '{gold}' | EM={em}, F1={f1:.4f}")

        aggregated = aggregate_metrics(all_metrics)
        results[method] = {
            "em": aggregated["em"],
            "f1": aggregated["f1"],
        }

    return results


def measure_efficiency(
    method: str,
    qa_pairs: List[Dict],
    evidences: List[str],
    answers: List[str],
    tokenizer,
) -> Dict:
    """Efficiency 측정"""
    # Total tokens (evidence + query)
    total_tokens = 0
    for qa, evidence in zip(qa_pairs, evidences):
        query_tokens = len(tokenizer.encode(qa["question"]))
        evidence_tokens = len(tokenizer.encode(evidence))
        total_tokens += query_tokens + evidence_tokens

    avg_tokens = total_tokens / len(qa_pairs)

    return {
        "total_tokens": total_tokens,
        "avg_tokens_per_sample": avg_tokens,
    }


def save_results(
    stage1_results: Dict,
    stage2_results: Dict,
    efficiency_results: Dict,
    config: OmegaConf,
    evidence_samples: Dict,
    answer_samples: Dict,
):
    """결과 저장"""
    results_dir = Path(config.logging.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 비교 테이블 (CSV)
    import csv
    csv_path = results_dir / f"comparison_table_{timestamp}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Method",
            "ROUGE-L", "Coverage", "Faithfulness",  # Stage 1
            "EM", "F1",  # Stage 2
            "Avg Tokens",  # Efficiency
        ])

        for method in stage1_results.keys():
            s1 = stage1_results[method]
            s2 = stage2_results.get(method, {})
            eff = efficiency_results.get(method, {})

            writer.writerow([
                method,
                f"{s1.get('rouge_l', 0):.4f}",
                f"{s1.get('answer_coverage', 0):.4f}",
                f"{s1.get('faithfulness', 0):.4f}",
                f"{s2.get('em', 0):.4f}",
                f"{s2.get('f1', 0):.4f}",
                f"{eff.get('avg_tokens_per_sample', 0):.1f}",
            ])

    logger.info(f"Saved comparison table: {csv_path}")

    # 2. 상세 결과 (JSON)
    json_path = results_dir / f"full_results_{timestamp}.json"

    full_results = {
        "timestamp": timestamp,
        "config": OmegaConf.to_container(config),
        "stage1": stage1_results,
        "stage2": stage2_results,
        "efficiency": efficiency_results,
    }

    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=2)

    logger.info(f"Saved full results: {json_path}")

    # 3. Evidence 샘플 저장
    if evidence_samples:
        samples_path = results_dir / f"evidence_samples_{timestamp}.json"
        with open(samples_path, "w", encoding="utf-8") as f:
            json.dump(evidence_samples, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved evidence samples: {samples_path}")

    # 4. Answer 샘플 저장
    if answer_samples:
        samples_path = results_dir / f"answer_samples_{timestamp}.json"
        with open(samples_path, "w", encoding="utf-8") as f:
            json.dump(answer_samples, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved answer samples: {samples_path}")


def print_results_table(
    stage1_results: Dict,
    stage2_results: Dict,
    efficiency_results: Dict,
):
    """결과 테이블 출력"""
    print("\n" + "=" * 80)
    print("PHASE 2 TRACK A RESULTS")
    print("=" * 80)

    header = f"{'Method':<12} | {'ROUGE-L':>8} {'Cover':>8} {'Faith':>8} | {'EM':>8} {'F1':>8} | {'Tokens':>8}"
    print(header)
    print("-" * 80)

    for method in stage1_results.keys():
        s1 = stage1_results[method]
        s2 = stage2_results.get(method, {})
        eff = efficiency_results.get(method, {})

        row = f"{method:<12} | {s1.get('rouge_l', 0):>8.4f} {s1.get('answer_coverage', 0):>8.4f} {s1.get('faithfulness', 0):>8.4f} | {s2.get('em', 0):>8.4f} {s2.get('f1', 0):>8.4f} | {eff.get('avg_tokens_per_sample', 0):>8.1f}"
        print(row)

    print("=" * 80)


def main(config_path: str, dry_run: bool = False, no_cache: bool = False, save_every: int = 10):
    """메인 실험 함수"""
    config = load_config(config_path)

    # CLI에서 전달된 save_every 값을 config에 반영
    if "checkpoint" not in config:
        config["checkpoint"] = {}
    config["checkpoint"]["save_every"] = save_every

    logger.info("=" * 60)
    logger.info("Phase 2 Track A: zRAG vs RAG Comparison")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # 캐시 삭제 (--no_cache 옵션)
    if no_cache:
        cache_dir = Path(config.phase1.checkpoint_dir).parent / "phase2_cache"
        if cache_dir.exists():
            import shutil
            logger.info(f"Deleting cache directory: {cache_dir}")
            shutil.rmtree(cache_dir)
            logger.info("  Cache deleted - will regenerate evidence/answers")

    # 1. Phase 1 checkpoint 로드
    logger.info("\n[Step 1] Loading Phase 1 checkpoint...")
    phase1_ckpt = load_phase1_checkpoint(config.phase1.checkpoint_dir)
    phase1_doc_ids = set(phase1_ckpt["doc_ids"])

    # 2. Corpus 및 QA 로드
    # Phase 2 corpus_dir이 있으면 사용 (권장 - 100% gold coverage)
    logger.info("\n[Step 2] Loading corpus and QA data...")
    corpus_dir = config.data.get("corpus_dir", "checkpoints/phase2_corpus")

    corpus, all_qa_pairs = load_corpus_for_phase1(
        phase1_doc_ids=list(phase1_doc_ids),
        dataset_name=config.data.dataset,
        num_docs=config.memory.num_docs,
        corpus_dir=corpus_dir,  # Phase 2 corpus 우선 사용
    )

    # 3. QA 필터링 (Phase 2 corpus 사용 시 이미 100% coverage이므로 skip 가능)
    logger.info("\n[Step 3] Filtering QA pairs for Phase 1 corpus...")

    # Phase 2 corpus 사용 시: 이미 100% coverage이므로 필터링 skip
    if corpus_dir and (Path(corpus_dir) / "corpus.json").exists():
        logger.info("  Using Phase 2 corpus - gold doc coverage already 100%")
        filtered_qa = all_qa_pairs
        filter_stats = {"total": len(all_qa_pairs), "filtered": len(all_qa_pairs), "ratio": 1.0}
    else:
        # Legacy: 필터링 필요
        filtered_qa, filter_stats = filter_qa_for_phase1_corpus(
            qa_pairs=all_qa_pairs,
            phase1_doc_ids=phase1_doc_ids,
            strict=False,
        )

    # 샘플 수 제한
    max_samples = config.data.get("max_samples", len(filtered_qa))
    qa_pairs = filtered_qa[:max_samples]
    logger.info(f"Using {len(qa_pairs)} QA pairs for evaluation")

    if dry_run:
        logger.info("\n[DRY RUN] Skipping actual generation...")
        return

    # 4. Stage 1: Evidence 생성 (JSONL crash-safe caching)
    logger.info("\n[Step 4] Stage 1: Generating Evidence...")

    # 캐시 디렉토리
    cache_dir = Path(config.phase1.checkpoint_dir).parent / "phase2_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cache directory: {cache_dir}")

    evidences = {}

    # zRAG Evidence (JSONL per-sample caching)
    evidences["zRAG"] = generate_zrag_evidence_jsonl(
        qa_pairs, config, cache_dir, device
    )

    # Baselines (JSONL per-sample caching)
    for baseline in config.baselines:
        name = baseline.name
        if baseline.type == "bm25":
            baseline_type = "bm25"
        elif baseline.type == "dense":
            baseline_type = "contriever" if "contriever" in baseline.model.lower() else "dense_e5"
        else:
            logger.warning(f"Unknown baseline type: {baseline.type}, skipping")
            continue

        evidences[name] = generate_baseline_evidence_jsonl(
            qa_pairs, corpus, baseline_type, config, cache_dir, device
        )

    # 5. Cost matching: Evidence truncate
    logger.info("\n[Step 5] Applying cost matching (truncate evidence)...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.llm_name, trust_remote_code=True)

    max_tokens = config.evidence_budget.max_tokens
    for method in evidences:
        evidences[method] = [
            truncate_evidence(e, max_tokens, tokenizer) for e in evidences[method]
        ]

    # 6. Stage 1 평가
    logger.info("\n[Step 6] Evaluating Stage 1 (Evidence quality)...")
    stage1_results = evaluate_stage1(evidences, qa_pairs, config)

    # 7. Stage 2: Answer 생성 (JSONL crash-safe caching)
    logger.info("\n[Step 7] Stage 2: Generating Answers (Fixed Reader)...")

    # Reader 초기화 (JSONL 캐시에서 미완료 샘플 있을 때만 실제 사용)
    reader = create_reader_from_config(config.reader)

    answers = {}
    for method in evidences.keys():
        answers[method] = generate_answers_jsonl(
            qa_pairs, evidences[method], method, cache_dir, reader
        )

    # 8. Stage 2 평가
    logger.info("\n[Step 8] Evaluating Stage 2 (Answer accuracy)...")
    stage2_results = evaluate_stage2(answers, qa_pairs)

    # 9. Efficiency 측정
    logger.info("\n[Step 9] Measuring efficiency...")
    efficiency_results = {}
    for method in evidences:
        efficiency_results[method] = measure_efficiency(
            method, qa_pairs, evidences[method], answers[method], tokenizer
        )

    # 10. 결과 출력 및 저장
    print_results_table(stage1_results, stage2_results, efficiency_results)

    # 샘플 저장용
    num_samples = config.logging.get("num_samples_to_save", 10)
    evidence_samples = {
        method: [
            {"question": qa["question"], "evidence": e}
            for qa, e in zip(qa_pairs[:num_samples], evs[:num_samples])
        ]
        for method, evs in evidences.items()
    }
    answer_samples = {
        method: [
            {"question": qa["question"], "gold": qa.get("answer", ""), "predicted": a}
            for qa, a in zip(qa_pairs[:num_samples], ans[:num_samples])
        ]
        for method, ans in answers.items()
    }

    save_results(
        stage1_results, stage2_results, efficiency_results,
        config, evidence_samples, answer_samples
    )

    logger.info("\n" + "=" * 60)
    logger.info("Phase 2 Track A Experiment Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 Track A Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase2_track_a.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run (skip generation)",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Delete cache and regenerate all evidence/answers",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing cache (continue from where it left off)",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save checkpoint every N samples (default: 10)",
    )
    args = parser.parse_args()

    # --resume와 --no_cache는 상호 배타적
    if args.resume and args.no_cache:
        parser.error("--resume and --no_cache are mutually exclusive")

    main(args.config, args.dry_run, args.no_cache, args.save_every)
