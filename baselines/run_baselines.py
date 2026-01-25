"""
전체 Baseline 실행 스크립트
- No Retrieval
- BM25 + LLM
- Standard RAG (Dense)
- (Optional) FlashRAG baselines: Self-RAG, IRCoT, Adaptive-RAG
- (Optional) CoRAG
"""

import sys
import json
import logging
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).parent.parent))

from data.download import download_dataset
from data.preprocess import preprocess_nq, preprocess_hotpotqa
from baselines.no_retrieval import NoRetrievalBaseline
from baselines.standard_rag import StandardRAGBaseline
from evaluation.evaluate_efficiency import measure_latency

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_all_baselines(
    config_path: str = None,
    config: dict = None,
    max_samples: int = None,
) -> dict:
    """모든 Baseline 실행"""

    if config is None:
        with open(config_path, "r") as f:
            config = OmegaConf.create(yaml.safe_load(f))

    device = config.hardware.device if torch.cuda.is_available() else "cpu"

    # Data
    logger.info("Loading dataset...")
    dataset = download_dataset(config.data.dataset)

    if config.data.dataset == "natural_questions":
        processed = preprocess_nq(
            dataset,
            corpus_size=config.data.corpus_size,
            max_samples=config.data.get("eval_samples", max_samples),
        )
    elif config.data.dataset == "hotpot_qa":
        processed = preprocess_hotpotqa(
            dataset,
            corpus_size=config.data.corpus_size,
            max_samples=config.data.get("eval_samples", max_samples),
        )

    qa_pairs = processed["qa_pairs"]
    corpus = processed["corpus"]
    eval_samples = max_samples or config.data.get("eval_samples", len(qa_pairs))

    logger.info(f"Dataset: {len(qa_pairs)} QA pairs, {len(corpus)} documents")

    all_results = {}

    # ==========================================
    # 1. No Retrieval
    # ==========================================
    logger.info("\n" + "=" * 40)
    logger.info("Baseline 1: No Retrieval")
    logger.info("=" * 40)

    no_ret = NoRetrievalBaseline(
        model_name=config.model.llm_name,
        quantization=config.model.quantization,
        device=device,
    )
    all_results["no_retrieval"] = no_ret.evaluate(
        qa_pairs, max_samples=eval_samples
    )
    del no_ret
    torch.cuda.empty_cache()

    # ==========================================
    # 2. BM25 + LLM
    # ==========================================
    logger.info("\n" + "=" * 40)
    logger.info("Baseline 2: BM25 + LLM")
    logger.info("=" * 40)

    bm25_rag = StandardRAGBaseline(
        llm_name=config.model.llm_name,
        retriever_type="bm25",
        quantization=config.model.quantization,
        device=device,
    )
    bm25_rag.build_index(corpus)
    all_results["bm25"] = bm25_rag.evaluate(
        qa_pairs,
        k=config.read_phase.top_k,
        max_samples=eval_samples,
    )
    del bm25_rag
    torch.cuda.empty_cache()

    # ==========================================
    # 3. Standard RAG (Dense - E5)
    # ==========================================
    logger.info("\n" + "=" * 40)
    logger.info("Baseline 3: Standard RAG (E5-base-v2)")
    logger.info("=" * 40)

    dense_rag = StandardRAGBaseline(
        llm_name=config.model.llm_name,
        retriever_type="dense",
        retriever_name="intfloat/e5-base-v2",
        quantization=config.model.quantization,
        device=device,
    )
    dense_rag.build_index(corpus)
    all_results["standard_rag"] = dense_rag.evaluate(
        qa_pairs,
        k=config.read_phase.top_k,
        max_samples=eval_samples,
    )
    del dense_rag
    torch.cuda.empty_cache()

    # ==========================================
    # 4. FlashRAG baselines (optional)
    # ==========================================
    if hasattr(config, "baselines"):
        for baseline_cfg in config.baselines:
            if baseline_cfg.type == "flashrag":
                logger.info(f"\nFlashRAG Baseline: {baseline_cfg.name}")
                try:
                    result = run_flashrag_baseline(
                        method=baseline_cfg.method,
                        config=config,
                        qa_pairs=qa_pairs,
                        corpus=corpus,
                        max_samples=eval_samples,
                    )
                    all_results[baseline_cfg.name] = result
                except Exception as e:
                    logger.warning(f"FlashRAG {baseline_cfg.name} failed: {e}")
                    all_results[baseline_cfg.name] = {"error": str(e)}

    # ==========================================
    # Summary
    # ==========================================
    logger.info("\n" + "=" * 60)
    logger.info("BASELINE RESULTS SUMMARY")
    logger.info("=" * 60)

    for method, result in all_results.items():
        if "error" in result:
            logger.info(f"  {method}: ERROR - {result['error']}")
        else:
            logger.info(f"  {method}: EM={result.get('em', 'N/A'):.4f}, "
                        f"F1={result.get('f1', 'N/A'):.4f}, "
                        f"Recall@K={result.get('recall_at_k', 'N/A')}")

    # Save
    save_dir = config.logging.save_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = f"{save_dir}/baseline_results.json"

    # Convert to serializable
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {sk: sv for sk, sv in v.items()
                           if isinstance(sv, (int, float, str, bool, type(None)))}

    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Results saved to {save_path}")

    return all_results


def run_flashrag_baseline(
    method: str,
    config,
    qa_pairs: list,
    corpus: dict,
    max_samples: int = None,
) -> dict:
    """
    FlashRAG 기반 baseline 실행

    Note: FlashRAG 설치 및 설정이 필요합니다.
    pip install flashrag
    """
    try:
        from flashrag.config import Config as FlashRAGConfig
        from flashrag.pipeline import SequentialPipeline
    except ImportError:
        raise ImportError(
            "FlashRAG not installed. Please install: pip install flashrag\n"
            "See: https://github.com/RUC-NLPIR/FlashRAG"
        )

    logger.info(f"Running FlashRAG baseline: {method}")

    # FlashRAG config setup
    flashrag_config = {
        "model_name": config.model.llm_name,
        "retriever": "e5-base-v2",
        "method": method,
        "top_k": config.read_phase.top_k,
    }

    # This is a placeholder - actual FlashRAG integration requires
    # specific setup per method. See FlashRAG documentation.
    logger.warning(f"FlashRAG {method}: Using placeholder implementation. "
                   "Please configure FlashRAG properly for actual benchmarking.")

    return {
        "em": 0.0,
        "f1": 0.0,
        "method": f"flashrag_{method}",
        "note": "placeholder - configure FlashRAG for actual results",
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase3_full.yaml")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    run_all_baselines(config_path=args.config, max_samples=args.max_samples)
