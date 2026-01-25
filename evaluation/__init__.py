# LLM-as-Memory Evaluation
# 새 설계: Evidence 품질 평가 중심

# Core metrics
from .metrics import compute_em, compute_f1, compute_recall_at_k

# Evidence 품질 평가 (신규)
from .evidence_metrics import (
    compute_rouge_l,
    compute_answer_coverage,
    compute_token_f1,
    compute_faithfulness_simple,
    evaluate_evidence_quality,
    evaluate_batch,
    compare_with_rag,
    print_comparison_table,
)

# QA 평가 (레거시 호환 + Evidence 평가)
from .evaluate_qa import (
    evaluate_qa,
    evaluate_evidence,
    evaluate_evidence_vs_rag,
    generate_results_table,
)

# Efficiency 평가
from .evaluate_efficiency import evaluate_efficiency

# Legacy (Write Phase 평가 - 더 이상 사용 안함)
# from .evaluate_write import evaluate_reconstruction
