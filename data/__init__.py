# LLM-as-Memory Data Module
# 새 설계: Evidence 생성 학습용 데이터

# Download and preprocess (requires 'datasets' library)
try:
    from .download import download_dataset
    from .preprocess import preprocess_nq, preprocess_hotpotqa, build_corpus
except ImportError:
    download_dataset = None
    preprocess_nq = None
    preprocess_hotpotqa = None
    build_corpus = None

# Evidence DataLoader (신규)
# Core utilities always available
from .evidence_dataloader import (
    extract_evidence_from_nq,
    extract_evidence_from_hotpotqa,
)

# Heavy classes require torch/transformers
try:
    from .evidence_dataloader import (
        EvidenceDataset,
        create_evidence_dataloader,
        collate_evidence_batch,
        prepare_evidence_pairs_from_nq,
        prepare_evidence_pairs_from_hotpotqa,
    )
except ImportError:
    EvidenceDataset = None
    create_evidence_dataloader = None
    collate_evidence_batch = None
    prepare_evidence_pairs_from_nq = None
    prepare_evidence_pairs_from_hotpotqa = None

# Legacy DataLoader (호환성 유지)
try:
    from .dataloader import WritePhaseDataset, ReadPhaseDataset, get_dataloader
except ImportError:
    WritePhaseDataset = None
    ReadPhaseDataset = None
    get_dataloader = None
