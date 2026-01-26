# LLM-as-Memory Models
#
# Phase 1 (Write): z_i → D_i (문서 생성, LLM freeze)
# Phase 3 (Read): [Z_all] + Query → Evidence (내부 routing)

# Phase 1: Write Phase
from .write_phase_model import WritePhaseModel, ZPoolManager

# Phase 3: Read Phase (Z_all concat)
from .parametric_memory_llm import ParametricMemoryLLM
from .evidence_trainer import EvidenceTrainer
from .adaptation import ZAdaptation

# Legacy imports (models/legacy/ 폴더로 이동됨)
# from .legacy.parametric_qa import ParametricQA
# from .legacy.write_phase import WritePhaseTrainer
# from .legacy.read_phase import ReadPhaseTrainer
# from .legacy.router import CosineSelector, LearnedRouter, AttentionSelector
