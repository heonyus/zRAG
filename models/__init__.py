# LLM-as-Memory Models
# 새 설계: Query → [내부 Routing over Z] → Evidence 생성

from .parametric_memory_llm import ParametricMemoryLLM
from .evidence_trainer import EvidenceTrainer
from .adaptation import ZAdaptation

# Legacy imports (models/legacy/ 폴더로 이동됨)
# from .legacy.parametric_qa import ParametricQA
# from .legacy.write_phase import WritePhaseTrainer
# from .legacy.read_phase import ReadPhaseTrainer
# from .legacy.router import CosineSelector, LearnedRouter, AttentionSelector
