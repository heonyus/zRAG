"""
Phase 1.5 Modules: Evidence Generation Training

This module implements LoRA-based evidence generation training
on frozen Phase 1 z_pool and projection.

Modules:
    - model_wrapper: Phase 1.5 forward wrapper
    - trainer: LoRA training loop
    - evaluator: Evidence evaluation + eyeball reports
    - regression: Phase 1 regression tests (A1/A3)

Note: Lazy imports to avoid circular dependencies and heavy module loading at import time.
"""

__all__ = [
    "Phase15ForwardWrapper",
    "Phase15Trainer",
    "run_phase15_training",
    "Phase15Evaluator",
    "run_phase15_evaluation",
    "run_phase1_regression",
]


def __getattr__(name):
    """Lazy import for module components."""
    if name == "Phase15ForwardWrapper":
        from .model_wrapper import Phase15ForwardWrapper
        return Phase15ForwardWrapper
    elif name == "Phase15Trainer":
        from .trainer import Phase15Trainer
        return Phase15Trainer
    elif name == "run_phase15_training":
        from .trainer import run_phase15_training
        return run_phase15_training
    elif name == "Phase15Evaluator":
        from .evaluator import Phase15Evaluator
        return Phase15Evaluator
    elif name == "run_phase15_evaluation":
        from .evaluator import run_phase15_evaluation
        return run_phase15_evaluation
    elif name == "run_phase1_regression":
        from .regression import run_phase1_regression
        return run_phase1_regression
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
