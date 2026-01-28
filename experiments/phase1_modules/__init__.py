"""
Phase 1 Verification & Ablation Modules

This package provides EVAL-ONLY modules for verifying Phase 1 (Write Phase) training results.
NO TRAINING is performed - all modules are inference-only.

Modules:
- A1: Full NLL Confusion Matrix (200x200)
- A2: Z-only Autoregressive Generation
- A3: Z-shuffle Sanity Check
- B1: Alpha Ablation
- B2: Projection Ablation
"""

from .utils import (
    setup_logging,
    get_logger,
    save_json,
    load_json,
    append_jsonl,
    load_jsonl,
    load_jsonl_cache,
    atomic_save,
    set_seed,
    get_env_info,
)
from .confusion_matrix import run_confusion_matrix
from .zonly_generation import run_zonly_generation
from .zshuffle import run_zshuffle_sanity
from .alpha_ablation import run_alpha_ablation
from .projection_ablation import run_projection_ablation
from .visualization import (
    plot_confusion_heatmap,
    plot_margin_histogram,
    plot_delta_barplot,
    plot_score_distribution,
)
from .reports import (
    generate_worst_cases_report,
    generate_eyeball_report,
    generate_dashboard,
    generate_readme,
)

__all__ = [
    # Utils
    "setup_logging",
    "get_logger",
    "save_json",
    "load_json",
    "append_jsonl",
    "load_jsonl",
    "load_jsonl_cache",
    "atomic_save",
    "set_seed",
    "get_env_info",
    # Modules
    "run_confusion_matrix",
    "run_zonly_generation",
    "run_zshuffle_sanity",
    "run_alpha_ablation",
    "run_projection_ablation",
    # Visualization
    "plot_confusion_heatmap",
    "plot_margin_histogram",
    "plot_delta_barplot",
    "plot_score_distribution",
    # Reports
    "generate_worst_cases_report",
    "generate_eyeball_report",
    "generate_dashboard",
    "generate_readme",
]
