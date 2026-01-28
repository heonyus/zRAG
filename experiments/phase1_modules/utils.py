"""
Utility functions for Phase 1 verification and ablation.

Provides:
- Logging setup (3 files: run.log, debug.log, warnings.log)
- JSON/JSONL caching with crash-safe writes
- Reproducibility utilities (seeding)
- Environment info collection
"""

import json
import logging
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Union

import numpy as np
import torch

# Global logger reference
_logger: Optional[logging.Logger] = None


def setup_logging(log_dir: Path, logger_name: str = "phase1_runner") -> logging.Logger:
    """
    Setup logging with 3 output files + console.

    Args:
        log_dir: Directory to store log files
        logger_name: Name for the logger

    Returns:
        Configured logger instance

    Files created:
        - run.log: INFO level (main progress)
        - debug.log: DEBUG level (detailed debugging)
        - warnings.log: WARNING+ level (issues only)
    """
    global _logger

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    logger.handlers.clear()

    # Formatter
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # INFO -> run.log
    info_handler = logging.FileHandler(log_dir / "run.log", mode="a", encoding="utf-8")
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(fmt)
    info_handler.addFilter(lambda record: record.levelno < logging.WARNING)  # INFO only
    logger.addHandler(info_handler)

    # Also add INFO+ to run.log (we want all INFO and above)
    info_plus_handler = logging.FileHandler(log_dir / "run.log", mode="a", encoding="utf-8")
    info_plus_handler.setLevel(logging.INFO)
    info_plus_handler.setFormatter(fmt)
    logger.addHandler(info_plus_handler)

    # Actually, let's simplify: run.log gets INFO+, debug.log gets DEBUG+, warnings.log gets WARNING+
    logger.handlers.clear()

    # run.log: INFO+
    run_handler = logging.FileHandler(log_dir / "run.log", mode="a", encoding="utf-8")
    run_handler.setLevel(logging.INFO)
    run_handler.setFormatter(fmt)
    logger.addHandler(run_handler)

    # debug.log: DEBUG+
    debug_handler = logging.FileHandler(log_dir / "debug.log", mode="a", encoding="utf-8")
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(fmt)
    logger.addHandler(debug_handler)

    # warnings.log: WARNING+
    warn_handler = logging.FileHandler(log_dir / "warnings.log", mode="a", encoding="utf-8")
    warn_handler.setLevel(logging.WARNING)
    warn_handler.setFormatter(fmt)
    logger.addHandler(warn_handler)

    # Console: INFO+
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """Get the configured logger, or create a default one."""
    global _logger
    if _logger is None:
        _logger = logging.getLogger("phase1_runner")
        if not _logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            _logger.addHandler(handler)
            _logger.setLevel(logging.INFO)
    return _logger


# ==============================================================================
# JSON/JSONL I/O with crash-safe writes
# ==============================================================================

def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file with atomic write (temp file + rename).

    Args:
        data: Data to serialize
        path: Output file path
        indent: JSON indentation (default 2)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file first, then rename (atomic on POSIX)
    temp_path = path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())

    shutil.move(str(temp_path), str(path))


def load_json(path: Union[str, Path]) -> Any:
    """Load data from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Union[str, Path], record: dict) -> None:
    """
    Append a single record to JSONL file with flush + fsync for crash safety.

    Args:
        path: JSONL file path
        record: Dictionary to append as one line
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_jsonl(path: Union[str, Path]) -> List[dict]:
    """
    Load all records from JSONL file, skipping corrupted lines.

    Args:
        path: JSONL file path

    Returns:
        List of parsed records
    """
    path = Path(path)
    if not path.exists():
        return []

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                get_logger().warning(f"Skipping corrupted line {line_num} in {path}: {e}")

    return records


def load_jsonl_cache(path: Union[str, Path], key_field: str = "i") -> Dict[int, dict]:
    """
    Load JSONL cache as dictionary keyed by a specific field.

    Args:
        path: JSONL file path
        key_field: Field to use as dictionary key (default "i" for row index)

    Returns:
        Dictionary mapping key -> record
    """
    records = load_jsonl(path)
    cache = {}
    for record in records:
        key = record.get(key_field)
        if key is not None:
            cache[key] = record
    return cache


def atomic_save(data: bytes, path: Union[str, Path]) -> None:
    """
    Save binary data atomically (temp file + rename).

    Args:
        data: Binary data to save
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(".tmp")
    with open(temp_path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    shutil.move(str(temp_path), str(path))


# ==============================================================================
# Reproducibility
# ==============================================================================

def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, enable deterministic CUDA operations (slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # PyTorch 1.8+ deterministic algorithms
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass


# ==============================================================================
# Environment Info
# ==============================================================================

def get_git_info() -> Dict[str, Optional[str]]:
    """Get git commit hash and branch info if available."""
    info = {"commit_hash": None, "branch": None, "is_dirty": None}

    try:
        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            info["commit_hash"] = result.stdout.strip()

        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()

        # Check if dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            info["is_dirty"] = bool(result.stdout.strip())

    except Exception:
        pass

    return info


def get_env_info() -> Dict[str, Any]:
    """
    Collect comprehensive environment information for reproducibility.

    Returns:
        Dictionary with system, Python, PyTorch, CUDA info
    """
    info = {
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "cwd": os.getcwd(),
    }

    # PyTorch info
    info["torch_version"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    # Git info
    info["git"] = get_git_info()

    # Key package versions
    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        info["transformers_version"] = None

    try:
        import numpy
        info["numpy_version"] = numpy.__version__
    except ImportError:
        info["numpy_version"] = None

    return info


# ==============================================================================
# Context Managers
# ==============================================================================

@contextmanager
def temporary_alpha(model, alpha_value: float):
    """
    Context manager to temporarily set model's alpha parameter.

    Args:
        model: WritePhaseModel instance
        alpha_value: Temporary alpha value to set

    Yields:
        None (model alpha is modified in-place)

    Example:
        with temporary_alpha(model, 0.0):
            # model.alpha is 0.0 here
            pass
        # model.alpha is restored
    """
    original_alpha = model.alpha.item()
    try:
        model.alpha.fill_(alpha_value)
        yield
    finally:
        model.alpha.fill_(original_alpha)


@contextmanager
def temporary_projection(model, new_state_dict: Optional[dict] = None, reinit: bool = False):
    """
    Context manager to temporarily modify model's projection layer.

    Args:
        model: WritePhaseModel instance
        new_state_dict: State dict to load (if provided)
        reinit: If True, reinitialize projection with random weights

    Yields:
        None (model projection is modified in-place)
    """
    import copy
    original_state = copy.deepcopy(model.z_to_embedding.state_dict())

    try:
        if reinit:
            # Reinitialize with small random weights
            for name, param in model.z_to_embedding.named_parameters():
                if "weight" in name:
                    torch.nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    torch.nn.init.zeros_(param)
        elif new_state_dict is not None:
            model.z_to_embedding.load_state_dict(new_state_dict)
        yield
    finally:
        model.z_to_embedding.load_state_dict(original_state)


# ==============================================================================
# Timing
# ==============================================================================

class Timer:
    """Simple timer for tracking module runtimes."""

    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed_seconds = None

    def __enter__(self):
        self.start_time = datetime.now()
        get_logger().info(f"[{self.name}] Started at {self.start_time.isoformat()}")
        return self

    def __exit__(self, *args):
        self.end_time = datetime.now()
        self.elapsed_seconds = (self.end_time - self.start_time).total_seconds()
        get_logger().info(
            f"[{self.name}] Completed in {self.elapsed_seconds:.2f}s "
            f"({self.elapsed_seconds/60:.2f}min)"
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "elapsed_seconds": self.elapsed_seconds,
        }


# ==============================================================================
# Metrics Helpers
# ==============================================================================

def compute_percentiles(values: List[float], percentiles: List[int] = [10, 25, 50, 75, 90]) -> Dict[str, float]:
    """
    Compute percentiles for a list of values.

    Args:
        values: List of numeric values
        percentiles: List of percentile values to compute

    Returns:
        Dictionary with p{N} keys
    """
    arr = np.array(values)
    result = {}
    for p in percentiles:
        result[f"p{p}"] = float(np.percentile(arr, p))
    return result


def safe_mean(values: List[float]) -> float:
    """Compute mean, returning 0.0 for empty lists."""
    if not values:
        return 0.0
    return float(np.mean(values))


def safe_std(values: List[float]) -> float:
    """Compute std, returning 0.0 for lists with < 2 elements."""
    if len(values) < 2:
        return 0.0
    return float(np.std(values))
