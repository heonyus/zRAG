# Configs module
from pathlib import Path

CONFIG_DIR = Path(__file__).parent

def get_config_path(name: str) -> Path:
    """Get path to a config file by name."""
    return CONFIG_DIR / name
