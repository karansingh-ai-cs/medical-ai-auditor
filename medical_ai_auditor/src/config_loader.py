"""
config_loader.py
----------------
Load config.yaml and expose a simple dict-like interface.
Usage:
    from src.config_loader import cfg
    lr = cfg["training"]["learning_rate"]
"""

import os
import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def load_config(path: str = _CONFIG_PATH) -> dict:
    with open(os.path.abspath(path), "r") as f:
        return yaml.safe_load(f)


# Singleton — import `cfg` directly in any module
cfg = load_config()
