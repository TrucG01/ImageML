"""Convenience exports for the DAVID training backend."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import ExperimentConfig, SplitConfig, load_config
from .pipeline import run_training

__all__ = [
    "ExperimentConfig",
    "SplitConfig",
    "load_config",
    "run_training",
]


def train_from_config(config_path: Optional[Path] = None) -> None:
    """Load configuration from YAML and run the training pipeline."""
    config = load_config(config_path)
    run_training(config)
