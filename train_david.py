#!/usr/bin/env python
"""Entry point for training DAVID semantic segmentation from YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path

from david_backend import train_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeepLabV3+ on the DAVID dataset")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_from_config(args.config)


if __name__ == "__main__":
    main()
