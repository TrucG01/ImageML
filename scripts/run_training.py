import argparse
from pathlib import Path
from model_backend.config import load_config
from model_backend.pipeline import run_training


def main():
    parser = argparse.ArgumentParser(description="Run ImageML training")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_training(cfg)


if __name__ == "__main__":
    main()
