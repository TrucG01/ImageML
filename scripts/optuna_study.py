import argparse
from pathlib import Path
import optuna
import time

from model_backend.config import load_config, ExperimentConfig
from model_backend.pipeline import run_training


def suggest_config(trial: optuna.Trial, cfg: ExperimentConfig) -> ExperimentConfig:
    # Copy base config (dataclass is mutable; we adjust fields directly)
    # Learning rate
    cfg.learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    # Batch size (respect VRAM/DRAM adaptive logic; choose within safe bounds)
    cfg.batch_size = trial.suggest_int("batch_size", 4, 24, step=4)
    # Image size (square)
    side = trial.suggest_categorical("image_side", [256, 384, 512])
    cfg.image_size = (side, side)
    # Optional: weight decay
    cfg.weight_decay = trial.suggest_float("weight_decay", 0.0, 1e-3, log=False)
    # Keep Windows-stable settings
    cfg.num_workers = 0
    # Short run
    cfg.epochs = 1
    # Diagnostics lighter
    cfg.batch_mem_log_every = 10
    cfg.dataset_mem_log_every = 200
    cfg.detailed_batch_logging = False
    # Ensure fresh training (no resume)
    cfg.resume_path = None
    return cfg


def objective(trial: optuna.Trial, base_config_path: Path) -> float:
    cfg = load_config(base_config_path)
    cfg = suggest_config(trial, cfg)
    # Run training for 1 epoch and read last validation mIoU if available; otherwise use training loss
    start = time.time()
    run_training(cfg)
    # Heuristic: read outputs/model/history.json and prefer val_miou
    history_path = cfg.output_dir / "history.json"
    val_miou = None
    train_loss = None
    try:
        import json
        with history_path.open("r", encoding="utf-8") as f:
            hist = json.load(f)
        if hist.get("val_miou"):
            if len(hist["val_miou"]) > 0:
                val_miou = float(hist["val_miou"][-1])
        if hist.get("train_loss"):
            if len(hist["train_loss"]) > 0:
                train_loss = float(hist["train_loss"][-1])
    except Exception:
        pass
    # Optimization direction: maximize mIoU if available, else minimize train loss
    if val_miou is not None:
        return val_miou
    # Fall back to negative loss (maximize)
    if train_loss is not None:
        return -train_loss
    # If nothing, return a small value based on runtime to avoid breaking the study
    return - (time.time() - start)


def main():
    parser = argparse.ArgumentParser(description="Run Optuna study for ImageML")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--study", type=str, default="imageml-study")
    args = parser.parse_args()

    from tqdm import tqdm
    study = optuna.create_study(direction="maximize", study_name=args.study)
    print(f"Running {args.trials} Optuna trials with progress bar...")
    with tqdm(total=args.trials, desc="Optuna Trials") as pbar:
        def callback(study, trial):
            pbar.update(1)
        study.optimize(lambda t: objective(t, args.config), n_trials=args.trials, callbacks=[callback])

    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
