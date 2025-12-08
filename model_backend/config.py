"""Configuration loading for the DAVID semantic segmentation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, List
import os
import platform

try:
    import yaml
except ImportError as exc:  # pragma: no cover - defensive import guard
    raise ImportError(
        "PyYAML is required to load configuration files. Install with 'pip install pyyaml'."
    ) from exc


CONFIG_DEFAULT_LOCATIONS = (Path("config.yaml"),)

DEFAULTS = {
    "batch_size": 4,
    "val_batch_size": 4,
    "epochs": 80,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "num_workers": 0,  # Lower default workers to minimize DRAM on Windows
    "image_size": (512, 512),
    "log_interval": 25,
    "val_interval": 1,
    "amp": True,
    "seed": 1337,
    "force_resplit": False,
    "visualization_interval": 5,
    "max_checkpoints": 5,
    "backend": "auto",
    "output_dir": "outputs",
    # DataLoader knobs
    "pin_memory": True,
    "persistent_workers": False,
    "prefetch_factor": 1,
    "timeout": 0,
    # Memory-safety knobs
    "max_grad_norm": 0.0,
    "cudnn_benchmark": True,
    "empty_cache_each_epoch": False,
    # Diagnostics
    "diagnostics.enable_tracemalloc": True,
    "diagnostics.gc_every_steps": 10,
    "diagnostics.batch_mem_log_every": 1,
    "diagnostics.dataset_mem_log_every": 200,
    "diagnostics.detailed_batch_logging": False,
}


@dataclass
class SplitConfig:
    train_count: int = 20
    val_count: int = 4
    test_count: int = 4
    seed: int = 42


@dataclass
class ExperimentConfig:
    data_root: Path
    output_dir: Path
    resume_path: Optional[Path]
    batch_size: int
    val_batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    image_size: Tuple[int, int]
    log_interval: int
    val_interval: int
    amp: bool
    seed: int
    force_resplit: bool
    visualization_interval: int
    max_checkpoints: int
    backend: str
    split: SplitConfig
    # dataloader
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    timeout: int
    # memory safety
    max_grad_norm: float
    cudnn_benchmark: bool
    empty_cache_each_epoch: bool
    # optional: restrict to specific sequences
    include_sequences: Optional[List[str]]
    # diagnostics
    enable_tracemalloc: bool
    gc_every_steps: int
    batch_mem_log_every: int
    dataset_mem_log_every: int
    detailed_batch_logging: bool
    # vram target
    vram_target_fraction: float
    # dram target (soft cap in MB)
    dram_target_mb: Optional[int]


def coalesce(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


def extract_nested(config: Dict[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _normalize_image_size(value: Any) -> Tuple[int, int]:
    if isinstance(value, (list, tuple)):
        if len(value) == 2:
            return int(value[0]), int(value[1])
        if len(value) == 1:
            side = int(value[0])
            return side, side
    if isinstance(value, (int, float)):
        side = int(value)
        return side, side
    if isinstance(value, str):
        cleaned = value.replace("x", " ").replace(",", " ")
        parts = [part for part in cleaned.split() if part]
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
        if len(parts) == 1:
            side = int(parts[0])
            return side, side
    raise ValueError("image_size must resolve to two integers")


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "on"}:
            return True
        if lowered in {"false", "no", "0", "off"}:
            return False
    raise ValueError(f"Cannot convert {value!r} into a boolean")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a top-level mapping")
    return data


def _pick_numeric(
    arg_value: Optional[Any],
    config: Dict[str, Any],
    keys: Sequence[str],
    default_key: str,
    cast_fn,
) -> Any:
    """
    Pick a numeric value from arguments, config, or defaults, applying a cast function.
    """
    if arg_value is not None:
        return cast_fn(arg_value)
    for key in keys:
        candidate = extract_nested(config, key)
        if candidate is not None:
            return cast_fn(candidate)
    return cast_fn(DEFAULTS[default_key])


def _resolve_num_workers(value: Any) -> int:
    """
    Resolve num_workers, allowing special string 'auto'.
    'auto' -> max(1, os.cpu_count() - 1), capped at 8 on Windows.
    """
    if isinstance(value, str):
        if value.strip().lower() == "auto":
            cores = os.cpu_count() or 1
            resolved = max(1, cores - 1)
            if platform.system().lower().startswith("windows"):
                resolved = min(resolved, 8)
            return resolved
        try:
            return int(value)
        except Exception:
            pass
    return int(value)


def load_config(config_path: Optional[Path]) -> ExperimentConfig:
    """
    Load experiment configuration from a YAML file.

    Args:
        config_path: Optional path to YAML config file. If None, uses default locations.

    Returns:
        ExperimentConfig: Parsed configuration dataclass.
    """
    path = config_path
    if path is None:
        for candidate in CONFIG_DEFAULT_LOCATIONS:
            if candidate.exists():
                path = candidate
                break
    if path is None:
        raise FileNotFoundError(
            "No configuration file supplied and default 'config.yaml' was not found."
        )
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    config_dir = path.parent
    yaml_cfg = _load_yaml(path)

    data_root_value = coalesce(
        extract_nested(yaml_cfg, "dataset.root"),
        extract_nested(yaml_cfg, "data.root"),
        extract_nested(yaml_cfg, "DATA_ROOT"),
        extract_nested(yaml_cfg, "INPUT_DIR"),
    )
    if data_root_value is None:
        raise ValueError("Configuration missing 'dataset.root' (or equivalent) entry")
    data_root = Path(str(data_root_value)).expanduser()
    if not data_root.is_absolute():
        data_root = (config_dir / data_root).resolve()
    data_root = data_root.resolve()

    output_dir_value = coalesce(
        extract_nested(yaml_cfg, "training.output_dir"),
        extract_nested(yaml_cfg, "output.dir"),
        extract_nested(yaml_cfg, "OUTPUT_DIR"),
        DEFAULTS["output_dir"],
    )
    if output_dir_value is None:
        output_dir_value = "outputs"
    output_dir = Path(str(output_dir_value)).expanduser()
    if not output_dir.is_absolute():
        output_dir = (config_dir / output_dir).resolve()
    output_dir = output_dir.resolve()

    resume_value = coalesce(
        extract_nested(yaml_cfg, "training.resume"),
        extract_nested(yaml_cfg, "resume"),
    )
    if resume_value is not None:
        resume_path: Optional[Path] = Path(str(resume_value)).expanduser()
        if not resume_path.is_absolute():
            resume_path = (config_dir / resume_path).resolve()
        resume_path = resume_path.resolve()
    else:
        resume_path = None

    batch_size = _pick_numeric(None, yaml_cfg, ["training.batch_size"], "batch_size", int)
    val_batch_size = _pick_numeric(
        None,
        yaml_cfg,
        ["training.val_batch_size", "training.batch_size"],
        "val_batch_size",
        int,
    )
    epochs = _pick_numeric(None, yaml_cfg, ["training.epochs"], "epochs", int)
    learning_rate = _pick_numeric(
        None,
        yaml_cfg,
        ["training.learning_rate", "training.lr"],
        "learning_rate",
        float,
    )
    weight_decay = _pick_numeric(None, yaml_cfg, ["training.weight_decay"], "weight_decay", float)
    num_workers_raw = coalesce(
        extract_nested(yaml_cfg, "training.num_workers"),
        extract_nested(yaml_cfg, "data.num_workers"),
        DEFAULTS["num_workers"],
    )
    num_workers = _resolve_num_workers(num_workers_raw)

    image_size_value = coalesce(
        extract_nested(yaml_cfg, "training.image_size"),
        extract_nested(yaml_cfg, "data.image_size"),
        DEFAULTS["image_size"],
    )
    image_size = _normalize_image_size(image_size_value)

    log_interval = _pick_numeric(None, yaml_cfg, ["training.log_interval"], "log_interval", int)
    val_interval = _pick_numeric(None, yaml_cfg, ["training.val_interval"], "val_interval", int)
    visualization_interval = _pick_numeric(
        None,
        yaml_cfg,
        ["training.visualization_interval"],
        "visualization_interval",
        int,
    )
    max_checkpoints = _pick_numeric(
        None,
        yaml_cfg,
        ["training.max_checkpoints"],
        "max_checkpoints",
        int,
    )

    backend_value = coalesce(
        extract_nested(yaml_cfg, "backend.target"),
        DEFAULTS["backend"],
    )
    backend = str(backend_value).lower()
    if backend not in {"auto", "cuda", "rocm", "cpu"}:
        raise ValueError(f"Unsupported backend '{backend}'. Choose from auto/cuda/rocm/cpu.")

    amp_value = coalesce(
        extract_nested(yaml_cfg, "backend.amp"),
        extract_nested(yaml_cfg, "training.amp"),
        DEFAULTS["amp"],
    )
    amp = _to_bool(amp_value)
    # DataLoader configuration
    pin_memory = _to_bool(
        coalesce(extract_nested(yaml_cfg, "dataloader.pin_memory"), DEFAULTS["pin_memory"])
    )
    persistent_workers = _to_bool(
        coalesce(
            extract_nested(yaml_cfg, "dataloader.persistent_workers"),
            DEFAULTS["persistent_workers"],
        )
    )
    prefetch_factor = int(
        coalesce(extract_nested(yaml_cfg, "dataloader.prefetch_factor"), DEFAULTS["prefetch_factor"])
    )
    timeout = int(coalesce(extract_nested(yaml_cfg, "dataloader.timeout"), DEFAULTS["timeout"]))

    # Memory-safety
    max_grad_norm = float(
        coalesce(extract_nested(yaml_cfg, "training.max_grad_norm"), DEFAULTS["max_grad_norm"])
    )
    cudnn_benchmark = _to_bool(
        coalesce(extract_nested(yaml_cfg, "training.cudnn_benchmark"), DEFAULTS["cudnn_benchmark"])
    )
    empty_cache_each_epoch = _to_bool(
        coalesce(
            extract_nested(yaml_cfg, "training.empty_cache_each_epoch"),
            DEFAULTS["empty_cache_each_epoch"],
        )
    )

    # Diagnostics configuration
    enable_tracemalloc = _to_bool(
        coalesce(extract_nested(yaml_cfg, "diagnostics.enable_tracemalloc"), DEFAULTS["diagnostics.enable_tracemalloc"])
    )
    gc_every_steps = int(
        coalesce(extract_nested(yaml_cfg, "diagnostics.gc_every_steps"), DEFAULTS["diagnostics.gc_every_steps"])
    )
    batch_mem_log_every = int(
        coalesce(
            extract_nested(yaml_cfg, "diagnostics.batch_mem_log_every"),
            DEFAULTS["diagnostics.batch_mem_log_every"],
        )
    )
    dataset_mem_log_every = int(
        coalesce(
            extract_nested(yaml_cfg, "diagnostics.dataset_mem_log_every"),
            DEFAULTS["diagnostics.dataset_mem_log_every"],
        )
    )
    detailed_batch_logging = _to_bool(
        coalesce(
            extract_nested(yaml_cfg, "diagnostics.detailed_batch_logging"),
            DEFAULTS["diagnostics.detailed_batch_logging"],
        )
    )

    vram_target_fraction = float(
        coalesce(
            extract_nested(yaml_cfg, "training.vram_target_fraction"),
            0.90,
        )
    )

    dram_target_mb_value = coalesce(
        extract_nested(yaml_cfg, "training.dram_target_mb"),
        None,
    )
    dram_target_mb = int(dram_target_mb_value) if dram_target_mb_value is not None else None

    seed_value = coalesce(
        extract_nested(yaml_cfg, "training.seed"),
        DEFAULTS["seed"],
    )
    seed = int(seed_value)

    force_resplit_value = coalesce(
        extract_nested(yaml_cfg, "dataset.force_resplit"),
        extract_nested(yaml_cfg, "training.force_resplit"),
        DEFAULTS["force_resplit"],
    )
    force_resplit = _to_bool(force_resplit_value)

    split_defaults = SplitConfig()
    split_cfg = SplitConfig(
        train_count=int(
            coalesce(
                extract_nested(yaml_cfg, "dataset.split.train"),
                split_defaults.train_count,
            )
        ),
        val_count=int(
            coalesce(
                extract_nested(yaml_cfg, "dataset.split.val"),
                split_defaults.val_count,
            )
        ),
        test_count=int(
            coalesce(
                extract_nested(yaml_cfg, "dataset.split.test"),
                split_defaults.test_count,
            )
        ),
        seed=int(
            coalesce(
                extract_nested(yaml_cfg, "dataset.split.seed"),
                split_defaults.seed,
            )
        ),
    )

    # Optional sequence restriction
    include_sequences_value = coalesce(
        extract_nested(yaml_cfg, "dataset.include"),
        extract_nested(yaml_cfg, "dataset.sequences"),
        extract_nested(yaml_cfg, "dataset.only_sequence"),
    )
    include_sequences: Optional[List[str]] = None
    if include_sequences_value is not None:
        if isinstance(include_sequences_value, (list, tuple)):
            include_sequences = [str(x) for x in include_sequences_value]
        else:
            include_sequences = [str(include_sequences_value)]

    return ExperimentConfig(
        data_root=data_root,
        output_dir=output_dir,
        resume_path=resume_path,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_workers=num_workers,
        image_size=image_size,
        log_interval=log_interval,
        val_interval=val_interval,
        amp=amp,
        seed=seed,
        force_resplit=force_resplit,
        visualization_interval=visualization_interval,
        max_checkpoints=max_checkpoints,
        backend=backend,
        split=split_cfg,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        timeout=timeout,
        max_grad_norm=max_grad_norm,
        cudnn_benchmark=cudnn_benchmark,
        empty_cache_each_epoch=empty_cache_each_epoch,
        # diagnostics
        enable_tracemalloc=enable_tracemalloc,
        gc_every_steps=gc_every_steps,
        batch_mem_log_every=batch_mem_log_every,
        dataset_mem_log_every=dataset_mem_log_every,
        detailed_batch_logging=detailed_batch_logging,
        vram_target_fraction=vram_target_fraction,
        dram_target_mb=dram_target_mb,
        include_sequences=include_sequences,
    )
