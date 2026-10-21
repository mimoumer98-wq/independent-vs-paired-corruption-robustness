from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Training/evaluation configuration. data_dir/ckpt_dir/results_dir have no
    machine-specific defaults -- pass them explicitly (see each script's --help)."""

    data_dir: Optional[Path] = None
    ckpt_dir: Optional[Path] = None
    results_dir: Optional[Path] = None

    # Data
    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 2

    # Training
    epochs: int = 100
    base_lr: float = 1e-3
    max_lr: float = 2e-3
    weight_decay: float = 1e-3
    use_amp: bool = True
    threshold: float = 0.5
    auto_run_subdir: bool = False
    compute_train_full_metrics: bool = True
    early_stopping_patience: int = 15

    # Normalization (must match src.transforms.build_transforms)
    input_norm_mean: float = 0.5
    input_norm_std: float = 0.5

    # Model
    in_channels: int = 1
    num_classes: int = 1
