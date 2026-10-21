import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def denorm_to_01(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    x01 = x * float(std) + float(mean)
    return torch.clamp(x01, 0.0, 1.0)


def renorm_from_01(x01: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (x01 - float(mean)) / float(std)


def add_speckle_noise_in_01(x_norm: torch.Tensor, mean: float, std: float,
                             speckle_std: float, clip_min: float, clip_max: float) -> torch.Tensor:
    """Multiplicative speckle noise applied in de-normalized [0,1] pixel space."""
    x01 = denorm_to_01(x_norm, mean, std)
    noise = torch.randn_like(x01) * float(speckle_std) + 1.0
    noise = torch.clamp(noise, min=float(clip_min), max=float(clip_max))
    x01_s = x01 * noise
    return renorm_from_01(x01_s, mean, std)
