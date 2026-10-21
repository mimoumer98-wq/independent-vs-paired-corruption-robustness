"""Interchangeable corruption functions. Common interface:
corrupt(x_norm, severity, mean, std) -> x_norm, operating in de-normalized
[0,1] pixel space then re-normalizing. severity<=0 is identity.

- speckle:  multiplicative Gaussian, clip [0.5, 1.5].
- gaussian: additive Gaussian noise, x01 + N(0, severity^2), clipped to [0,1].
- blur:     Gaussian blur, kernel sigma = severity (px); kernel size ~ 6*sigma+1.
- gamma:    multiplicative Gamma speckle, mean 1, std=severity
            -> Gamma(concentration=1/severity^2, rate=1/severity^2).
"""
from __future__ import annotations

import torch
import torchvision.transforms.functional as TF

from .utils import add_speckle_noise_in_01, denorm_to_01, renorm_from_01


def speckle(x_norm, severity, mean, std):
    if severity <= 0:
        return x_norm
    return add_speckle_noise_in_01(x_norm, mean, std, severity, 0.5, 1.5)


def gaussian(x_norm, severity, mean, std):
    if severity <= 0:
        return x_norm
    x01 = denorm_to_01(x_norm, mean, std)
    x01 = torch.clamp(x01 + torch.randn_like(x01) * float(severity), 0.0, 1.0)
    return renorm_from_01(x01, mean, std)


def blur(x_norm, severity, mean, std):
    if severity <= 0:
        return x_norm
    x01 = denorm_to_01(x_norm, mean, std)
    k = max(int(2 * round(3 * float(severity)) + 1), 3)
    x01 = TF.gaussian_blur(x01, kernel_size=[k, k], sigma=[float(severity), float(severity)])
    return renorm_from_01(x01, mean, std)


def gamma(x_norm, severity, mean, std):
    if severity <= 0:
        return x_norm
    x01 = denorm_to_01(x_norm, mean, std)
    shape = 1.0 / (float(severity) ** 2)  # mean=1, var=severity^2
    g = torch.distributions.Gamma(torch.tensor(shape, device=x01.device),
                                   torch.tensor(shape, device=x01.device)).sample(x01.shape)
    x01 = torch.clamp(x01 * g, 0.0, 1.0)
    return renorm_from_01(x01, mean, std)


REGISTRY = {"speckle": speckle, "gaussian": gaussian, "blur": blur, "gamma": gamma}

SEVERITIES = {
    "speckle":  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "gaussian": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
    "blur":     [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
    "gamma":    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
}
