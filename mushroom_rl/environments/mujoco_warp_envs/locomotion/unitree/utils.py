"""
util/helper script
"""

import torch


def uniform(bounds, shape, device):
    """Uniform draw between ``bounds``, drawing nothing when they coincide."""
    lower, upper = bounds
    if lower == upper:
        return torch.full(shape, float(lower), device=device)
    return torch.rand(shape, device=device) * (upper - lower) + lower


def bernoulli(probability, n, device):
    if probability <= 0.0 or probability >= 1.0:
        return torch.full((n,), probability >= 1.0, dtype=torch.bool, device=device)
    return torch.rand(n, device=device) < probability
