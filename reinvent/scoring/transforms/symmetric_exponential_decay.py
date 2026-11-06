"""Symmetric exponential decay: exp(-k * |x - target|)

Rewards predictions close to a fixed target value, decaying exponentially
with the absolute distance from that target.  Unlike :class:`ExponentialDecay`,
which is one-sided (clamped to 1.0 for x < 0), this transform is symmetric
around ``target`` and always peaks at exactly 1.0 when x equals ``target``.
"""

__all__ = ["SymmetricExponentialDecay"]

from dataclasses import dataclass

import numpy as np

from .transform import Transform


@dataclass
class Parameters:
    type: str
    target: float
    k: float = 1.0


class SymmetricExponentialDecay(Transform, param_cls=Parameters):
    def __init__(self, params: Parameters):
        super().__init__(params)

        self.target = params.target
        self.k = params.k

        if self.k <= 0:
            raise ValueError(f"SymmetricExponentialDecay Transform: k must be > 0, got {self.k}")

    def __call__(self, values) -> np.ndarray:
        values = np.array(values, dtype=np.float32)
        delta = np.abs(values - self.target)
        return np.exp(-self.k * delta)
