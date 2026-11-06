"""Configurations for TL."""

__all__ = [
    "StepLRConfiguration",
    "LambdaLRConfiguration",
]
from dataclasses import dataclass


@dataclass(frozen=True)
class StepLRConfiguration:
    lr: float = 1e-4
    min: float = 1e-07
    gamma: float = 0.95
    step: int = 10

    def __post_init__(self):
        if self.lr <= 0:
            raise ValueError(f"StepLR: lr must be > 0, got {self.lr}")
        if self.min <= 0:
            raise ValueError(f"StepLR: min must be > 0, got {self.min}")
        if not (0 < self.gamma <= 1):
            raise ValueError(f"StepLR: gamma must be in (0, 1], got {self.gamma}")
        if self.step < 1:
            raise ValueError(f"StepLR: step must be >= 1, got {self.step}")


@dataclass(frozen=True)
class LambdaLRConfiguration:
    lr: float = 1e-4
    min: float = 1e-10
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-9
    factor: float = 1.0
    warmup: float = 4000

    def __post_init__(self):
        if self.lr <= 0:
            raise ValueError(f"LambdaLR: lr must be > 0, got {self.lr}")
        if self.min <= 0:
            raise ValueError(f"LambdaLR: min must be > 0, got {self.min}")
        if not (0 < self.beta1 < 1):
            raise ValueError(f"LambdaLR: beta1 must be in (0, 1), got {self.beta1}")
        if not (0 < self.beta2 < 1):
            raise ValueError(f"LambdaLR: beta2 must be in (0, 1), got {self.beta2}")
        if self.eps <= 0:
            raise ValueError(f"LambdaLR: eps must be > 0, got {self.eps}")
        if self.factor <= 0:
            raise ValueError(f"LambdaLR: factor must be > 0, got {self.factor}")
        if self.warmup <= 0:
            raise ValueError(f"LambdaLR: warmup must be > 0, got {self.warmup}")
