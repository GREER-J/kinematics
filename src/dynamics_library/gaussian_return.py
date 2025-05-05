from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class GaussianReturn:
    magnitude: np.ndarray
    gaussian_magnitude: Optional["Gaussian"] = None
    grad_magnitude: Optional[np.ndarray] = None
    hess_magnitude: Optional[np.ndarray] = None

    @property
    def has_gaussian(self) -> bool:
        return self.gaussian_magnitude is not None

    @property
    def has_grad(self) -> bool:
        return self.grad_magnitude is not None

    @property
    def has_hesh(self) -> bool:
        return self.hess_magnitude is not None

    def __iter__(self):
        """
        Allow unpacking like (magnitude, grad, hess) = result
        but skip None values cleanly depending on what is available.
        """
        yield self.magnitude
        if self.grad_magnitude is not None:
            yield self.grad_magnitude
        if self.hess_magnitude is not None:
            yield self.hess_magnitude
