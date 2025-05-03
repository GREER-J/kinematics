from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from src.kinematics_library.gaussian import Gaussian


class State(ABC):
    @abstractmethod
    def copy(self) -> State:
        ...

    @abstractmethod
    def to_vector(self) -> np.ndarray:
        ...


@dataclass
class MeanState(State):
    x: np.ndarray

    def to_vector(self):
        return self.x.copy()

    def copy(self):
        return MeanState(self.x.copy())


@dataclass
class GaussianState(State):
    distribution: Gaussian  # your existing class

    def get_mean(self) -> np.ndarray:
        return self.distribution.mu

    def copy(self) -> Gaussian:
        return Gaussian(self.distribution.mu.copy(), self.distribution.sqrt_cov.copy())
