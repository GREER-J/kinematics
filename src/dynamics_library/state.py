from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from src.dynamics_library.gaussian import Gaussian
from typing import Protocol, Optional
from src.dynamics_library.transform import SupportedTransformMode


@dataclass
class FunctionReturn:
    magnitude: np.ndarray
    gradient: Optional[np.ndarray] = None
    hessian: Optional[np.ndarray] = None


class StandardFunctionProtocol(Protocol):
    def __call__(self, x: np.ndarray,
                 return_grad: bool = False,
                 return_hess: bool = False
                 ) -> FunctionReturn:
        """
        Standard function return for either a measurement, or a dynamics function
        """


class State(ABC):
    @abstractmethod
    def copy(self) -> State:
        ...

    @abstractmethod
    def to_vector(self) -> np.ndarray:
        ...

    # @abstractmethod
    # def transform(self, f: StandardFunctionProtocol, method: SupportedTransformMode) -> State:
    #     ...

    # @abstractmethod
    # def affine_transform(self, H, mu_x):
    #     ...

    # @abstractmethod
    # def unscented_transform(self, H, mu_x):
    #     ...

    # @abstractmethod
    # def particle_transform(self, f, samples, N):
    #     ...


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

    def to_vector(self) -> tuple[np.ndarray, np.ndarray]:
        return self.distribution.mu.copy(), self.distribution.cov.copy()

    def from_array(self, mean: np.ndarray, cov: np.ndarray) -> Gaussian:
        return Gaussian.from_moment(mu=mean, P=cov)

    def copy(self) -> Gaussian:
        return Gaussian(self.distribution.mu.copy(), self.distribution.sqrt_cov.copy())
