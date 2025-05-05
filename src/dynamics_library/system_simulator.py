from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple
import copy
import numpy as np
from src.kinematics_library.system import BaseSystem
from src.kinematics_library.state import State


class SystemSimulator(BaseSystem, ABC):
    def __init__(self, x0: State, time: float = 0.0):
        super().__init__(time=time, state=x0)

    @abstractmethod
    def input(self, t: float) -> np.ndarray:
        """
        Must be implemented by subclasses to return control input at time t
        """

    def dim(self) -> int:
        return len(self.x_sim)

    def dynamicsSim(self, t: float, x: np.ndarray, u: np.ndarray):
        """
        Default implementation just calls `dynamics`. Subclasses may override.
        """
        return self.dynamics(t, x, u)

    @abstractmethod
    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray):
        """
        Default dynamics call forwards to dynamicsSim unless overridden.
        """

    def predict(self, time_next: float) -> tuple:
        """
        Simple Euler integration of x_sim forward to time_next.
        """
        dt = time_next - self.time
        u = self.input(self.time)
        dx, *_ = self.dynamicsSim(self.time, self.x_sim, u)
        self.x_sim = self.x_sim + dt * dx
        self.time = time_next
        return self

    def copy(self) -> "SystemSimulator":
        return copy.deepcopy(self)
