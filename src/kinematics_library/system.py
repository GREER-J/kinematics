from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple
from src.kinematics_library.state import State


class BaseSystem(ABC):
    def __init__(self, state: State, time: float = 0.0):
        self.time = time
        self.state = state

    @abstractmethod
    def predict(self, time: float) -> BaseSystem:  # TODO named tuple / sructure
        """
        Predict system state forward to the given time.
        """

    @abstractmethod
    def dynamics(self, t: float, x, u) -> Tuple:  # TODO named tupple / structure
        """
        Compute system dynamics f(t, x, u), and optionally return Jacobians Jx, Ju.
        """

    @abstractmethod
    def copy(self) -> BaseSystem:
        """
        Return a copy of the system.
        """
