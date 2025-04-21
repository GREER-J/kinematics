from abc import ABC, abstractmethod
from typing import Tuple


class BaseSystem(ABC):
    def __init__(self):
        self.time: float = 0.0

    @abstractmethod
    def predict(self, time: float) -> "BaseSystem":
        """
        Predict system state forward to the given time.
        """

    @abstractmethod
    def dynamics(self, t: float, x, u) -> Tuple:
        """
        Compute system dynamics f(t, x, u), and optionally return Jacobians Jx, Ju.
        """

    @abstractmethod
    def copy(self) -> "BaseSystem":
        """
        Return a copy of the system.
        """
