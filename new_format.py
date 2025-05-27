import numpy as np
from dataclasses import dataclass
from src.dynamics_library.event import BaseEvent
from src.dynamics_library.system import BaseSystem
from src.dynamics_library.integration import rk4, IntegratorFunction


class SubSystem:
    def __init__(self, x0: np.ndarray, dynamics: IntegratorFunction, t0: float = 0.0):
        self.t = t0
        self.x = x0
        self.f = dynamics
    
    def predict(self, time, x0):
        pass

class System:
    def __init__(self, subsystems: list[SubSystem]):
        self.subsystems = subsystems

    def predict(self, t):
        self.x = self.F(self.x)

A = np.eye(2)
x0 = np.zeros((2,1))

sim = System([SubSystem(lambda x: A@x, x0)])

@dataclass
class LogEvent(BaseEvent):
    def update(self, system: BaseSystem) -> None:
        print(f" [LOG] {system.state}", end='')

    @staticmethod
    def get_process_string() -> str:
        return "Logging state"


print("Rounds complete")
