from __future__ import annotations
from src.kinematics_library.system import BaseSystem
from src.kinematics_library.state import MeanState
from src.kinematics_library.event import BaseEvent, LogEvent, StateUpdateEvent
import numpy as np


class CarState(MeanState):
    def __init__(self, x: np.ndarray):
        assert x.shape == (4, 1), "Expected a 4×1 column vector"
        super().__init__(x)

    @classmethod
    def from_components(cls, x: float, y: float, vx: float, vy: float) -> CarState:
        vec = np.array([[x], [y], [vx], [vy]])
        return cls(vec)

    def __repr__(self):
        return f"x = [{self.x_pos}, {self.y_pos}, {self.vx}, {self.vy}]"

    @classmethod
    def from_array(cls, x: np.ndarray) -> CarState:
        return CarState(x)

    @property
    def x_pos(self):
        return self.x[0, 0]

    @property
    def y_pos(self):
        return self.x[1, 0]

    @property
    def vx(self):
        return self.x[2, 0]

    @property
    def vy(self):
        return self.x[3, 0]

    def copy(self):
        return CarState(self.x.copy())


class Car(BaseSystem):
    def __init__(self, state: CarState, time: float = 0.0):
        super().__init__(state=state, time=time)
        self.state = state

    def F_matrix(self, dt) -> np.ndarray:
        _F_cv = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        return _F_cv

    def dynamics(self, t, x, u) -> np.ndarray:
        dt = t - self.time
        F = self.F_matrix(dt)
        return F@x

    def predict(self, time) -> Car:
        system_next = self.copy()
        x_next = self.dynamics(time, self.state.to_vector(), None)
        system_next.state = CarState.from_array(x_next)
        return system_next

    def copy(self):
        return Car(state=self.state.copy(), time=self.time)


T_update = 1.0
T_log = 2.5
T_end = 10.0

update_times = np.arange(0, T_end + 1e-6, T_update)
log_times = np.arange(0, T_end + 1e-6, T_log)

x0 = CarState(np.array([[1], [2], [3], [4]]))
car_plant = Car(x0)

events: list[BaseEvent] = []
events.extend([StateUpdateEvent(time=t) for t in update_times])
events.extend([LogEvent(time=t) for t in log_times])
events.sort()

# Do sim simulation
for event in events:
    _, car_plant = event.process(car_plant)
