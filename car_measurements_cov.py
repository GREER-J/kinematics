from __future__ import annotations
from src.dynamics_library.system import BaseSystem
from src.dynamics_library.state import GaussianState
from src.dynamics_library.event import BaseEvent, LogEvent, StateUpdateEvent
from src.dynamics_library.gaussian import Gaussian
import numpy as np
from matplotlib import pyplot as plt


class CarState(GaussianState):
    def __init__(self, x: np.ndarray, P: np.ndarray):
        super().__init__(distribution=Gaussian.from_moment(mu=x, P=P))


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
        mu, cov = self.state.to_vector()
        x_next = self.dynamics(time, mu, None)
        system_next.state = CarState.from_array(x_next)
        return system_next

    def copy(self):
        return Car(state=self.state.copy(), time=self.time)


T_update = 0.5
T_log = 2.5
T_end = 10.0

update_times = np.arange(0, T_end + 1e-6, T_update)
log_times = np.arange(0, T_end + 1e-6, T_log)

x0 = CarState(np.array([[1], [2], [3], [4]]), 10*np.eye(4))
car_plant = Car(x0)

events: list[BaseEvent] = []
events.extend(state_upd_events := [StateUpdateEvent(time=t) for t in update_times])
events.extend([LogEvent(time=t) for t in log_times])
events.sort()

# Do sim simulation
for event in events:
    _, car_plant = event.process(car_plant)

# Plots
times = np.zeros(len(update_times))
xs = np.zeros((4, len(update_times)))

for i, event in enumerate(state_upd_events):
    xs[:, i] = event.system.state.to_vector().flatten()
    times[i] = event.time

# Plot x, y scatter plot
X, Y, Vx, Vy = xs[0], xs[1], xs[2], xs[3]

# XY Scatter Plot
plt.figure()
plt.scatter(X, Y)
plt.title("XY Position Scatter")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)

# State Time-Series Subplots
plt.figure(figsize=(8, 10))

plt.subplot(4, 1, 1)
plt.plot(times, X)
plt.ylabel("x")
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(times, Y)
plt.ylabel("y")
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(times, Vx)
plt.ylabel("vx")
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(times, Vy)
plt.ylabel("vy")
plt.xlabel("time [s]")
plt.grid(True)

plt.tight_layout()
plt.show()
