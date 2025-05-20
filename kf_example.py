from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pickle
import copy
from src.dynamics_library.system import BaseSystem
from src.dynamics_library.gaussian_return import GaussianReturn
from functools import partial
from src.dynamics_library.gaussian import Gaussian
from src.dynamics_library.kalman_filter_measurement import ClassicKalmanMeasurement
from src.dynamics_library.gaussian_measurement import MeasurementGaussianLikelihood


class OscillatorySystem(BaseSystem):
    def __init__(self, state: Gaussian, time=0):
        self.time = time
        self.state = state
        self.Q = np.diag([0.1, 0, 0.1])

    @property
    def density(self):
        return self.state

    @property
    def Ad(self):
        f0 = 50
        fs = 1000
        a = 2 * np.cos(2 * np.pi * f0 / fs)

        Ad = np.array([
            [1, 0, 0],
            [0, a, -1],
            [0, 1, 0]
        ])
        return Ad

    def predict(self, time: float) -> OscillatorySystem:
        system_next = self.copy()
        h = partial(self.dynamics, t=time, u=self.input(self.time))
        next_state = self.state.affine_transform(h)

        mu = next_state.mu
        cov = next_state.cov
        next_state = Gaussian.from_moment(mu, cov + self.Q)  # TODO This is not the place to add Q

        system_next.state = next_state
        return system_next

    def input(self, t: float) -> np.ndarray:
        return np.zeros((0, 1))  # No control input

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray, return_grad: bool) -> GaussianReturn:
        magnitude = self.Ad @ x
        rv = GaussianReturn(magnitude=magnitude)

        if return_grad:
            rv.grad_magnitude = self.Ad.copy()

        # rv.gaussian_magnitude = Gaussian.from_moment(np.zeros((3, 1)), self.Q) #TODO Gaussian can't handle zero noise on a state for Q
        return rv

    def copy(self) -> OscillatorySystem:
        return copy.deepcopy(self)


class VoltageMeasurementEvent_kf(ClassicKalmanMeasurement):
    def __init__(self, time: float, y: np.ndarray, **kwargs):
        R = 1
        super().__init__(system=None, y=y, R=R, time=time, update_method='affine', **kwargs)

    def predict_density(self, x: np.ndarray, return_grad=False, return_hessian=False) -> GaussianReturn:
        C = np.array([[1, 1, 0]])
        meas = C @ x

        rv = GaussianReturn(magnitude=meas)

        if return_grad:
            rv.grad_magnitude = C.copy()

        return rv

    @staticmethod
    def get_process_string() -> str:
        return "Updating state"


class VoltageMeasurementEvent(MeasurementGaussianLikelihood):
    def __init__(self, time: float, y: float, R: float = 1.0, **kwargs):
        super().__init__(system=None, time=time, y=np.array([[y]]), update_method="affine", **kwargs)
        self.R = np.array([[R]])  # Support multivariate future

    def predict_density(self, x: np.ndarray, return_grad=False, return_hessian=False) -> GaussianReturn:
        C = np.array([[1, 1, 0]])
        meas = C @ x
        noise = Gaussian.from_moment(np.array(0), np.array(self.R))

        rv = GaussianReturn(magnitude=meas, gaussian_magnitude=noise)

        if return_grad:
            rv.grad_magnitude = C.copy()

        return rv

    @staticmethod
    def get_process_string() -> str:
        return "Updating state"


# --- Load data ---
data = loadmat("data/data_lectorial5.mat")
y = data["y"].flatten()

measurement_times = np.arange(len(y)) * 1 / 1000
event_que = [VoltageMeasurementEvent(time=measurement_times[i], y=np.array(y[i]).reshape((1,1)), verbosity=0) for i in range(len(y))]

event_que.sort()

# --- Simulation ---
xhp = np.array([[y[0]], [0], [0]])  # Initial state estimate
Pp = 5 * np.eye(3)                  # Initial covariance

sys = OscillatorySystem(Gaussian.from_moment(xhp, Pp))

print("mu0")
print(xhp)

x = Gaussian.from_moment(xhp, Pp)

event_que[0].system = sys
x1 = event_que[0].augmented_predict_density(x.mu, return_grad=True)

print(x1.magnitude)
print('\n')
print(x1.grad_magnitude)

# Run simulation
# for i, event in enumerate(event_que):
#     _, sys = event.process(sys)

# Process data
# N = len(y)
# xh = np.zeros((N, 3))               # Filtered states
# KK_data = np.zeros((N, 3))          # Kalman gain over time
# Pm_data = np.zeros((N, 3))          # Covariance diagonals

# kf_log = []  # List to store data at each timestep
# for i, event in enumerate(event_que):
#     xh[i, :] = event.system.state.mu.flatten()
#     KK_data[i, :] = event.Kk.flatten()
#     Pm_data[i, :] = np.diag(event.system.state.cov)

#     # Save current step
#     kf_log.append({"step": i,
#                    "xhp": event.system.state.mu.copy(),
#                    "Pp": event.system.state.cov.copy(),
#                    "Kk": event.Kk.copy()
#                    })

# print(f"First measurement: {event_que[0].y}")
# print(f"\nmu: {event_que[0].system.state.mu.copy()}")
# print(f"\ncov: {event_que[0].system.state.cov.copy()}")
# print(f"\nKk: {event_que[0].Kk.copy()}")
# # --- Save new data once
# # with open("data/kf_output.pkl", "wb") as f:
# #     pickle.dump(kf_log, f)
# # --- Save new data once

# # Correctness tests
# with open("data/kf_output.pkl", "rb") as f:
#     ref_log = pickle.load(f)

# for i, (ref, test) in enumerate(zip(ref_log, kf_log)):
#     for key in ["xhp", "Pp", "Kk"]:
#         all_close = np.allclose(ref[key], test[key])
#         if not all_close:
#             print(f"\n Mismatch at step {i} in {key}")
#             print("Ref:\n", ref[key])
#             print("Test:\n", test[key])
#             print("Diff:\n", ref[key] - test[key])
#             assert all_close
#         else:
#             continue
#     break  # Stop after first failure
# print("Tests complete :)")

# --- Plotting ---
# plt.figure()
# plt.plot(y, 'b', label='Measured')
# plt.plot(xh[:, 0], 'r', label='Filtered (state 1)')
# plt.title('Filtered Output vs Measured')
# plt.legend()

# # Kalman gain plots
# fig, axes = plt.subplots(3, 1, figsize=(6, 6))
# axes[0].plot(KK_data[:, 0])
# axes[0].set_title("Kalman gain for y_bar")
# axes[1].plot(KK_data[:, 1])
# axes[1].set_title("Kalman gain for z(t)")
# axes[2].plot(KK_data[:, 2])
# axes[2].set_title("Kalman gain for z(t-1)")
# plt.tight_layout()
# plt.show()
