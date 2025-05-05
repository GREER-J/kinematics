import numpy as np
from matplotlib import pyplot as plt
from src.dynamics_library.system_simulator_estimator import SystemSimulatorEstimator
from src.dynamics_library.gaussian_measurement import MeasurementGaussianLikelihood
from src.dynamics_library.gaussian import Gaussian


class OscillatingMeasurement(MeasurementGaussianLikelihood):
    def __init__(self, y: np.ndarray, R: float = 1e-2):
        super().__init__(y)
        self.H = np.array([[1.0, 1.0, 0.0]])
        self.R = np.array([[R]])

    def predict_density(self, x, system, return_gradient=False, return_hessian=False):
        mu = self.H @ x
        S = np.linalg.cholesky(self.R).T
        py = Gaussian(mu, S)

        if return_hessian:
            dhdx = self.H
            d2hdx2 = np.zeros((1, x.shape[0], x.shape[0]))
            return py, dhdx, d2hdx2

        if return_gradient:
            dhdx = self.H
            return py, dhdx

        return py


class OscillatorySystem(SystemSimulatorEstimator):
    def __init__(self):
        super().__init__()

        mu0 = np.zeros((3, 1))  # Default initial state: [y_bar(t), z(t), z(t-1)]
        S0 = np.diag([10, 10, 10])
        self.density = Gaussian(mu0, S0)

        self.time = 0
        self.x_sim = np.zeros((3,1))  # This is the simulated state

        self.f0 = 1_000    # Hz (or whatever makes sense)
        self.fs = 100.0  # Sample rate
        self.w0 = 2 * np.pi * self.f0 / self.fs
        self.c_w0 = np.cos(self.w0)

    # def dynamics(self, t: float, x: np.ndarray, u: np.ndarray):
    #     """
    #     Computes the time derivative dx/dt (or difference x_{t+1} - x_t in discrete setup)
    #     """
    #     # Unpack state: x = []
    #     A = np.array([
    #         [1.0, 0.0, 0.0],
    #         [0.0, self.c_w0, -1.0],
    #         [0.0, 1.0, 0.0]
    #     ])

    #     dx = A @ x
    #     assert self.x_sim.shape == (3, 1), f"x_sim shape was {self.x_sim.shape}"

    #     return dx, None, None

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray):
        """
        x = [ y_bar, z_k, z_k-1 ]
        """
        y_bar = x[0, 0]
        z_k = x[1, 0]
        z_k1 = x[2, 0]

        # Next z
        w = 0  # optional noise
        z_next = 2 * self.c_w0 * z_k - z_k1 + w
        y_next = z_k + 0  # no e

        x_next = np.array([
            [y_next],
            [z_next],
            [z_k]
        ])

        dx = x_next - x  # discrete delta (pseudo-derivative)
        return dx, None, None

    def predict(self, time_next: float) -> "OscillatorySystem":
        """
        Discrete-time prediction using:
            x_next = A x
        """
        A = np.array([
            [1, 0, 0],
            [0, np.cos(self.w0), -1],
            [0, 1, 0]
        ])
        self.x_sim = A @ self.x_sim  # advance one time step
        self.time = time_next
        return self

    def input(self, t):
        # Placeholder: zero input for both motors
        return np.zeros(1)


y_data = np.loadtxt("y.csv", delimiter=",")  # shape: (T,) or (T,1)
y_data = y_data.reshape(-1, 1)  # Ensure shape is (T, 1)


fs = 1000  # Hz
dt = 1.0 / fs
T = y_data.shape[0]
event_queue = []

system = OscillatorySystem()

for i in range(T):
    t = i * dt
    meas = OscillatingMeasurement(y_data[i])  # shape (1,)
    meas.verbosity = 1
    meas.time = t
    meas.need_to_simulate = False  # using real data
    meas.update_method = 'affine'  # or 'unscented', etc.
    event_queue.append(meas)

for i, event in enumerate(event_queue):
    event_queue[i], system = event.process(system)

print("Simulation complete")

# Post process
n_events = len(event_queue)

t_hist = np.full((1, n_events,), np.nan)
x_hist = np.full((3, n_events), np.nan)
y_hist = np.full((1, n_events), np.nan)

for k, event in enumerate(event_queue):
    if event.system is None:
        continue  # skip if state wasn't saved
    t = event.time
    x_sim = event.system.x_sim
    t_hist[:, k] = t
    y_hist[:, k] = y_data[k]
    x_hist[:, k] = x_sim.flatten()

plt.figure()
plt.plot(t_hist.flatten(), y_hist.flatten(), label="Noisy Measurements (y)")
plt.plot(t_hist.flatten(), x_hist[0, :], label="Filtered Output (x1)", color="red")
plt.xlabel("Time Step")
plt.ylabel("Signal")
plt.legend()
plt.title("Filtered Signal vs Noisy Measurements")
plt.grid()
plt.show()
