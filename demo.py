import numpy as np
from matplotlib import pyplot as plt
from src.dynamics_library.gaussian import Gaussian
from src.dynamics_library.gaussian_measurement import MeasurementGaussianLikelihood

# ------------------------
# 0. Scenario Configuration
# ------------------------

# True position (what we're estimating)
x_true = np.array([[2.0], [3.5], [0.0]])

# Two fixed beacon locations
beacon_1 = np.array([[2.0], [0.0], [1.0]])
beacon_2 = np.array([[0.0], [3.0], [1.0]])


# ------------------------
# 1. Define Measurement Class
# ------------------------
def dhdx_mat(x):
    diffs = [x - beacon_1, x - beacon_2]  # list of (3,1) vectors
    J = np.zeros((2, 3))  # (number of measurements, state dimension)
    for i, d in enumerate(diffs):
        norm_d = np.linalg.norm(d)
        if norm_d == 0:
            raise ValueError("x is exactly at beacon location, gradient undefined.")
        J[i, :] = (d / norm_d).flatten()
    return J


class BeaconMeasurement(MeasurementGaussianLikelihood):
    def __init__(self):
        super().__init__()
        self.R = np.diag([0.5, 0.5])  # Standard deviation = 0.707m

    def predict_density(self, x, system=None, return_gradient=False, return_hessian=False):
        y = np.array([
            [np.linalg.norm(x - beacon_1)],
            [np.linalg.norm(x - beacon_2)]
        ]).reshape(-1, 1)

        py = Gaussian.from_moment(y, self.R)

        if return_gradient:
            dhdx = dhdx_mat(x)
            return py, dhdx

        return py


meas = BeaconMeasurement()

# ------------------------
# 2. Simulate measurement from true position
# ------------------------

y_true = np.array([
    [np.linalg.norm(x_true - beacon_1)],
    [np.linalg.norm(x_true - beacon_2)]
])

print(f"y_true: {y_true}")

y_obs = y_true + np.random.multivariate_normal(mean=np.zeros(2), cov=meas.R).reshape(-1, 1)
print(f"y_obs: {y_obs}")
print(f"delta: {y_obs-y_true}")

# ------------------------
# 3. Prior belief over x
# ------------------------

mu_x = np.array([[1.0], [2.0], [0.0]])          # Initial guess
P = np.diag([1.0, 1.0, 0.2])                   # Uncertainty
prior = Gaussian.from_moment(mu_x, P)


y_guess = meas.predict_density(mu_x)
print(f"y_guess: {y_guess.mu}")

# ------------------------
# 4. Compute joint over [y; x] using augmented_predict_density
# ------------------------


class DummySystem:
    def __init__(self, density: Gaussian):
        self.density = density

dummy_system = DummySystem(density=prior)


joint, _ = meas.augmented_predict_density(mu_x, dummy_system, return_gradient=True)

print(f"Join: {joint.mu}")
print(f"Join: {joint.sqrt_cov}")

# ------------------------
# 5. Conditioning: Get posterior over x given y
# ------------------------

ny = y_obs.shape[0]
nx = mu_x.shape[0]
idx_y = list(range(ny))
idx_x = list(range(ny, ny + nx))

posterior = joint.conditional(idx_x, idx_y, y_obs)

# ------------------------
# 6. Print and Visualize
# ------------------------

print("True position:\n", x_true)
print("Observed y (noisy distances):\n", y_obs)
print("Prior mean:\n", prior.mu)
print("Posterior mean:\n", posterior.mu)
print("Posterior sqrt covariance:\n", posterior.sqrt_cov)
