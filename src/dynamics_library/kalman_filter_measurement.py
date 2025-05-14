from __future__ import annotations
from abc import abstractmethod
import numpy as np
from src.dynamics_library.measurement import Measurement
from src.dynamics_library.gaussian_return import GaussianReturn
from src.dynamics_library.gaussian import Gaussian


# TODO implement steady state KF as perhaps a superclass of this?
class ClassicKalmanMeasurement(Measurement):
    def __init__(self, y: np.ndarray, system, time: float, R: np.ndarray, update_method='NewtonTrustEig', need_to_simulate=False, **kwargs):
        super().__init__(system, time, update_method, need_to_simulate, **kwargs)
        self.y = y
        self.R = R

    @abstractmethod
    def predict_density(self, x: np.ndarray, return_grad=False, return_hessian=False) -> GaussianReturn:
        ...

    def simulate(self, x: np.ndarray, system) -> np.ndarray:
        raise NotImplementedError()

    def log_likelihood(self, x):
        raise NotImplementedError()

    def _do_update(self) -> None:
        system = self.system
        predicted = system.state.affine_transform(self.predict_density)
        S = predicted.cov + self.R  # R is scalar, so this works for 1D output | #TODO This is not the place to add R?

        # Kalman gain
        P = system.state.cov
        H = self.predict_density(system.state.mu, return_grad=True).grad_magnitude
        Kk = P @ H.T @ np.linalg.inv(S)

        # Innovation
        y_pred = predicted.mu
        innovation = self.y - y_pred

        # Posterior update
        new_mean = system.state.mu + Kk @ innovation
        I = np.eye(system.state.dim())
        new_cov = (I - Kk @ H) @ P @ (I - Kk @ H).T + Kk * self.R * Kk.T

        system.time = self.time
        system.state = Gaussian.from_moment(new_mean, new_cov)
        self.Kk = Kk.copy()
