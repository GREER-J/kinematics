from src.kinematics_library.guassian import Gaussian, GaussianReturn
from src.kinematics_library.gaussian_measurement import MeasurementGaussianLikelihood
from src.kinematics_library.system_simulator_estimator import SystemSimulatorEstimator
from src.kinematics_library.state import GaussianState
import numpy as np


class DummySystem(SystemSimulatorEstimator):
    def __init__(self, state: GaussianState, time: float = 0.0):
        super().__init__(state=state, time=time)
        self.state = state

    def dynamics(self, t, x, u):
        raise NotImplementedError("This is a dummy function")

    def input(self, t):
        raise NotImplementedError("This is a dummy function")


class DummyMeasurement(MeasurementGaussianLikelihood):
    def __init__(self, time, y, system, **kwargs):
        super().__init__(time, y, system, **kwargs)
        self.H = np.array([[2.0, 3.0]])
        self.R = np.array([[0.5]])

    def predict_density(self, x, return_gradient=False, return_hessian=False) -> GaussianReturn:
        mu = self.H @ x
        S = np.linalg.cholesky(self.R).T

        rv = GaussianReturn(magnitude=mu, gaussian_magnitude=Gaussian(mu, S))

        if return_gradient:
            rv.grad_magnitude = self.H.copy()

        return rv
