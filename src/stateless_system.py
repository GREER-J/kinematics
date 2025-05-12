from src.dynamics_library.system import BaseSystem
from src.dynamics_library.gaussian_measurement_function import MeasurementFunctionProtocol
from src.dynamics_library.gaussian import Gaussian
from typing import Protocol, Optional
import numpy as np
from src.dynamics_library.state import State


class dynamicsFunctionProtocol(Protocol):
    def __call__(self, x: np.ndarray,
                 return_grad: bool = False,
                 return_hess: bool = False
                 ) -> "GaussianReturn":
        """
        A function that maps an input Gaussian mean (n,1) to a GaussianReturn:
          - magnitude: (m,1) array (output mean)
          - optionally grad_magnitude: Jacobian (m,n)
          - optionally hess_magnitude: Hessian (not usually needed for affine)
        """


class StatelessSystem(BaseSystem):
    def __init__(
        self,
        f: dynamicsFunctionProtocol,  # dynamics
        h: MeasurementFunctionProtocol,  # measurement
        noise: Optional[Gaussian] = None,
    ):
        self._f = f
        self._h = h
        self.noise = noise or Gaussian.zero((1, 1))  # measurement noise model

    def predict(self, t: float) -> State:
        # Could implement Euler or use a model-defined step function
        raise NotImplementedError("StatelessSystem doesn't advance time by default")

    def h(self, x: np.ndarray) -> "GaussianReturn":
        return self._h(x)

    def get_measurement_model(self) -> MeasurementModel:
        return AffineMeasurementModel(h=self._h, noise=self.noise)


def linear_heat_model(x, u, dt):
    A = np.eye(2) - dt / 120 * np.eye(2)
    B = dt * np.array([[0.8], [0.2]])
    return A @ x + B @ u


def h_T1_only(x):
    y = np.array([[x[0, 0]]])
    J = np.array([[1.0, 0.0]])
    return y, J

