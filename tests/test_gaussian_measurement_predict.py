import numpy as np
from src.kinematics_library.gaussian_measurement import MeasurementGaussianLikelihood
from src.kinematics_library.gaussian_return import GaussianReturn
from src.kinematics_library.guassian import Gaussian


class TestMeasurement(MeasurementGaussianLikelihood):
    def __init__(self, y: np.ndarray):
        super().__init__(y)
        self.H = np.array([[1.0, 3.0]])
        self.R = np.array([[0.5]])

    def predict_density(self, x, system, return_gradient=False, return_hessian=False) -> GaussianReturn:
        mu = self.H @ x
        S = np.linalg.cholesky(self.R).T

        rv = GaussianReturn(magnitude=mu, gaussian_magnitude=Gaussian(mu, S))

        if return_gradient:
            rv.grad_magnitude = self.H

        return rv


def test_gaussian_measurement():
    mux = np.array([
        [1.0],
        [2.0]
    ])

    measurement = TestMeasurement(np.zeros((1, 1)))

    rv = measurement.predict_density(x=mux, system=None, return_gradient=True)
    assert isinstance(rv, GaussianReturn), "h function should return a GaussianReturn object"

    py = rv.gaussian_magnitude
    assert isinstance(py, Gaussian), "h function should return include Gaussian object"
    assert py.mu.shape == (1, 1), f"Expected mean shape (1,1), got {py.mu.shape}"
    assert py.cov.shape == (1, 1), f"Expected covariance shape (1,1), got {py.cov.shape}"


    # Expected mean
    # h(x) = 1*x1 + 3*x2 = 1*1 + 3*2 = 1 + 6 = 7
    muy_expected = np.array([
        [7.0]
    ])

    # Check mean
    np.testing.assert_allclose(py.mu, muy_expected, atol=1e-10)

    # Expected covariance
    Py_expected = measurement.R

    np.testing.assert_allclose(py.cov, Py_expected, atol=1e-10)
