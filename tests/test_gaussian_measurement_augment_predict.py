import numpy as np
from src.kinematics_library.gaussian_measurement import MeasurementGaussianLikelihood
from src.kinematics_library.system_simulator_estimator import SystemSimulatorEstimator
from src.kinematics_library.gaussian_return import GaussianReturn
from src.kinematics_library.guassian import Gaussian


class DummySystem(SystemSimulatorEstimator):
    def __init__(self, density: Gaussian):
        super().__init__()
        self.density = density

    def dynamics(self, t, x, u):
        raise NotImplementedError("This is a dummy function")
        return super().dynamics(t, x, u)

    def input(self, t):
        raise NotImplementedError("This is a dummy function")
        return super().input(t)


class TestMeasurement(MeasurementGaussianLikelihood):
        super().__init__(y)
        self.H = np.array([[2.0, 3.0]])
        self.R = np.array([[0.5]])

    def predict_density(self, x, system, return_gradient=False, return_hessian=False) -> GaussianReturn:
        mu = self.H @ x
        S = np.linalg.cholesky(self.R).T

        rv = GaussianReturn(magnitude=mu, gaussian_magnitude=Gaussian(mu, S))

        if return_gradient:
            rv.grad_magnitude = self.H.copy()

        return rv


def test_augment_simple_measurement_function():
    mux = np.array([
        [1.0],
        [2.0]
    ])

    Px = np.diag([1.0, 4.0])

    px = Gaussian.from_moment(mu=mux, P=Px)
    assert isinstance(px, Gaussian), "In setup px should be a Gaussian"
    assert px.mu.shape == (2, 1), f"Expected mean shape (2,1), got {px.mu.shape}"
    assert px.cov.shape == (2, 2), f"Expected covariance shape (2,2), got {px.cov.shape}"

    sys = DummySystem(density=px)
    measurement = TestMeasurement(np.zeros((1, 1)))  # y doesn't matter in this case

    rv = measurement.augmented_predict_density(x=mux, system=sys, return_gradient=True)
    assert isinstance(rv, GaussianReturn), "h function should return a GaussianReturn object"
    assert rv.has_grad, "h was asked for a grad ... it should have one"

    p_ayx = rv.gaussian_magnitude
    assert isinstance(p_ayx, Gaussian), "h function should return include Gaussian object"
    assert p_ayx.mu.shape == (3, 1), f"Expected mean shape (3,1), got {p_ayx.mu.shape}"
    assert p_ayx.cov.shape == (3, 3), f"Expected covariance shape (3,3), got {p_ayx.cov.shape}"

    # Measurement model:
    # y = H x
    # where H = [2 3]

    # State mean:
    # x = [1.0;
    #      2.0]

    # Predict measurement:
    # y = (2 * 1.0) + (3 * 2.0)
    # y = 2.0 + 6.0
    # y = 8.0

    # mu_aug = [ y;
    #            x1;
    #            x2 ]
    #        = [ 8.0;
    #            1.0;
    #            2.0 ]

    muy_expected = np.array([[8.0], [1.0], [2.0]])
    np.testing.assert_allclose(p_ayx.mu, muy_expected, atol=1e-10)

    # Jacobian of h(x) w.r.t x:
    # dh/dx = [2.0  3.0]  (1x2)

    # Augmented Jacobian Ja is:
    # Ja = [2.0  3.0;
    #       1.0  0.0;
    #       0.0  1.0]

    dhdx_expected = np.array([[2.0, 3.0],
                              [1.0, 0.0],
                              [0.0, 1.0]
                              ])
    assert rv.grad_magnitude.shape == (3, 2), f"Expected grad shape (3,2), got {rv.grad_magnitude.shape}"
    np.testing.assert_allclose(rv.grad_magnitude, dhdx_expected, atol=1e-10)
