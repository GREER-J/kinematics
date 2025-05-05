import numpy as np
from src.dynamics_library.gaussian_measurement import MeasurementGaussianLikelihood
from src.dynamics_library.system_simulator_estimator import SystemSimulatorEstimator
from src.dynamics_library.gaussian_return import GaussianReturn
from src.dynamics_library.gaussian import Gaussian, AffineMode


class DummySystem(SystemSimulatorEstimator):
    def __init__(self, density: Gaussian):
        super().__init__()
        self.density = density

    def dynamics(self, t, x, u):
        raise NotImplementedError("This is a dummy function")

    def input(self, t):
        raise NotImplementedError("This is a dummy function")


class DummyMeasurement(MeasurementGaussianLikelihood):
    def __init__(self, y: np.ndarray):
        super().__init__(y)
        self.H = np.array([[2.0, 3.0]])
        self.R = np.array([[0.5]])
        system: SystemSimulatorEstimator

    def predict_density(self, x, return_grad=False, return_hessian=False) -> GaussianReturn:
        mu = self.H @ x
        S = np.linalg.cholesky(self.R).T

        rv = GaussianReturn(magnitude=mu, gaussian_magnitude=Gaussian(mu, S))

        if return_grad:
            rv.grad_magnitude = self.H.copy()

        return rv

    def get_pyx(self, x, system: SystemSimulatorEstimator, return_gradient=False, return_hessian=False) -> Gaussian:

        noise_gaussian = Gaussian(np.zeros((3,1)), np.diag([np.sqrt(0.5), 0, 0]))
        pyx = system.density.affine_transform(h=self.predict_density, noise=noise_gaussian, mode=AffineMode.MOMENT)

        return pyx


def test_get_pyx_with_affine_transform():
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
    measurement = DummyMeasurement(np.zeros((1, 1)))  # y doesn't matter in this case

    pyx = measurement.get_pyx(x=mux, system=sys, return_gradient=True)
    assert isinstance(pyx, Gaussian), "h function should return include Gaussian object"
    assert pyx.mu.shape == (3, 1), f"Expected mean shape (3,1), got {pyx.mu.shape}"

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
    np.testing.assert_allclose(pyx.mu, muy_expected, atol=1e-10)

    # Covariance prediction:
    # P_x = diag(1.0, 4.0)

    # H P_x = [2 3] @ diag(1.0, 4.0)
    #       = [2*1.0 + 3*0.0, 2*0.0 + 3*4.0]
    #       = [2.0, 12.0]

    # (H P_x) H^T = (2.0 * 2.0) + (12.0 * 3.0)
    #             = 4.0 + 36.0
    #             = 40.0

    # Add measurement noise R = 0.5
    # P_y = 40.0 + 0.5
    #     = 40.5

    # Cross-covariance:
    # P_yx = [2.0, 12.0]

    # Augmented covariance P_aug:
    # P_aug = [ P_y   P_yx
    #           P_yx' P_x ]

    # Expanded:
    # P_aug = [40.5  2.0  12.0
    #          2.0   1.0  0.0
    #          12.0  0.0  4.0]

    cov_expected = np.array([[40.5, 2.0, 12.0],
                             [2.0, 1.0, 0.0],
                             [12.0, 0.0, 4.0]
                             ])
    assert pyx.cov.shape == (3, 3), f"Expected covariance shape (3,3), got {pyx.cov.shape}"
    np.testing.assert_allclose(pyx.cov, cov_expected, atol=1e-10)
