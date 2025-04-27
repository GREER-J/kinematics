import numpy as np
from numdifftools import Gradient, Hessian
from src.kinematics_library.guassian import Gaussian


def test_log_pdf_value():
    x = np.array([[1], [2], [3]])
    mu = np.array([[2], [4], [6]])
    S = np.diag([1, 2, 3])
    g = Gaussian.from_sqrt_moment(mu, S)

    expected = -6.04857506884207
    actual = g.log(x)
    np.testing.assert_allclose(actual.magnitude, expected, atol=1e-12)


def test_nominal_gradient():
    x = np.array([[1], [2], [3]])
    mu = np.array([[2], [4], [6]])
    S = np.diag([1, 2, 3])
    p = Gaussian.from_sqrt_moment(mu, S)
    l, grad = p.log(x, return_grad=True)
    grad_fn = Gradient(lambda v: float(p.log(v.reshape(-1, 1)).magnitude))
    expected = grad_fn(x.flatten()).reshape(-1, 1)

    np.testing.assert_allclose(grad, expected, atol=1e-8)


def test_nominal_hessian():
    x = np.array([[1], [2], [3]])
    mu = np.array([[2], [4], [6]])
    S = np.diag([1, 2, 3])
    p = Gaussian.from_sqrt_moment(mu, S)
    l, grad, hess = p.log(x, return_grad=True, return_hess=True)
    hess_fn = Hessian(lambda v: float(p.log(v.reshape(-1, 1)).magnitude))
    expected = hess_fn(x.flatten())

    hess = np.squeeze(hess)  # removes trailing dim (3,3,1) → (3,3)
    np.testing.assert_allclose(hess, expected, atol=1e-8)


def test_multiple_x():
    x = np.array([[1, 2, 3, 4, 5],
                  [2, 4, 6, 8, 10],
                  [3, 6, 9, 12, 15]])
    mu = np.array([[2], [4], [6]])
    S = np.diag([1, 2, 3])
    p = Gaussian.from_sqrt_moment(mu, S)
    expected = np.array([
        -6.04857506884207, -4.54857506884207, -6.04857506884207,
        -10.5485750688421, -18.0485750688421
    ])
    actual = p.log(x)
    np.testing.assert_allclose(actual.magnitude, expected, atol=1e-12)


def test_negative_S():
    x = np.array([[1], [2], [3]])
    mu = np.array([[2], [4], [6]])
    S = np.diag([-1, -2, -3])
    p = Gaussian.from_sqrt_moment(mu, S)
    actual = p.log(x)
    assert np.isreal(actual)


def test_underflow():
    x = np.array([[0]])
    mu = np.sqrt(350 * np.log(10) / np.pi)  # Approx 16
    S = 1 / np.sqrt(2 * np.pi)  # Approx 0.4
    p = Gaussian.from_sqrt_moment(np.array([[mu]]), np.array([[S]]))
    z = (x - mu) / S
    expected = np.array([-0.5 * float(z**2)])  # -805.904782547916
    actual = p.log(x)
    np.testing.assert_allclose(actual.magnitude, expected, atol=1e-10)


def test_det_underflow():
    a = 1e-4
    n = 100
    S = a * np.eye(n)
    mu = np.zeros((n, 1))
    x = np.zeros((n, 1))
    p = Gaussian.from_sqrt_moment(mu, S)
    expected = -n * np.log(a) - 0.5 * n * np.log(2 * np.pi)
    actual = p.log(x)
    np.testing.assert_allclose(actual.magnitude, expected, atol=1e-5)


def test_det_overflow():
    a = 1e4
    n = 100
    S = a * np.eye(n)
    mu = np.zeros((n, 1))
    x = np.zeros((n, 1))
    p = Gaussian.from_sqrt_moment(mu, S)
    expected = -n * np.log(a) - 0.5 * n * np.log(2 * np.pi)
    actual = p.log(x)
    np.testing.assert_allclose(actual.magnitude, expected, atol=1e-5)


def test_covariance_overflow():
    n = 2
    mu = np.zeros((n, 1))
    S = 1e300 * np.eye(n)
    p = Gaussian.from_sqrt_moment(mu, S)
    assert not np.all(np.isfinite(p.cov))
    x = np.zeros((n, 1))
    actual = p.log(x)
    assert np.isfinite(actual.magnitude)


def test_covariance_inverse_edge_case():
    e = 1.5 * np.sqrt(np.finfo(float).eps)
    mu = np.array([[0], [e]])
    S = np.array([[1, np.sqrt(1 - e**2)], [0, e]])
    x = np.array([[0], [0]])
    p = Gaussian.from_sqrt_moment(mu, S)
    n = p.dim()
    expected = -0.5 - np.log(e) - n/2 * np.log(2 * np.pi)
    actual = p.log(x)
    np.testing.assert_allclose(actual.magnitude, expected, atol=1e-5)
