import numpy as np
from src.kinematics_library.guassian import Gaussian
from src.kinematics_library.gaussian_return import GaussianReturn
# Assuming Gaussian and GaussianReturn already imported


def transform_2x_transform(x: np.ndarray, return_grad=False) -> GaussianReturn:
    """
    Very simple 2 x tranform
    """
    A = 2 * np.eye(2)

    f = A@x

    rv = GaussianReturn(magnitude=f)

    J = A
    if return_grad:
        rv.grad_magnitude = J

    return rv


def transform_square_x1(x: np.ndarray, return_grad=False) -> GaussianReturn:
    """
    Simple transform:
    y1 = x1^2
    y2 = x2
    """
    x1, x2 = x[0, 0], x[1, 0]

    f = np.array([
        [x1 ** 2],
        [x2]
    ])

    grad = None
    if return_grad:
        grad = np.array([
            [2 * x1, 0],
            [0, 1]
        ])

    return GaussianReturn(
        magnitude=f,
        grad_magnitude=grad
    )


def test_gaussian_transform_affine_2x():
    mux = np.array([[1.0], [2.0]])
    Sx = np.array([[0.1, 0.0], [0.0, 0.1414213562]])
    assert np.allclose(Sx, np.triu(Sx)), "Input Sx should be upper triangular"

    px = Gaussian.from_sqrt_moment(mux, Sx)
    assert px.mu.shape == (2, 1), f"Expected mean shape (2,1), got {px.mu.shape}"
    assert px.cov.shape == (2, 2), f"Expected covariance shape (2,2), got {px.cov.shape}"

    # Affine transform
    py = px.affine_transform(transform_2x_transform)

    assert isinstance(py, Gaussian), "Affine transform should return a Gaussian object"
    assert py.mu.shape == (2, 1), f"Expected mean shape (2,1), got {py.mu.shape}"
    assert py.cov.shape == (2, 2), f"Expected covariance shape (2,2), got {py.cov.shape}"

    # Check mean
    muy_expected = np.array([[2.0], [4.0]])
    np.testing.assert_allclose(py.mu, muy_expected, atol=1e-10)

    # Check covariance
    Py_expected = np.array([
        [0.04, 0.0],
        [0.0, 0.08]
    ])
    np.testing.assert_allclose(py.cov, Py_expected, atol=1e-10)


def test_gaussian_transform_affine_square_x1():
    mux = np.array([[2.0], [3.0]])
    Sx = np.array([[0.1, 0.0], [0.0, 0.1]])
    assert np.allclose(Sx, np.triu(Sx)), "Input Sx should be upper triangular"

    px = Gaussian.from_sqrt_moment(mux, Sx)
    assert px.mu.shape == (2, 1), f"Expected mean shape (2,1), got {px.mu.shape}"
    assert px.cov.shape == (2, 2), f"Expected covariance shape (2,2), got {px.cov.shape}"

    # Affine transform
    py = px.affine_transform(transform_square_x1)

    assert isinstance(py, Gaussian), "Affine transform should return a Gaussian object"
    assert py.mu.shape == (2, 1), f"Expected mean shape (2,1), got {py.mu.shape}"
    assert py.cov.shape == (2, 2), f"Expected covariance shape (2,2), got {py.cov.shape}"

    # Check mean
    muy_expected = np.array([[4.0], [3.0]])
    np.testing.assert_allclose(py.mu, muy_expected, atol=1e-10)

    # Check covariance
    Py_expected = np.array([
        [0.16, 0.0],
        [0.0, 0.01]
    ])
    np.testing.assert_allclose(py.cov, Py_expected, atol=1e-10)
