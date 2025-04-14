import numpy as np


def assert_valid_rotation_matrix(R: np.ndarray, tol: float = 1e-6) -> None:
    """
    Assert that a matrix R is a valid 3x3 rotation matrix.

    Parameters:
    R (np.ndarray): The matrix to test.
    tol (float): Tolerance for numerical comparisons.

    Raises:
    AssertionError: If R does not satisfy the properties of a rotation matrix.
    """
    # Check if R is a 3x3 matrix
    assert R.shape == (3, 3), f"Matrix shape is {R.shape}, expected (3, 3)."

    # Check orthogonality: R.T @ R == I
    should_be_identity = np.dot(R.T, R)
    I = np.identity(3)
    if not np.allclose(should_be_identity, I, atol=tol):
        diff = should_be_identity - I
        raise AssertionError(f"Matrix is not orthogonal. Difference:\n{diff}")

    # Check determinant: det(R) == 1
    det = np.linalg.det(R)
    assert np.isclose(det, 1.0, atol=tol), f"Determinant is {det}, expected 1.0."
