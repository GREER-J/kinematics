import numpy as np


def get_R_from_A(A: np.ndarray) -> np.ndarray:
    """
    Extracts the 3x3 rotation matrix from a 4x4 homogeneous transformation matrix.
    """
    assert A.shape == (4, 4), f"Expected a 4x4 matrix, got shape {A.shape}."
    return A[:3, :3]


def get_r_from_A(A: np.ndarray) -> np.ndarray:
    """
    Extracts the 3x1 translation vector from a 4x4 homogeneous transformation matrix.
    """
    assert A.shape == (4, 4), f"Expected a 4x4 matrix, got shape {A.shape}."
    return A[:3, 3].reshape((3, 1))


def get_r_and_R_from_A(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts the 3x1 translation vector and 3x3 rotation matrix from a 4x4 homogeneous transformation matrix.
    """
    assert A.shape == (4, 4), f"Expected a 4x4 matrix, got shape {A.shape}."
    return (get_r_from_A(A), get_R_from_A(A))
