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


def get_e1_axis_from_A(A: np.ndarray) -> np.ndarray:
    """
    Extracts the e1 unit vector from a 4x4 homogeneous transform.
    """
    assert A.shape == (4, 4), f"Expected a 4x4 matrix, got shape {A.shape}."
    return np.reshape(A[0:3, 0], (3,1))


def get_e2_axis_from_A(A: np.ndarray) -> np.ndarray:
    """
    Extracts the e2 unit vector from a 4x4 homogeneous transform.
    """
    assert A.shape == (4, 4), f"Expected a 4x4 matrix, got shape {A.shape}."
    return np.reshape(A[0:3, 1], (3,1))


def get_e3_axis_from_A(A: np.ndarray) -> np.ndarray:
    """
    Extracts the e3 unit vector from a 4x4 homogeneous transform.
    """
    assert A.shape == (4, 4), f"Expected a 4x4 matrix, got shape {A.shape}."
    return np.reshape(A[0:3, 2], (3,1))
