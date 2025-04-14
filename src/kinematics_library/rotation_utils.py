import numpy as np


def get_e1_axis_from_R(R: np.ndarray) -> np.ndarray:
    """
    Extracts the e1 unit vector from a 3x3 rotation matrix.
    """
    assert R.shape == (3, 3), f"Expected a 3x3 matrix, got shape {R.shape}."
    return np.reshape(R[:, 0], (3,1))


def get_e2_axis_from_R(R: np.ndarray) -> np.ndarray:
    """
    Extracts the e2 unit vector from a 3x3 rotation matrix.
    """
    assert R.shape == (3, 3), f"Expected a 3x3 matrix, got shape {R.shape}."
    return np.reshape(R[:, 1], (3,1))


def get_e3_axis_from_R(R: np.ndarray) -> np.ndarray:
    """
    Extracts the e3 unit vector from a 3x3 rotation matrix.
    """
    assert R.shape == (3, 3), f"Expected a 3x3 matrix, got shape {R.shape}."
    return np.reshape(R[:, 2], (3,1))
