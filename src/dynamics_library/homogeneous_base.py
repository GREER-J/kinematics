import numpy as np
from scipy.linalg import expm
from src.kinematics_library.skew import skew_np


# def hat_se3(xi: np.ndarray) -> np.ndarray:
#     """
#     Converts a 6D twist vector xi = [v; w] into a 4x4 matrix in se(3).
#     """
#     v = xi[0:3]
#     w = xi[3:6]
#     hat = np.zeros((4, 4))
#     hat[0:3, 0:3] = skew_np(w)
#     hat[0:3, 3] = v
#     return hat

def hat_se3(xi: np.ndarray) -> np.ndarray:
    """
    Converts a 6D twist vector xi = [v; w] into a 4x4 matrix in se(3).
    Assumes skew_np expects 3x1 column vector.
    """
    v = xi[0:3].reshape((3, 1))  # Ensure column vector
    w = xi[3:6].reshape((3, 1))  # Ensure column vector

    hat = np.zeros((4, 4))
    hat[0:3, 0:3] = skew_np(w)
    hat[0:3, 3] = v.flatten()  # flatten in case make_transform expects 1D
    return hat


def tranx_se3(mu: float) -> np.ndarray:
    """Returns homogeneous transformation matrix for a translation along X by mu units."""
    xi = np.array([1, 0, 0, 0, 0, 0])
    return expm(mu * hat_se3(xi))


def trany_se3(mu: float) -> np.ndarray:
    """Returns homogeneous transformation matrix for a translation along X by mu units."""
    xi = np.array([0, 1, 0, 0, 0, 0])
    return expm(mu * hat_se3(xi))


def tranz_se3(mu: float) -> np.ndarray:
    """Returns homogeneous transformation matrix for a translation along X by mu units."""
    xi = np.array([0, 0, 1, 0, 0, 0])
    return expm(mu * hat_se3(xi))


def rotx_se3(mu: float) -> np.ndarray:
    """Returns homogeneous transformation matrix for a rotation about X by mu radians."""
    xi = np.array([0, 0, 0, 1, 0, 0])
    return expm(mu * hat_se3(xi))


def roty_se3(mu: float) -> np.ndarray:
    """Returns homogeneous transformation matrix for a rotation about X by mu radians."""
    xi = np.array([0, 0, 0, 0, 1, 0])
    return expm(mu * hat_se3(xi))


def rotz_se3(mu: float) -> np.ndarray:
    """Returns homogeneous transformation matrix for a rotation about X by mu radians."""
    xi = np.array([0, 0, 0, 0, 0, 1])
    return expm(mu * hat_se3(xi))
