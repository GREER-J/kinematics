import numpy as np


def Rz_trig(psi_rad: float) -> np.ndarray:
    c, s = np.cos(psi_rad), np.sin(psi_rad)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])


def Ry_trig(theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])


def Rx_trig(phi_rad: float) -> np.ndarray:
    c, s = np.cos(phi_rad), np.sin(phi_rad)
    return np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]
    ])


def rpy(phi_rad: float, theta_rad: float, psi_rad: float) -> np.ndarray:
    return Rz_trig(psi_rad)@Ry_trig(theta_rad)@Rx_trig(phi_rad)


def euler_from_matrix(R: np.ndarray) -> tuple[float, float, float]:
    """
    Extracts Euler angles (roll, pitch, yaw) from a 3x3 rotation matrix.
    Assumes ZYX rotation order: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    Returns:
        roll (phi), pitch (theta), yaw (psi) in radians
    """
    assert R.shape == (3, 3), "Rotation matrix must be 3x3"

    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-6):
        raise ValueError("Matrix determinant is not 1. Rotation matrix may be invalid.")

    # Pitch angle (theta)
    theta = -np.arcsin(R[2, 0])

    # Handle gimbal lock case (cos(theta) ~ 0)
    if np.isclose(np.cos(theta), 0.0, atol=1e-6):
        # Gimbal lock, infinite solutions for roll + yaw
        phi = 0.0
        psi = np.arctan2(-R[0, 1], R[1, 1])
    else:
        phi = np.arctan2(R[2, 1], R[2, 2])  # roll
        psi = np.arctan2(R[1, 0], R[0, 0])  # yaw

    return phi, theta, psi
