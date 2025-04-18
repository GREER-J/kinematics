import numpy as np


def transx(x: float) -> np.ndarray:
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def transy(y: float) -> np.ndarray:
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, y],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def transz(z: float) -> np.ndarray:
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])


def rotx(phi_rad: float) -> np.ndarray:
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi_rad), -np.sin(phi_rad), 0],
        [0, np.sin(phi_rad),  np.cos(phi_rad), 0],
        [0, 0, 0, 1]
    ])


def roty(theta_rad: float) -> np.ndarray:
    return np.array([
        [ np.cos(theta_rad), 0, np.sin(theta_rad), 0],
        [ 0,             1, 0,             0],
        [-np.sin(theta_rad), 0, np.cos(theta_rad), 0],
        [ 0,             0, 0,             1]
    ])


def rotz(psi_rad: float) -> np.ndarray:
    return np.array([
        [np.cos(psi_rad), -np.sin(psi_rad), 0, 0],
        [np.sin(psi_rad),  np.cos(psi_rad), 0, 0],
        [0,            0,           1, 0],
        [0,            0,           0, 1]
    ])
