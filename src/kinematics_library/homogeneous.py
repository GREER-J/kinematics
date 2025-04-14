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


def rotx(phi: float) -> np.ndarray:
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi),  np.cos(phi), 0],
        [0, 0, 0, 1]
    ])


def roty(theta: float) -> np.ndarray:
    return np.array([
        [ np.cos(theta), 0, np.sin(theta), 0],
        [ 0,             1, 0,             0],
        [-np.sin(theta), 0, np.cos(theta), 0],
        [ 0,             0, 0,             1]
    ])


def rotz(psi: float) -> np.ndarray:
    return np.array([
        [np.cos(psi), -np.sin(psi), 0, 0],
        [np.sin(psi),  np.cos(psi), 0, 0],
        [0,            0,           1, 0],
        [0,            0,           0, 1]
    ])
