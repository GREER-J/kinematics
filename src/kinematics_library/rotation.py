import numpy as np
from scipy.linalg import expm
from src.kinematics_library.base_vec import e1, e2, e3
from src.kinematics_library.skew import skew_np


def Rz(psi_rad: float) -> np.ndarray:
    S_e3 = skew_np(e3)
    R = expm(psi_rad * S_e3)
    return R


def Ry(theta_rad: float) -> np.ndarray:
    S_e2 = skew_np(e2)
    R = expm(theta_rad * S_e2)
    return R


def Rx(phi_rad: float) -> np.ndarray:
    S_e1 = skew_np(e1)
    R = expm(phi_rad * S_e1)
    return R


def rpy(phi_rad: float, theta_rad: float, psi_rad: float) -> np.ndarray:
    return Rz(psi_rad)@Ry(theta_rad)@Rx(phi_rad)
