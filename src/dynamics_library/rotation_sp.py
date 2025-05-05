import sympy as sp


# Rotation about the X-axis (Roll)
def Rx_sp(phi):
    return sp.Matrix([
        [1, 0, 0],
        [0, sp.cos(phi), -sp.sin(phi)],
        [0, sp.sin(phi),  sp.cos(phi)]
    ])


# Rotation about the Y-axis (Pitch)
def Ry_sp(theta):
    return sp.Matrix([
        [ sp.cos(theta), 0, sp.sin(theta)],
        [0, 1, 0],
        [-sp.sin(theta), 0, sp.cos(theta)]
    ])


# Rotation about the Z-axis (Yaw)
def Rz_sp(psi):
    return sp.Matrix([
        [sp.cos(psi), -sp.sin(psi), 0],
        [sp.sin(psi),  sp.cos(psi), 0],
        [0, 0, 1]
    ])


def R_zyx_sym(phi, theta, psi):
    return Rz_sp(psi) * Ry_sp(theta) * Rx_sp(phi)
