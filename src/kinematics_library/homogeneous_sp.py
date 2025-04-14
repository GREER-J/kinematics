import sympy as sp


def transx_sp(x):
    return sp.Matrix([[1, 0, 0, x],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])


def transy_sp(y):
    return sp.Matrix([[1, 0, 0, 0],
                      [0, 1, 0, y],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])


def transz_sp(z):
    return sp.Matrix([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, z],
                      [0, 0, 0, 1]])


def rotx_sp(phi):
    return sp.Matrix([[1, 0, 0, 0],
                      [0, sp.cos(phi), -sp.sin(phi), 0],
                      [0, sp.sin(phi), sp.cos(phi), 0],
                      [0, 0, 0, 1]])


def roty_sp(theta):
    return sp.Matrix([[sp.cos(theta), 0, sp.sin(theta), 0],
                      [0, 1, 0, 0],
                      [-sp.sin(theta), 0, sp.cos(theta), 0],
                      [0, 0, 0, 1]])


def rotz_sp(psi):
    return sp.Matrix([[sp.cos(psi), -sp.sin(psi), 0, 0],
                      [sp.sin(psi), sp.cos(psi), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
