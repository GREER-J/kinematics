import sympy as sp
import numpy as np
from numpy.testing import assert_allclose
from src.dynamics_library.rotation_sp import Rx_sp, Ry_sp, Rz_sp
from src.dynamics_library.rotation_trig import Rx_trig, Ry_trig, Rz_trig
from tests.test_rotation_matrix import assert_valid_rotation_matrix

TOL = 1e-10  # Very tight tolerance


def test_Rz_gives_valid_rotation_matrix():
    gamma = sp.symbols('gamma')
    angle = 0
    R_expm_sym = Rz_sp(gamma).evalf(subs={gamma: angle})
    R_expm_np = np.array(R_expm_sym.tolist(), dtype=np.float64)
    assert_valid_rotation_matrix(R_expm_np)


def test_Ry_gives_valid_rotation_matrix():
    gamma = sp.symbols('gamma')
    angle = 0
    R_expm_sym = Ry_sp(gamma).evalf(subs={gamma: angle})
    R_expm_np = np.array(R_expm_sym.tolist(), dtype=np.float64)
    assert_valid_rotation_matrix(R_expm_np)


def test_Rx_gives_valid_rotation_matrix():
    gamma = sp.symbols('gamma')
    angle = 0
    R_expm_sym = Rx_sp(gamma).evalf(subs={gamma: angle})
    R_expm_np = np.array(R_expm_sym.tolist(), dtype=np.float64)
    assert_valid_rotation_matrix(R_expm_np)


def test_rotx_sp_vs_trig():
    gamma = sp.symbols('gamma')
    for angle in np.linspace(-2*np.pi, 2*np.pi, 100):
        R_expm_sym = Rx_sp(gamma).evalf(subs={gamma: angle})
        R_expm_np = np.array(R_expm_sym.tolist(), dtype=np.float64)
        R_trig = Rx_trig(angle)
        assert_allclose(R_expm_np, R_trig, atol=TOL)


def test_roty_sp_vs_trig():
    gamma = sp.symbols('gamma')
    for angle in np.linspace(-2*np.pi, 2*np.pi, 100):
        R_expm_sym = Ry_sp(gamma).evalf(subs={gamma: angle})
        R_expm_np = np.array(R_expm_sym.tolist(), dtype=np.float64)
        R_trig = Ry_trig(angle)
        assert_allclose(R_expm_np, R_trig, atol=TOL)


def test_rotz_sp_vs_trig():
    gamma = sp.symbols('gamma')
    for angle in np.linspace(-2*np.pi, 2*np.pi, 100):
        R_expm_sym = Rz_sp(gamma).evalf(subs={gamma: angle})
        R_expm_np = np.array(R_expm_sym.tolist(), dtype=np.float64)
        R_trig = Rz_trig(angle)
        assert_allclose(R_expm_np, R_trig, atol=TOL)
