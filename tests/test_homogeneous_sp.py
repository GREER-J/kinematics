import sympy as sp
import numpy as np
from numpy.testing import assert_allclose
from src.dynamics_library.homogeneous_sp import rotx_sp, roty_sp, rotz_sp, transx_sp, transy_sp, transz_sp
from src.dynamics_library.homogeneous import rotx, roty, rotz, transx, transy, transz

TOL = 1e-10


def test_rotx_trig_vs_expm():
    gamma = sp.symbols('gamma')
    for angle in np.linspace(-2*np.pi, 2*np.pi, 100):
        R_expm_sym = rotx_sp(gamma).evalf(subs={gamma: angle})
        R_expm_np = np.array(R_expm_sym.tolist(), dtype=np.float64)
        R_trig = rotx(angle)
        assert_allclose(R_expm_np, R_trig, atol=TOL)


def test_roty_trig_vs_expm():
    gamma = sp.symbols('gamma')
    for angle in np.linspace(-2*np.pi, 2*np.pi, 100):
        R_expm_sym = roty_sp(gamma).evalf(subs={gamma: angle})
        R_expm_np = np.array(R_expm_sym.tolist(), dtype=np.float64)
        R_trig = roty(angle)
        assert_allclose(R_expm_np, R_trig, atol=TOL)


def test_rotz_trig_vs_expm():
    gamma = sp.symbols('gamma')
    for angle in np.linspace(-2*np.pi, 2*np.pi, 100):
        R_expm_sym = rotz_sp(gamma).evalf(subs={gamma: angle})
        R_expm_np = np.array(R_expm_sym.tolist(), dtype=np.float64)
        R_trig = rotz(angle)
        assert_allclose(R_expm_np, R_trig, atol=TOL)


def test_transx_trig_vs_expm():
    gamma = sp.symbols('gamma')
    for angle in np.linspace(-2*np.pi, 2*np.pi, 100):
        R_expm_sym = transx_sp(gamma).evalf(subs={gamma: angle})
        R_expm_np = np.array(R_expm_sym.tolist(), dtype=np.float64)
        R_trig = transx(angle)
        assert_allclose(R_expm_np, R_trig, atol=TOL)


def test_transy_trig_vs_expm():
    gamma = sp.symbols('gamma')
    for angle in np.linspace(-2*np.pi, 2*np.pi, 100):
        R_expm_sym = transy_sp(gamma).evalf(subs={gamma: angle})
        R_expm_np = np.array(R_expm_sym.tolist(), dtype=np.float64)
        R_trig = transy(angle)
        assert_allclose(R_expm_np, R_trig, atol=TOL)


def test_transz_trig_vs_expm():
    gamma = sp.symbols('gamma')
    for angle in np.linspace(-2*np.pi, 2*np.pi, 100):
        R_expm_sym = transz_sp(gamma).evalf(subs={gamma: angle})
        R_expm_np = np.array(R_expm_sym.tolist(), dtype=np.float64)
        R_trig = transz(angle)
        assert_allclose(R_expm_np, R_trig, atol=TOL)
