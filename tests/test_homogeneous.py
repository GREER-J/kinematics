import numpy as np
from numpy.testing import assert_allclose
from src.kinematics_library.homogeneous_base import rotx_se3, roty_se3, rotz_se3, tranx_se3, trany_se3, tranz_se3
from src.kinematics_library.homogeneous import rotx, roty, rotz, transx, transy, transz

TOL = 1e-10


def test_rotx_trig_vs_expm():
    for angle in np.linspace(-2*np.pi, 2*np.pi, 100):
        R_expm = rotx_se3(angle)
        R_trig = rotx(angle)
        assert_allclose(R_expm, R_trig, atol=TOL)


def test_roty_trig_vs_expm():
    for angle in np.linspace(-2*np.pi, 2*np.pi, 100):
        R_expm = roty_se3(angle)
        R_trig = roty(angle)
        assert_allclose(R_expm, R_trig, atol=TOL)


def test_rotz_trig_vs_expm():
    for angle in np.linspace(-2*np.pi, 2*np.pi, 100):
        R_expm = rotz_se3(angle)
        R_trig = rotz(angle)
        assert_allclose(R_expm, R_trig, atol=TOL)


def test_transx_trig_vs_expm():
    for d in np.linspace(-5.0, 5.0, 50):
        T_expm = tranx_se3(d)
        T_trig = transx(d)
        assert_allclose(T_expm, T_trig, atol=TOL)


def test_transy_trig_vs_expm():
    for d in np.linspace(-5.0, 5.0, 50):
        T_expm = trany_se3(d)
        T_trig = transy(d)
        assert_allclose(T_expm, T_trig, atol=TOL)


def test_transz_trig_vs_expm():
    for d in np.linspace(-5.0, 5.0, 50):
        T_expm = tranz_se3(d)
        T_trig = transz(d)
        assert_allclose(T_expm, T_trig, atol=TOL)
