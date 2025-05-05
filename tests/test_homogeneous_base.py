import numpy as np
from numpy.testing import assert_allclose
from src.dynamics_library.homogeneous_base import rotx_se3, roty_se3, rotz_se3, tranx_se3, trany_se3, tranz_se3


def test_rot_x_90():
    A = rotx_se3(np.deg2rad(90))
    expected = np.array([
        [1, 0,  0, 0],
        [0, 0, -1, 0],
        [0, 1,  0, 0],
        [0, 0,  0, 1]
    ])
    assert_allclose(A, expected, atol=1e-10)


def test_rot_y_90():
    A = roty_se3(np.deg2rad(90))
    expected = np.array([
        [ 0, 0, 1, 0],
        [ 0, 1, 0, 0],
        [-1, 0, 0, 0],
        [ 0, 0, 0, 1]
    ])
    assert_allclose(A, expected, atol=1e-10)


def test_rot_z_90():
    A = rotz_se3(np.deg2rad(90))
    expected = np.array([
        [0, -1, 0, 0],
        [1,  0, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ])
    assert_allclose(A, expected, atol=1e-10)


def test_translate_x():
    A = tranx_se3(1)
    expected = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    assert_allclose(A, expected, atol=1e-10)


def test_translate_y():
    A = trany_se3(1)
    expected = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    assert_allclose(A, expected, atol=1e-10)


def test_translate_z():
    A = tranz_se3(1)
    expected = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1]
    ])
    assert_allclose(A, expected, atol=1e-10)
