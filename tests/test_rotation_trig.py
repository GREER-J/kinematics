import pytest
import numpy as np
from scipy.linalg import expm
from src.kinematics_library.base_vec import e1, e2, e3
from src.kinematics_library.skew import skew_np
from src.kinematics_library.rotation_trig import Rx_trig, Ry_trig, Rz_trig, rpy, euler_from_matrix
from tests.test_rotation_matrix import assert_valid_rotation_matrix

TOL = 1e-10  # Very tight tolerance


def test_Rz_gives_valid_rotation_matrix():
    assert_valid_rotation_matrix(Rz_trig(0))


def test_Ry_gives_valid_rotation_matrix():
    assert_valid_rotation_matrix(Ry_trig(0))


def test_Rx_gives_valid_rotation_matrix():
    assert_valid_rotation_matrix(Rx_trig(0))


def test_rx_trig_vs_expm():
    for angle in np.linspace(-2*np.pi, 2*np.pi, 100):
        R_expm = expm(angle * skew_np(e1))
        R_trig = Rx_trig(angle)
        np.testing.assert_allclose(R_expm, R_trig, atol=TOL)


def test_ry_trig_vs_expm():
    for angle in np.linspace(-2*np.pi, 2*np.pi, 100):
        R_expm = expm(angle * skew_np(e2))
        R_trig = Ry_trig(angle)
        np.testing.assert_allclose(R_expm, R_trig, atol=TOL)


def test_rz_trig_vs_expm():
    for angle in np.linspace(-2*np.pi, 2*np.pi, 100):
        R_expm = expm(angle * skew_np(e3))
        R_trig = Rz_trig(angle)
        np.testing.assert_allclose(R_expm, R_trig, atol=TOL)


def identity():
    return np.eye(3)


def assert_matrix_close(actual, expected, tol=1e-6):
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)


def test_Rz_zero_angle():
    assert_matrix_close(Rx_trig(0), identity())


def test_Rz_positive_90_degrees():
    expected = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ])
    assert_matrix_close(Rz_trig(np.deg2rad(90)), expected)


def test_Rz_negative_90_degrees():
    expected = np.array([
        [ 0, 1, 0],
        [-1, 0, 0],
        [ 0, 0, 1]
    ])
    assert_matrix_close(Rz_trig(np.deg2rad(-90)), expected)


def test_Rz_full_rotation_360_degrees():
    assert_matrix_close(Rz_trig(np.deg2rad(360)), identity())


def test_Rz_arbitrary_angle():
    angle = 45
    angle_rad = np.radians(angle)
    expected = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0,                 0,                 1]
    ])
    assert_matrix_close(Rz_trig(angle_rad), expected)


def test_Ry_zero_angle():
    assert_matrix_close(Ry_trig(0), identity())


def test_Ry_positive_90_degrees():
    expected = np.array([
        [ 0, 0, 1],
        [ 0, 1, 0],
        [-1, 0, 0]
    ])
    assert_matrix_close(Ry_trig(np.deg2rad(90)), expected)


def test_Ry_negative_90_degrees():
    expected = np.array([
        [ 0, 0, -1],
        [ 0, 1,  0],
        [ 1, 0,  0]
    ])
    assert_matrix_close(Ry_trig(np.deg2rad(-90)), expected)


def test_Ry_full_rotation_360_degrees():
    assert_matrix_close(Ry_trig(np.deg2rad(360)), identity())


def test_Ry_arbitrary_angle():
    angle = 30
    angle_rad = np.radians(angle)
    expected = np.array([
        [ np.cos(angle_rad), 0, np.sin(angle_rad)],
        [ 0,                 1, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])
    assert_matrix_close(Ry_trig(angle_rad), expected)


def test_Rx_zero_angle():
    assert_matrix_close(Rx_trig(0), identity())


def test_Rx_positive_90_degrees():
    expected = np.array([
        [1,  0,  0],
        [0,  0, -1],
        [0,  1,  0]
    ])
    assert_matrix_close(Rx_trig(np.deg2rad(90)), expected)


def test_Rx_negative_90_degrees():
    expected = np.array([
        [1,  0,  0],
        [0,  0,  1],
        [0, -1,  0]
    ])
    assert_matrix_close(Rx_trig(np.deg2rad(-90)), expected)


def test_Rx_full_rotation_360_degrees():
    assert_matrix_close(Rx_trig(np.deg2rad(360)), identity())


def test_Rx_arbitrary_angle():
    angle = 60
    angle_rad = np.radians(angle)
    expected = np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad), -np.sin(angle_rad)],
        [0, np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    assert_matrix_close(Rx_trig(angle_rad), expected)


def test_rpy_zero_rotation():
    assert_matrix_close(rpy(0, 0, 0), identity())


def test_rpy_roll_only():
    assert_matrix_close(rpy(np.deg2rad(90), 0, 0), Rx_trig(np.deg2rad(90)))


def test_rpy_pitch_only():
    assert_matrix_close(rpy(0, np.deg2rad(90), 0), Ry_trig(np.deg2rad(90)))


def test_rpy_yaw_only():
    assert_matrix_close(rpy(0, 0, np.deg2rad(90)), Rz_trig(np.deg2rad(90)))


def test_rpy_combined_rotation():
    phi, theta, psi = np.deg2rad(30), np.deg2rad(45), np.deg2rad(60)
    result = rpy(phi, theta, psi)
    expected = Rz_trig(psi) @ Ry_trig(theta) @ Rx_trig(phi)
    assert_matrix_close(result, expected)


def test_rpy_commutativity():
    A = rpy(np.deg2rad(30), np.deg2rad(45), np.deg2rad(60))
    B = rpy(np.deg2rad(60), np.deg2rad(45), np.deg2rad(30))
    with pytest.raises(AssertionError):
        assert_matrix_close(A, B)


def test_rpy_inverse_rotations():
    phi = np.deg2rad(30)
    theta = np.deg2rad(45)
    psi = np.deg2rad(60)
    R = rpy(phi, theta, psi)
    R_inv = rpy(-phi, -theta, -psi)
    assert_matrix_close(R @ R_inv, identity())


def test_rpy_gimbal_lock():
    R1 = rpy(np.deg2rad(90), np.deg2rad(90), 0)
    R2 = rpy(0, np.deg2rad(90), np.deg2rad(90))
    with pytest.raises(AssertionError):
        assert_matrix_close(R1, R2)


def test_rpy_full_360_rotation():
    assert_matrix_close(rpy(np.deg2rad(360), 0, 0), identity())
    assert_matrix_close(rpy(0, np.deg2rad(360), 0), identity())
    assert_matrix_close(rpy(0, 0, np.deg2rad(360)), identity())


def test_rpy_over_360_degrees():
    assert_matrix_close(rpy(np.deg2rad(450), 0, 0), Rx_trig(np.deg2rad(90)))


def test_Rz_orthogonality():
    R = Rx_trig(np.deg2rad(45))
    assert_matrix_close(R @ R.T, identity())


def test_Rz_determinant():
    assert np.isclose(np.linalg.det(Rx_trig(np.deg2rad(45))), 1.0)


def test_Ry_orthogonality():
    R = Ry_trig(np.deg2rad(45))
    assert_matrix_close(R @ R.T, identity())


def test_Ry_determinant():
    assert np.isclose(np.linalg.det(Ry_trig(np.deg2rad(45))), 1.0)


def test_Rx_orthogonality():
    R = Rx_trig(np.deg2rad(45))
    assert_matrix_close(R @ R.T, identity())


def test_Rx_determinant():
    assert np.isclose(np.linalg.det(Rx_trig(np.deg2rad(45))), 1.0)


def test_rpy_orthogonality():
    R = rpy(np.deg2rad(30), np.deg2rad(45), np.deg2rad(60))
    assert_matrix_close(R @ R.T, identity())


def test_rpy_determinant():
    assert np.isclose(np.linalg.det(rpy(np.deg2rad(30), np.deg2rad(45), np.deg2rad(60))), 1.0)


def test_euler_from_matrix_roundtrip():
    angles = [(0, 0, 0), (np.pi/4, np.pi/6, np.pi/3), (-np.pi/2, np.pi/2, -np.pi/4)]
    for phi, theta, psi in angles:
        R = rpy(phi, theta, psi)
        phi_r, theta_r, psi_r = euler_from_matrix(R)
        np.testing.assert_allclose([phi_r, theta_r, psi_r], [phi, theta, psi], atol=1e-6)
