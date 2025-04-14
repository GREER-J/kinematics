import numpy as np
import pytest
from src.kinematics_library.rotation_expm import Rz, Ry, Rx, rpy


def identity():
    return np.eye(3)


def assert_matrix_close(actual, expected, tol=1e-6):
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)


def test_Rz_zero_angle():
    assert_matrix_close(Rz(0), identity())


def test_Rz_positive_90_degrees():
    expected = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ])
    assert_matrix_close(Rz(np.deg2rad(90)), expected)


def test_Rz_negative_90_degrees():
    expected = np.array([
        [ 0, 1, 0],
        [-1, 0, 0],
        [ 0, 0, 1]
    ])
    assert_matrix_close(Rz(np.deg2rad(-90)), expected)


def test_Rz_full_rotation_360_degrees():
    assert_matrix_close(Rz(np.deg2rad(360)), identity())


def test_Rz_arbitrary_angle():
    angle = 45
    angle_rad = np.radians(angle)
    expected = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0,                 0,                 1]
    ])
    assert_matrix_close(Rz(angle_rad), expected)


def test_Ry_zero_angle():
    assert_matrix_close(Ry(0), identity())


def test_Ry_positive_90_degrees():
    expected = np.array([
        [ 0, 0, 1],
        [ 0, 1, 0],
        [-1, 0, 0]
    ])
    assert_matrix_close(Ry(np.deg2rad(90)), expected)


def test_Ry_negative_90_degrees():
    expected = np.array([
        [ 0, 0, -1],
        [ 0, 1,  0],
        [ 1, 0,  0]
    ])
    assert_matrix_close(Ry(np.deg2rad(-90)), expected)


def test_Ry_full_rotation_360_degrees():
    assert_matrix_close(Ry(np.deg2rad(360)), identity())


def test_Ry_arbitrary_angle():
    angle = 30
    angle_rad = np.radians(angle)
    expected = np.array([
        [ np.cos(angle_rad), 0, np.sin(angle_rad)],
        [ 0,                 1, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])
    assert_matrix_close(Ry(angle_rad), expected)


def test_Rx_zero_angle():
    assert_matrix_close(Rx(0), identity())


def test_Rx_positive_90_degrees():
    expected = np.array([
        [1,  0,  0],
        [0,  0, -1],
        [0,  1,  0]
    ])
    assert_matrix_close(Rx(np.deg2rad(90)), expected)


def test_Rx_negative_90_degrees():
    expected = np.array([
        [1,  0,  0],
        [0,  0,  1],
        [0, -1,  0]
    ])
    assert_matrix_close(Rx(np.deg2rad(-90)), expected)


def test_Rx_full_rotation_360_degrees():
    assert_matrix_close(Rx(np.deg2rad(360)), identity())


def test_Rx_arbitrary_angle():
    angle = 60
    angle_rad = np.radians(angle)
    expected = np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad), -np.sin(angle_rad)],
        [0, np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    assert_matrix_close(Rx(angle_rad), expected)


def test_rpy_zero_rotation():
    assert_matrix_close(rpy(0, 0, 0), identity())


def test_rpy_roll_only():
    assert_matrix_close(rpy(np.deg2rad(90), 0, 0), Rx(np.deg2rad(90)))


def test_rpy_pitch_only():
    assert_matrix_close(rpy(0, np.deg2rad(90), 0), Ry(np.deg2rad(90)))


def test_rpy_yaw_only():
    assert_matrix_close(rpy(0, 0, np.deg2rad(90)), Rz(np.deg2rad(90)))


def test_rpy_combined_rotation():
    phi, theta, psi = np.deg2rad(30), np.deg2rad(45), np.deg2rad(60)
    result = rpy(phi, theta, psi)
    expected = Rz(phi) @ Ry(theta) @ Rx(psi)
    assert_matrix_close(result, expected)


def test_rpy_commutativity():
    A = rpy(np.deg2rad(30), np.deg2rad(45), np.deg2rad(60))
    B = rpy(np.deg2rad(60), np.deg2rad(45), np.deg2rad(30))
    with pytest.raises(AssertionError):
        assert_matrix_close(A, B)


def test_rpy_inverse_rotations():
    R = rpy(np.deg2rad(30), np.deg2rad(45), np.deg2rad(60))
    R_inv = rpy(np.deg2rad(-30), np.deg2rad(-45), np.deg2rad(-60))
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
    assert_matrix_close(rpy(np.deg2rad(450), 0, 0), Rx(np.deg2rad(90)))


def test_Rz_orthogonality():
    R = Rz(np.deg2rad(45))
    assert_matrix_close(R @ R.T, identity())


def test_Rz_determinant():
    assert np.isclose(np.linalg.det(Rz(np.deg2rad(45))), 1.0)


def test_Ry_orthogonality():
    R = Ry(np.deg2rad(45))
    assert_matrix_close(R @ R.T, identity())


def test_Ry_determinant():
    assert np.isclose(np.linalg.det(Ry(np.deg2rad(45))), 1.0)


def test_Rx_orthogonality():
    R = Rx(np.deg2rad(45))
    assert_matrix_close(R @ R.T, identity())


def test_Rx_determinant():
    assert np.isclose(np.linalg.det(Rx(np.deg2rad(45))), 1.0)


def test_rpy_orthogonality():
    R = rpy(np.deg2rad(30), np.deg2rad(45), np.deg2rad(60))
    assert_matrix_close(R @ R.T, identity())


def test_rpy_determinant():
    assert np.isclose(np.linalg.det(rpy(np.deg2rad(30), np.deg2rad(45), np.deg2rad(60))), 1.0)
