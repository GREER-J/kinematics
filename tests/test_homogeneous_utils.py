import numpy as np
from src.dynamics_library.homogeneous_utils import get_R_from_A, get_r_from_A, get_r_and_R_from_A, get_e1_axis_from_A, get_e2_axis_from_A, get_e3_axis_from_A
from tests.test_rotation_matrix import assert_valid_rotation_matrix
TOL = 1e-10


def test_get_R_from_A():
    test_A = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    expected_R = np.eye(3)
    actual_R = get_R_from_A(test_A)
    assert_valid_rotation_matrix(actual_R)
    assert np.allclose(actual_R, expected_R, atol=TOL), f"Expected rotation matrix:\n{expected_R}\nGot:\n{actual_R}"


def test_get_r_from_A():
    test_A = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    expected_r = np.array([[1], [0], [0]])
    actual_r = get_r_from_A(test_A)
    assert np.allclose(actual_r, expected_r, atol=TOL), f"Expected translation vector:\n{expected_r}\nGot:\n{actual_r}"


def test_get_r_and_R_from_A():
    test_A = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    expected_r = np.array([[1], [0], [0]])
    expected_R = np.eye(3)
    actual_r, actual_R = get_r_and_R_from_A(test_A)
    assert np.allclose(actual_r, expected_r, atol=TOL), f"Expected translation vector:\n{expected_r}\nGot:\n{actual_r}"
    assert_valid_rotation_matrix(actual_R)
    assert np.allclose(actual_R, expected_R, atol=TOL), f"Expected rotation matrix:\n{expected_R}\nGot:\n{actual_R}"


def test_get_e1_from_A():
    test_A = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    actual_e1 = get_e1_axis_from_A(test_A)
    expected_e1 = np.matrix([1, 0, 0]).T
    assert np.allclose(actual_e1, expected_e1, atol=TOL), f"Expected e1 vector:\n{expected_e1}\nGot:\n{actual_e1}"


def test_get_e2_from_A():
    test_A = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    actual_e2 = get_e2_axis_from_A(test_A)
    expected_e2 = np.matrix([0, 1, 0]).T
    assert np.allclose(actual_e2, expected_e2, atol=TOL), f"Expected e2 vector:\n{expected_e2}\nGot:\n{actual_e2}"


def test_get_e3_from_A():
    test_A = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    actual_e3 = get_e3_axis_from_A(test_A)
    expected_e3 = np.matrix([0, 0, 1]).T
    assert np.allclose(actual_e3, expected_e3, atol=TOL), f"Expected e3 vector:\n{expected_e3}\nGot:\n{actual_e3}"
