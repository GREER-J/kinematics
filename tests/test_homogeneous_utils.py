import numpy as np
from src.kinematics_library.homogeneous_utils import get_R_from_A, get_r_from_A, get_r_and_R_from_A
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
    assert np.allclose(actual_R, expected_R, atol=TOL), f"Expected rotation matrix:\n{expected_R}\nGot:\n{actual_R}"
