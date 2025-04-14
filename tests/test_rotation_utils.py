import numpy as np
from src.kinematics_library.rotation_utils import get_e1_axis, get_e2_axis, get_e3_axis
TOL = 1e-10


def test_get_e1_from_R():
    test_R = np.eye(3)
    actual_e1 = get_e1_axis(test_R)
    expected_e1 = np.matrix([1, 0, 0]).T
    assert np.allclose(actual_e1, expected_e1, atol=TOL), f"Expected e1 vector:\n{expected_e1}\nGot:\n{actual_e1}"


def test_get_e2_from_R():
    test_R = np.eye(3)
    actual_e2 = get_e2_axis(test_R)
    expected_e2 = np.matrix([0, 1, 0]).T
    assert np.allclose(actual_e2, expected_e2, atol=TOL), f"Expected e1 vector:\n{expected_e2}\nGot:\n{actual_e2}"


def test_get_e3_from_R():
    test_R = np.eye(3)
    actual_e3 = get_e3_axis(test_R)
    expected_e3 = np.matrix([0, 0, 1]).T
    assert np.allclose(actual_e3, expected_e3, atol=TOL), f"Expected e1 vector:\n{expected_e3}\nGot:\n{actual_e3}"
