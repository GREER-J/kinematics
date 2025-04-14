import numpy as np


def skew(u: np.ndarray) -> np.ndarray:
    a1 = u[0, 0]
    a2 = u[1, 0]
    a3 = u[2, 0]
    return np.matrix([[0, -a3, a2], [a3, 0, -a1], [-a2, a1, 0]])


def test_skew():
    # Given we have a vector [1, 2, 3].T
    a1 = 1
    a2 = 2
    a3 = 3
    vec = np.matrix([a1, a2, a3]).T
    # Then the skew of that vector should be:
    exp_skew = np.matrix([[0, -a3, a2], [a3, 0, -a1], [-a2, a1, 0]])
    act_skew = skew(vec)
    assert np.allclose(act_skew, exp_skew)
