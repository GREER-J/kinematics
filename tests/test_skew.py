import numpy as np
import sympy as sp
from src.kinematics_library.skew import skew_np, skew_sp


def test_skew_np():
    # Given we have a vector [1, 2, 3].T
    a1 = 1
    a2 = 2
    a3 = 3
    vec = np.matrix([a1, a2, a3]).T
    # Then the skew of that vector should be:
    exp_skew = np.matrix([[0, -a3, a2], [a3, 0, -a1], [-a2, a1, 0]])
    act_skew = skew_np(vec)
    assert np.allclose(act_skew, exp_skew)


def test_skew_sp():
    # Given we have a vector [1, 2, 3].T
    a1 = sp.Symbol('a_1')
    a2 = sp.Symbol('a_2')
    a3 = sp.Symbol('a_3')
    vec = sp.Matrix([[a1], [a2], [a3]])
    # Then the skew of that vector should be:
    exp_skew = sp.Matrix([[0, -a3, a2], [a3, 0, -a1], [-a2, a1, 0]])
    act_skew = skew_sp(vec)
    assert act_skew == exp_skew
