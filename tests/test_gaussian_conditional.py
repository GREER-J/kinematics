import numpy as np
from src.kinematics_library.guassian import Gaussian  # adjust if needed
from numpy.testing import assert_allclose


def run_conditional_case(mu, S, idxA, idxB, xB):
    p = Gaussian.from_sqrt_moment(np.array(mu).reshape(-1, 1), np.array(S))
    xB = np.array(xB).reshape(-1, 1)

    pc = p.conditional(idxA, idxB, xB)
    muc_actual = pc.mu
    Sc_actual = pc.sqrt_cov
    Pc_actual = pc.cov

    mu = np.array(mu).reshape(-1, 1)
    S = np.array(S)
    P = S.T @ S

    # Expected conditional mean
    muA = mu[idxA]
    muB = mu[idxB]
    PAB = P[np.ix_(idxA, idxB)]
    PBB = P[np.ix_(idxB, idxB)]
    expected_muc = muA + PAB @ np.linalg.solve(PBB, xB - muB)
    assert_allclose(muc_actual, expected_muc, atol=1e-12)

    # Check that Sc is upper-triangular
    assert np.allclose(Sc_actual, np.triu(Sc_actual)), "Sqrt covariance not upper-triangular"

    # Expected conditional covariance
    PAA = P[np.ix_(idxA, idxA)]
    expected_Pc = PAA - PAB @ np.linalg.solve(PBB, PAB.T)
    assert_allclose(Pc_actual, expected_Pc, atol=1e-10)


def test_conditional_43_head():
    mu = [1, 1, 1, 1]
    S = [[-0.649, -1.110, -0.559,  0.586],
         [0,     -0.846,  0.178, -0.852],
         [0,      0,     -0.197,  0.800],
         [0,      0,      0,     -1.509]]
    xB = [0.876, -0.243, 0.167]
    run_conditional_case(mu, S, idxA=[3], idxB=[0, 1, 2], xB=xB)


def test_conditional_43_tail():
    mu = [1, 1, 1, 1]
    S = [[-0.649, -1.110, -0.559,  0.586],
         [0,     -0.846,  0.178, -0.852],
         [0,      0,     -0.197,  0.800],
         [0,      0,      0,     -1.509]]
    xB = [0.876, -0.243, 0.167]
    run_conditional_case(mu, S, idxA=[0], idxB=[1, 2, 3], xB=xB)


def test_conditional_41_segment():
    mu = [1, 1, 1, 1]
    S = [[-0.649, -1.110, -0.559,  0.586],
         [0,     -0.846,  0.178, -0.852],
         [0,      0,     -0.197,  0.800],
         [0,      0,      0,     -1.509]]
    xB = [0.876]
    run_conditional_case(mu, S, idxA=[0, 1, 3], idxB=[2], xB=xB)


def test_conditional_63_noncontiguous():
    mu = [1, 1, 1, 1, 1, 1]
    S = [[-0.649, -1.110, -0.559,  0.586, -1.509,  0.167],
         [0,     -0.846,  0.178, -0.852,  0.876, -1.965],
         [0,      0,     -0.197,  0.800, -0.243, -1.270],
         [0,      0,      0,      1.175,  0.604, -1.865],
         [0,      0,      0,      0,      1.781, -1.051],
         [0,      0,      0,      0,      0,     -0.417]]
    xB = [0.876, -0.243, 0.167]
    run_conditional_case(mu, S, idxA=[0, 2, 4], idxB=[1, 3, 5], xB=xB)
