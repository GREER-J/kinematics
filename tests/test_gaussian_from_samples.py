import numpy as np
from src.kinematics_library.statistics.guassian import Gaussian


def test_gaussian_from_samples_mean():
    X = np.array([
        [-0.15508, -1.0443, -1.1714,  0.92622, -0.55806],
        [ 0.61212, -0.34563, -0.68559, -1.4817, -0.028453]
    ])
    g = Gaussian.from_samples(X)
    mu_expected = np.array([[-0.40052], [-0.38585]])
    np.testing.assert_allclose(g.mu, mu_expected, atol=1e-5)


def test_gaussian_from_samples_cov():
    X = np.array([
        [-0.15508, -1.0443, -1.1714,  0.92622, -0.55806],
        [ 0.61212, -0.34563, -0.68559, -1.4817, -0.028453]
    ])
    g = Gaussian.from_samples(X)
    P_expected = np.array([
        [0.7135,   -0.26502],
        [-0.26502,  0.60401]
    ])
    np.testing.assert_allclose(g.cov, P_expected, atol=1e-5)
