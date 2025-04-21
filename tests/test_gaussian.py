import numpy as np
from src.kinematics_library.guassian import Gaussian


def run_gaussian_case(p: Gaussian, mu_expected, P_expected, eta_expected, Lambda_expected):
    n = p.dim()

    # Mu checks
    mu_actual = p.mu
    assert mu_actual.shape == (n, 1), f"Expected mu shape (n,1), got {mu_actual.shape}"
    np.testing.assert_allclose(mu_actual, mu_expected, atol=1e-10)

    # Eta
    eta_actual = p.info_vec
    assert eta_actual.shape == (n, 1), f"Expected eta shape (n,1), got {eta_actual.shape}"
    np.testing.assert_allclose(eta_actual, eta_expected, atol=1e-10)

    # Xi
    Xi_actual = p.sqrt_info_mat
    assert Xi_actual.shape == (n, n), f"Expected Xi shape (n,n), got {Xi_actual.shape}"
    assert np.allclose(Xi_actual, np.triu(Xi_actual)), "Expected Xi to be upper triangular"

    # Nu
    nu_actual = p.sqrt_info_vec
    assert nu_actual.shape == (n, 1), f"Expected nu shape (n,1), got {nu_actual.shape}"

    # P
    P_actual = p.cov
    assert P_actual.shape == (n, n), f"Expected P shape (n,n), got {P_actual.shape}"
    np.testing.assert_allclose(P_actual, P_expected, atol=1e-10)

    # S
    S_actual = p.sqrt_cov
    assert S_actual.shape == (n, n), f"Expected S shape (n,n), got {S_actual.shape}"
    assert np.allclose(S_actual, np.triu(S_actual)), "Expected S to be upper triangular"

    # Lambda
    Lambda_actual = p.info_mat
    assert Lambda_actual.shape == (n, n), f"Expected Lambda shape (n,n), got {Lambda_actual.shape}"
    np.testing.assert_allclose(Lambda_actual, Lambda_expected, atol=1e-10)

    # Derived equality checks
    np.testing.assert_allclose(S_actual.T @ S_actual, P_actual, atol=1e-10, err_msg="Expected S.T @ S = P")
    np.testing.assert_allclose(Xi_actual.T @ Xi_actual, Lambda_actual, atol=1e-10, err_msg="Expected Xi.T @ Xi = Lambda")
    np.testing.assert_allclose(Lambda_actual @ mu_actual, eta_actual, atol=1e-10, err_msg="Expected Lambda * mu = eta")
    np.testing.assert_allclose(P_actual @ eta_actual, mu_actual, atol=1e-10, err_msg="Expected P * eta = mu")
    np.testing.assert_allclose(Xi_actual @ mu_actual, nu_actual, atol=1e-10, err_msg="Expected Xi * mu = nu")
    np.testing.assert_allclose(Xi_actual.T @ nu_actual, eta_actual, atol=1e-10, err_msg="Expected Xi.T * nu = eta")


def test_non_zero_mean_full_consistency():
    mu = np.matrix([1,2,3,4,5]).T
    S = np.array([
        [10, 11, 12, 13, 14],
        [ 0, 15, 16, 17, 18],
        [ 0,  0, 19, 20, 21],
        [ 0,  0,  0, 23, 24],
        [ 0,  0,  0,  0, 25]
    ])
    P = S.T @ S
    Xi = np.linalg.qr(np.linalg.solve(S.T, np.eye(5)))[1]
    Lambda = Xi.T @ Xi
    eta = Lambda @ mu
    nu = Xi @ mu

    g = Gaussian(mu, S)
    run_gaussian_case(g, mu, P, eta, Lambda)


def test_zero_mean_full_consistency():
    mu = np.zeros((5,1))
    S = np.array([
        [10, 11, 12, 13, 14],
        [ 0, 15, 16, 17, 18],
        [ 0,  0, 19, 20, 21],
        [ 0,  0,  0, 23, 24],
        [ 0,  0,  0,  0, 25]
    ])
    P = S.T @ S
    Xi = np.linalg.qr(np.linalg.solve(S.T, np.eye(5)))[1]
    Lambda = Xi.T @ Xi
    eta = Lambda @ mu
    nu = Xi @ mu

    g = Gaussian(mu, S)
    run_gaussian_case(g, mu, P, eta, Lambda)


def run_gaussian_cases(mu, S):
    """
    Replicates the MATLAB `runGaussianCases` logic.
    Constructs a Gaussian using all 4 available factories
    and tests consistency using `run_gaussian_case`.
    """
    n = S.shape[0]
    P = S.T @ S
    Xi = np.linalg.qr(np.linalg.solve(S.T, np.eye(n)))[1]  # Xi = upper-triangular QR of S^{-T}
    Lambda = Xi.T @ Xi
    eta = Lambda @ mu
    nu = Xi @ mu

    # Base constructor: from mu, S
    g0 = Gaussian(mu, S)
    run_gaussian_case(g0, mu, P, eta, Lambda)

    # Constructor: from sqrt moment
    # g1 = Gaussian.from_sqrt_moment(mu, S)
    # run_gaussian_case(g1, mu, P, eta, Lambda)

    # Constructor: from full covariance
    g2 = Gaussian.from_moment(mu, P)
    run_gaussian_case(g2, mu, P, eta, Lambda)

    # Constructor: from information form
    # g3 = Gaussian.from_info(eta, Lambda)
    # run_gaussian_case(g3, mu, P, eta, Lambda)

    # Constructor: from sqrt info form
    # g4 = Gaussian.from_sqrt_info(nu, Xi)
    # run_gaussian_case(g4, mu, P, eta, Lambda)


def test_zero_mean_all_constructors():
    mu = np.zeros((5, 1))
    S = np.array([
        [10, 11, 12, 13, 14],
        [ 0, 15, 16, 17, 18],
        [ 0,  0, 19, 20, 21],
        [ 0,  0,  0, 23, 24],
        [ 0,  0,  0,  0, 25]
    ])
    run_gaussian_cases(mu, S)
