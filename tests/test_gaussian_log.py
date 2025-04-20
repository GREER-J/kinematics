import numpy as np
from src.kinematics_library.statistics.guassian import Gaussian


def test_log_pdf_value():
    x = np.array([[1], [2], [3]])
    mu = np.array([[2], [4], [6]])
    S = np.diag([1, 2, 3])
    g = Gaussian.from_sqrt_moment(mu, S)

    expected = -6.04857506884207
    actual = g.log(x)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
