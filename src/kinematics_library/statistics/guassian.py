import numpy as np


class Gaussian:
    def __init__(self, mu: np.ndarray, S: np.ndarray):
        self._mu = mu.reshape(-1, 1)
        self._S = S

    @property
    def mu(self) -> np.ndarray:
        return self._mu

    @property
    def sqrt_cov(self) -> np.ndarray:
        return self._S

    @property
    def cov(self) -> np.ndarray:
        return self._S.T @ self._S

    @property
    def info_mat(self):
        Xi = np.linalg.qr(np.linalg.solve(self._S.T, np.eye(self._S.shape[0])))[1]
        return Xi.T @ Xi

    @property
    def info_vec(self):
        return self.info_mat @ self._mu

    @property
    def sqrt_info_mat(self):
        return np.linalg.qr(np.linalg.solve(self._S.T, np.eye(self._S.shape[0])))[1]

    @property
    def sqrt_info_vec(self):
        return self.sqrt_info_mat @ self._mu

    def dim(self):
        return self._mu.shape[0]

    @classmethod
    def from_moment(cls, mu, P):
        S = np.linalg.cholesky(P).T
        return cls(mu, S)

    @classmethod
    def from_samples(cls, X: np.ndarray) -> "Gaussian":
        mu = np.mean(X, axis=1, keepdims=True)  # (n, 1)
        Xc = X - mu                             # center the samples
        m = X.shape[1]
        P = (1 / (m - 1)) * Xc @ Xc.T           # sample covariance
        S = np.linalg.cholesky(P).T            # upper triangular
        return cls(mu, S)
