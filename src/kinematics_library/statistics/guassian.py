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
