import numpy as np
from scipy.stats import chi2, norm
from typing import Callable, Union, Tuple


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

    @classmethod
    def from_sqrt_moment(cls, mu: np.ndarray, S: np.ndarray) -> "Gaussian":
        """
        Construct a Gaussian object from a mean vector and square-root covariance matrix.

        Args:
            mu (np.ndarray): Mean vector of shape (n, 1)
            S (np.ndarray): Upper-triangular square-root covariance of shape (n, n)

        Returns:
            Gaussian: Gaussian object with mean `mu` and sqrt covariance `S`
        """
        mu = np.atleast_2d(mu)
        if mu.shape[1] != 1:
            raise ValueError(f"mu should be a column vector of shape (n, 1), but got shape {mu.shape}")

        n = mu.shape[0]
        if S.shape != (n, n):
            raise ValueError(f"S must be square with shape ({n}, {n}), but got shape {S.shape}")

        if not np.allclose(S, np.triu(S)):
            raise ValueError("S is expected to be upper triangular")

        return cls(mu, S)

    def simulate(self, m: int = 1) -> np.ndarray:
        """
        Draw `m` samples from the Gaussian distribution.

        Returns:
            np.ndarray: Samples of shape (n, m), where each column is a sample.
        """
        n = self.dim()
        W = np.random.randn(n, m)
        return self.sqrt_cov.T @ W + self.mu  # Equivalent to S' * W + mu

    def log(self, X: np.ndarray, return_grad: bool = False, return_hess: bool = False):
        """
        Evaluate the log of the Gaussian PDF at one or more input points.

        Args:
            X (np.ndarray): Input points of shape (n,) or (n, m)
            return_grad (bool): If True, also return the gradient(s)
            return_hess (bool): If True, also return the Hessian(s)

        Returns:
            logPDF: shape (m,)
            grad: shape (n, m), optional
            hess: shape (n, n, m), optional
        """
        mu = self.mu
        S = self.sqrt_cov
        n = self.dim()

        # Ensure 2D input
        X = np.atleast_2d(X)
        if X.shape[0] != n:
            if X.shape[1] == n and X.shape[0] != n:
                X = X.T
            else:
                raise ValueError(f"Expected input shape (n, m), got {X.shape}")
        m = X.shape[1]

        X_mu = X - mu
        logPDF = np.zeros((m,))

        try:
            ST_inv = np.linalg.inv(S.T)
        except np.linalg.LinAlgError:
            raise ValueError("S.T is not invertible")

        for i in range(m):
            w = ST_inv @ X_mu[:, i]
            log_det_term = np.sum(np.log(np.abs(np.diag(S))))
            logPDF[i] = -0.5 * n * np.log(2 * np.pi) - log_det_term - 0.5 * np.dot(w, w)

        result = (logPDF,)

        if return_grad:
            Z = np.linalg.solve(S, X_mu)
            grad = -np.linalg.solve(S, Z)  # shape (n, m)
            result += (grad,)

        if return_hess:
            Lambda = self.info_mat  # shape (n, n)
            H = np.repeat(Lambda[:, :, None], m, axis=2) * -1
            result += (H,)

        return result if len(result) > 1 else result[0]

    def eval(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the probability density function (PDF) of the Gaussian at one or more input points.

        Args:
            X (np.ndarray): Points of shape (n,) or (n, m)

        Returns:
            np.ndarray: PDF values of shape (m,)
        """
        log_pdf = self.log(X)
        return np.exp(log_pdf)

    def affine_transform(
        self,
        h: Callable[[np.ndarray], Union["Gaussian", Tuple[np.ndarray, np.ndarray]]],
        noise: "Gaussian" = None
    ) -> "Gaussian":
        """
        Apply an affine (linearized) transformation to the Gaussian using Jacobian-based uncertainty propagation.

        Args:
            h: A function h(x) that returns either:
                - A Gaussian object with mean and sqrt_cov
                - A tuple (output, J) where output is a mean vector and J is the Jacobian
            noise: Optional additive Gaussian noise in the output space

        Returns:
            Gaussian: Transformed approximation
        """
        mu = self.mu
        S = self.sqrt_cov

        result = h(mu)

        if isinstance(result, Gaussian):
            # Function returns a Gaussian directly (with its own model noise)
            if noise is not None:
                raise ValueError("Noise should not be provided when h() already returns a Gaussian")
            y_mu = result.mu
            SR = result.sqrt_cov

        elif isinstance(result, tuple):
            # Function returns (output, Jacobian)
            y_mu, J = result

            y_mu = np.atleast_2d(y_mu)
            if y_mu.shape[1] != 1:
                y_mu = y_mu.T

            SJ_T = S @ J.T
            parts = [SJ_T]

            if noise is not None:
                y_mu = y_mu + noise.mu
                parts.append(noise.sqrt_cov)

            stacked = np.vstack(parts)
            SR = np.linalg.qr(stacked, mode="reduced")[1].T  # upper-triangular

        else:
            raise ValueError("Expected h(mu) to return either a Gaussian or (output, Jacobian) tuple")

        # Truncate to output dimensionality
        SR = SR[:y_mu.shape[0], :]

        return self.__class__(y_mu, SR)


    def unscented_transform(
        self,
        h: Callable[[np.ndarray], Union[np.ndarray, "Gaussian"]],
        noise: "Gaussian" = None
        ) -> "Gaussian":
        """
        Propagate this Gaussian through a nonlinear function using the Unscented Transform (UT).

        Args:
            h: Function h(x) -> y or Gaussian. Should accept (n,) or (n,1) vectors.
            noise: Optional additive Gaussian noise in the output space.

        Returns:
            Gaussian: Approximated distribution of y = h(x)
        """
        mu = self.mu
        S = self.sqrt_cov
        n = self.dim()
        nsigma = 2 * n + 1

        # UT parameters
        alpha = 1.0
        kappa = 0.0
        beta = 2.0
        lambda_ = alpha**2 * (n + kappa) - n
        gamma = np.sqrt(n + lambda_)

        # Generate sigma points
        gamma_ST = gamma * S.T  # shape (n, n)
        sigma_points = np.hstack([
            mu + gamma_ST,
            mu - gamma_ST,
            mu
        ])  # shape (n, 2n+1)

        # Evaluate function at each sigma point
        Y = []
        SR = None
        implicit_noise = False

        for i in range(nsigma):
            result = h(sigma_points[:, i].reshape(-1, 1))

            if isinstance(result, Gaussian):
                if noise is not None:
                    raise ValueError("Noise should not be provided when function h returns a Gaussian")
                if SR is None:
                    SR = result.sqrt_cov
                Y.append(result.mu.flatten())
                implicit_noise = True
            else:
                Y.append(np.atleast_1d(result).flatten())
                if SR is None:
                    SR = np.zeros((0, len(Y[0])))  # No noise

        Y = np.array(Y).T  # shape (ny, nsigma)
        ny = Y.shape[0]

        # Unscented weights
        wm = np.full(2 * n, 1 / (2 * (n + lambda_)))
        wc = np.full(2 * n, 1 / (2 * (n + lambda_)))
        wm = np.append(wm, lambda_ / (n + lambda_))
        wc = np.append(wc, lambda_ / (n + lambda_) + (1 - alpha**2 + beta))

        # Mean of transformed points
        mu_y = Y @ wm

        # Covariance
        dY = (Y - mu_y[:, None]) * np.sqrt(wc)
        stacked = np.vstack([dY.T, SR]) if noise is None else np.vstack([dY.T, SR, noise.sqrt_cov])
        Syy = np.linalg.qr(stacked, mode="reduced")[1].T

        if noise is not None:
            mu_y = mu_y + noise.mu.flatten()

        return self.__class__(mu_y.reshape(-1, 1), Syy)

    def confidence_ellipse(self, n_sigma: float = 3, n_samples: int = 100) -> np.ndarray:
        """
        Generate a 2D confidence ellipse for a bivariate Gaussian.

        Args:
            n_sigma (float): Number of standard deviations (default: 3)
            n_samples (int): Number of points to draw along the ellipse

        Returns:
            np.ndarray: Array of shape (2, n_samples) representing ellipse points
        """
        if self.dim() != 2:
            raise ValueError("Confidence ellipse is only defined for 2D Gaussians")

        # Calculate probability mass enclosed by ±n_sigma in 1D
        c = norm.cdf(n_sigma) - norm.cdf(-n_sigma)

        # Mahalanobis radius for enclosing that probability mass in 2D
        r = np.sqrt(chi2.ppf(c, df=2))

        # Points around the unit circle scaled by r
        t = np.linspace(0, 2 * np.pi, n_samples)
        W = r * np.vstack([np.cos(t), np.sin(t)])  # Shape (2, n_samples)

        # Transform circle into ellipse using sqrt covariance
        ellipse = self.sqrt_cov.T @ W + self.mu  # Shape (2, n_samples)
        return ellipse
