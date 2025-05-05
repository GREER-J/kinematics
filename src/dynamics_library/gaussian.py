from __future__ import annotations
from typing import Callable, Union, Optional
import numpy as np
from scipy.stats import chi2, norm
from src.kinematics_library.gaussian_measurement_function import MeasurementFunctionProtocol
from src.kinematics_library.gaussian_return import GaussianReturn
from enum import Enum


class AffineMode(Enum):
    """Mode to complete transform with

    Args:
        Enum (String): Transform method
    """
    MOMENT = 'moment'
    SQRT = 'sqrt'


class Gaussian:
    """This is our core math representation of uncertainty."""
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
        """Create a Gaussian object from mean and covariance.

        Args:
            mu (np.ndarray): Mean of the distribution.
            P (np.ndarray): Covariance of the distribution.

        Returns:
            Gaussian: Gaussian object capturing the distribution.
        """
        S = np.linalg.cholesky(P).T
        return cls(mu, S)

    @classmethod
    def from_samples(cls, X: np.ndarray) -> Gaussian:
        """
        Construct a Gaussian object from samples from a distribution.

        Args:
            X (np.ndarray): samples from the target distribution

        Returns:
            Gaussian: Gaussian object with mean `mu` and sqrt covariance `S`
        """
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

    def log(self, X: np.ndarray, return_grad: bool = False, return_hess: bool = False) -> GaussianReturn:
        """
        Evaluate the log of the Gaussian PDF at one or more input points.

        Args:
            X (np.ndarray): Input of shape (n, m)
            return_grad (bool): Whether to return ∇ log p
            return_hess (bool): Whether to return ∇² log p

        Returns:
            logPDF (np.ndarray): shape (m,)
            grad (np.ndarray): shape (n, m), optional
            hess (np.ndarray): shape (n, n, m), optional
        """
        mu = self.mu
        S = self.sqrt_cov
        n = self.dim()

        # Ensure (n, m) shape
        X = np.atleast_2d(X)
        if X.shape[0] != n:
            if X.shape[1] == n:
                X = X.T
            else:
                raise ValueError(f"Expected input shape (n, m), got {X.shape}")
        m = X.shape[1]

        X_mu = X - mu
        logPDF = np.zeros((m,))
        ST_inv = np.linalg.inv(S.T)

        for i in range(m):
            w = ST_inv @ X_mu[:, i]
            log_det_term = np.sum(np.log(np.abs(np.diag(S))))
            logPDF[i] = -0.5 * n * np.log(2 * np.pi) - log_det_term - 0.5 * np.dot(w, w)

        results = GaussianReturn(magnitude=logPDF)  # results = (logPDF,)

        if return_grad:
            Z = np.linalg.solve(S, X_mu)
            grad = -np.linalg.solve(S, Z)  # shape (n, m)
            results.grad_magnitude = grad  #results += (grad,)

        if return_hess:
            Lambda = self.info_mat  # (n, n)
            H = -Lambda[:, :, None].repeat(m, axis=2)  # (n, n, m)
            results.hess_magnitude = H  # results += (H,)

        return results

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
        h: MeasurementFunctionProtocol,
        noise: Optional[Gaussian] = None,
        mode=AffineMode.SQRT
    ) -> Gaussian:
        """
        Apply an affine (linearized) transformation to the Gaussian using Jacobian-based uncertainty propagation.

        Args:
            h: A function h(x) that returns a GaussianReturn object.
            noise: Optional additive Gaussian noise in the output space.

        Returns:
            Gaussian: Transformed approximation.
        """        
        mu = self.mu

        # 1. Pull out required information
        result: GaussianReturn = h(x=mu, return_grad=True)

        # --- CASE 1: Full Gaussian provided
        if result.has_gaussian:
            if noise is not None:
                raise ValueError("Noise should not be provided when h() already returns a Gaussian")
            return result.gaussian_magnitude

        # --- CASE 2: Linearized Transform
        if not result.has_grad:
            raise ValueError("Affine transform requires gradient information (Jacobian) when no Gaussian is provided")

        # 2. Call out to sub func for Gaussian
        if mode == AffineMode.MOMENT:
            J = result.grad_magnitude
            Px = self.cov
            Py = self.compute_affine_transform_moment(J, Px, noise)
            return self.__class__.from_moment(result.magnitude, Py)
        if mode == AffineMode.SQRT:
            y_part = result.magnitude
            S = self.sqrt_cov
            J = result.grad_magnitude
            return self.compute_affine_transform_sqrt(noise, S, y_part, J)

        raise ValueError(f"Incorrect mode, received: {mode}")

    def compute_affine_transform_moment(self, J: np.ndarray, Px: np.ndarray, noise: Gaussian) -> Gaussian:
        """
        Full moment calculation for [h(x); x].
        """
        R = noise.cov if noise is not None else np.zeros((J.shape[0], J.shape[0]))

        assert Px.shape[0] == J.shape[1], "Incompatible dimensions for affine transform"
        P_aug = J @ Px @ J.T + R

        return P_aug

    def compute_affine_transform_sqrt(self, noise, S, y_part, J) -> Gaussian:
        if y_part.ndim == 1:
            y_part = y_part[:, None]

        # Propagate uncertainty
        # SJ_T = J @ S.T
        SJ_T = S @ J.T
        parts = [SJ_T]

        if noise is not None:
            parts.append(noise.sqrt_cov)

        # Stack all uncertainty contributions vertically
        max_cols = max(p.shape[1] for p in parts)
        padded_parts = [
            np.pad(p, ((0, 0), (0, max_cols - p.shape[1]))) if p.shape[1] < max_cols else p
            for p in parts
        ]
        stacked = np.vstack(padded_parts)

        # QR decomposition to obtain square-root of covariance
        _, R = np.linalg.qr(stacked, mode="reduced")
        SR = R.T

        # Ensure full (n, n) shape
        expected_dim = J.shape[0]
        curr_rows, curr_cols = SR.shape
        if curr_rows < expected_dim:
            pad_rows = expected_dim - curr_rows
            SR = np.pad(SR, ((0, pad_rows), (0, 0)))
        if curr_cols < expected_dim:
            pad_cols = expected_dim - curr_cols
            SR = np.pad(SR, ((0, 0), (0, pad_cols)))

        return self.__class__(y_part, SR)

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

    def conditional(self, idx_A, idx_B, x_B):
        """
        Compute the conditional distribution p(x_A | x_B = xB)
        from the joint Gaussian.

        Args:
            idx_A (List[int]): Indices of target variables A
            idx_B (List[int]): Indices of known variables B
            x_B (np.ndarray): Observed value of x_B (shape: (len(B), 1))

        Returns:
            Gaussian: The conditional distribution p(x_A | x_B = xB)
        """
        # Step 1: Reorder and decompose S into [S_B, S_A]
        S = self.sqrt_cov

        S_joint = np.hstack([S[:, idx_B], S[:, idx_A]])  # shape (n, len(B)+len(A))
        Q, R = np.linalg.qr(S_joint, mode="reduced")
        SS = np.triu(R)

        nB = len(idx_B)
        # nA = len(idx_A)  # not strictly needed

        print(f"[CONDITIONAL] mu before: {self.mu.T}")
        print(f"[CONDITIONAL] xB = {x_B.flatten()}")

        SBB = SS[0:nB, 0:nB]                     # (nB, nB)
        SBA = SS[0:nB, nB:]                      # (nB, nA)
        SAA = SS[nB:, nB:]                       # (nA, nA)

        muA = self.mu[idx_A]
        muB = self.mu[idx_B]

        # Solve: delta = SBB \ (xB - muB)
        delta = np.linalg.solve(SBB.T, x_B - muB)  # forward solve (triangular)
        muc = muA + SBA.T @ delta                  # updated mean
        Sc = SAA                                   # updated sqrt covariance

        print(f"[CONDITIONAL] mu after: {muc.T}")
        print(f"[CONDITIONAL] sqrt_cov:\n{self.sqrt_cov}")

        return Gaussian(muc, Sc)
