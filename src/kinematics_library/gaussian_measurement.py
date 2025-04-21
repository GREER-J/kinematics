from abc import abstractmethod
from typing import Tuple
import numpy as np
from src.kinematics_library.measurement import Measurement
from src.kinematics_library.system import BaseSystem
from src.kinematics_library.guassian import Gaussian


class MeasurementGaussianLikelihood(Measurement):
    def __init__(self, y=None):
        super().__init__(update_method='affine')
        self.y = y  # The actual measurement vector

    @abstractmethod
    def predict_density(self, x: np.ndarray, system) -> Tuple["Gaussian", np.ndarray, np.ndarray]:
        """
        Predict the distribution of y given x.

        Returns:
            py: Gaussian object for predicted measurement distribution
            dhdx: Jacobian matrix (ny × nx)
            d2hdx2: Hessian tensor (ny × nx × nx)
        """

    def simulate(self, x, system):
        """
        Simulate a synthetic measurement from the predicted distribution p(y | x).
        Sets self.y in-place.
        """
        py, *_ = self.predict_density(x, system)
        self.y = py.simulate(m=1)  # shape (ny, 1)
        return self

    def log_likelihood(self, x: np.ndarray, system: BaseSystem, return_gradient=False, return_hessian=False):
        """
        Computes the log-likelihood log p(y | x), along with its gradient and Hessian if requested.

        Args:
            x (np.ndarray): current state (nx × 1)
            system: system object (passed to predict_density)
            return_gradient (bool): whether to return gradient ∇ log p(y | x)
            return_hessian (bool): whether to return Hessian ∇² log p(y | x)

        Returns:
            l (float): log-likelihood
            dldx (optional): gradient of log-likelihood
            d2ldx2 (optional): Hessian of log-likelihood
        """
        if return_hessian:
            likelihood, dhdx, d2hdx2 = self.predict_density(x, system)
            l, dldy, d2ldy2 = likelihood.log(self.y, return_grad=True, return_hess=True)

            dldx = -dhdx.T @ dldy

            """
            % Hessian of log likelihood:
            %
            %              d                                 d  ( dh_k     d                    )
            % H_{ij} = --------- log N(y; h(x), R) = sum_k ---- ( ---- * ---- log N(y; h(x), R) )
            %          dx_i dx_j                           dx_j ( dx_i   dh_k                   )
            %
            %                      dh_k   d^2 log N(y; h(x), R)   dh_l          d^2 h_k      d
            % H_{ij} = sum_k sum_l ---- * --------------------- * ---- + sum_k --------- * ---- log N(y; h(x), R)
            %                      dx_i         dh_k dh_l         dx_j         dx_i dx_j   dh_k
            %
            %                      dh_k   d^2 log N(y; h(x), R)   dh_l          d^2 h_k      d
            % H_{ij} = sum_k sum_l ---- * --------------------- * ---- - sum_k --------- * ---- log N(y; h(x), R)
            %                      dx_i         dy_k dy_l         dx_j         dx_i dx_j   dy_k
            %
            """
            # H = dhdx.T @ d2log(y) @ dhdx - sum_k d2h_k/dx2 * dlogp/dy_k
            nh = self.y.shape[0]
            nx = x.shape[0]
            contraction = np.sum(d2hdx2 * dldy.reshape((nh, 1, 1)), axis=0)
            d2ldx2 = dhdx.T @ d2ldy2 @ dhdx - contraction

            return float(l), dldx, d2ldx2

        elif return_gradient:
            likelihood, dhdx = self.predict_density(x, system)
            l, dldy = likelihood.log(self.y, return_grad=True)

            """
            % Gradient of log likelihood:
            %
            %         d
            % g_i = ---- log N(y; h(x), R)
            %       dx_i
            %
            %             dh_k     d
            % g_i = sum_k ---- * ---- log N(y; h(x), R)
            %             dx_i   dh_k
            %
            %               dh_k     d
            % g_i = - sum_k ---- * ---- log N(y; h(x), R)
            %               dx_i   dy_k
            %
            """
            dldx = -dhdx.T @ dldy
            return float(l), dldx

        else:

            likelihood = self.predict_density(x, system)  # By default this is just a Gaussian
            l = likelihood.log(self.y)
            return float(l)

    def update(self, system):
        """
        Update the system's Gaussian density with this measurement.
        Supports multiple update methods: affine, unscented, etc.
        """
        if self.need_to_simulate:
            self.simulate(system.x_sim, system)
            self.need_to_simulate = False

        method = self.update_method.lower()
        if method in ['affine', 'unscented']:
            nx = system.density.dim()
            ny = self.y.shape[0]

            # Call predict_density with only what's needed
            def joint_func(x):
                py_aug, J_aug = self.augmented_predict_density(
                    x, system,
                    return_gradient=(method == 'affine'),
                    return_hessian=(method == 'unscented')
                )
                return py_aug, J_aug

            if method == 'affine':
                pyx = system.density.affine_transform(joint_func)

            elif method == 'unscented':
                pyx = system.density.unscented_transform(joint_func)

            idx_x = list(range(ny, ny + nx))
            idx_y = list(range(ny))

            system.density = pyx.conditional(idx_x, idx_y, self.y)

        else:
            return super().update(system)

        return self, system

    def augmented_predict_density(self, x, system, return_gradient=False, return_hessian=False):
        if return_hessian:
            py, dhdx, d2hdx2 = self.predict_density(x, system, return_gradient=True, return_hessian=True)
        elif return_gradient:
            py, dhdx = self.predict_density(x, system, return_gradient=True)
            d2hdx2 = np.zeros((py.mu.shape[0], x.shape[0], x.shape[0]))
        else:
            py = self.predict_density(x, system)
            dhdx = np.zeros((py.mu.shape[0], x.shape[0]))
            d2hdx2 = np.zeros((py.mu.shape[0], x.shape[0], x.shape[0]))

        ny = py.mu.shape[0]
        nx = x.shape[0]

        # Construct full S_aug: (ny+nx, ny+nx)
        S_aug = np.zeros((ny + nx, ny + nx))
        S_aug[:ny, :ny] = py.sqrt_cov  # top-left block is R
        S_aug[ny:, ny:] = system.density.sqrt_cov  # bottom-right = state uncertainty


        mu_aug = np.vstack([py.mu, x])
        J_aug = np.vstack([dhdx, np.eye(nx)])

        py_aug = Gaussian(mu_aug, S_aug)

        return py_aug, J_aug
