from typing import Protocol
import numpy as np


class MeasurementFunctionProtocol(Protocol):
    def __call__(self, x: np.ndarray,
                 return_grad: bool = False,
                 return_hess: bool = False
                 ) -> "GaussianReturn":
        """
        A function that maps an input Gaussian mean (n,1) to a GaussianReturn:
          - magnitude: (m,1) array (output mean)
          - optionally grad_magnitude: Jacobian (m,n)
          - optionally hess_magnitude: Hessian (not usually needed for affine)
        """
