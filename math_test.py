import numpy as np
from src.dynamics_library.gaussian import Gaussian

x0 = np.zeros((3,1))
x0[0,0] = 2.24411562

Pp = 5 * np.eye(3)
C = np.array([[1, 1, 0]])

meas = C @ x0
# => [[2.24411562]]

py_x = C @ Pp @ C.T + 1
# => 5 + 5 + 1 = 11

aug_mu = np.vstack([meas, x0])
print("Augmented Mean:\n", aug_mu)

S_yy = py_x
S_yx = C @ Pp  # shape (1,3)
S_xy = S_yx.T  # shape (3,1)
S_xx = Pp      # shape (3,3)

# Assemble full covariance
top = np.hstack([S_yy, S_yx])
bottom = np.hstack([S_xy, S_xx])
aug_cov = np.vstack([top, bottom])

print("Augmented Covariance:\n", aug_cov)
