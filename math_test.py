import numpy as np
from src.dynamics_library.gaussian import Gaussian, GaussianReturn, AffineMode

def compare_matrices(name: str, A: np.ndarray, B: np.ndarray, atol=1e-10, rtol=1e-10):
    abs_diff = np.abs(A - B)
    rel_diff = np.abs(abs_diff / (np.abs(B) + atol))

    max_abs = np.max(abs_diff)
    max_rel = np.max(rel_diff)

    if not np.allclose(A, B, atol=atol, rtol=rtol):
        print(f"\n❌ Matrix comparison failed: {name}")
        print("Absolute difference:\n", abs_diff)
        print("Relative difference:\n", rel_diff)
        print(f"Max absolute difference: {max_abs:.3e}")
        print(f"Max relative difference: {max_rel:.3e}")
        print(f"Expected (B):\n{B}")
        print(f"Actual   (A):\n{A}")
        raise AssertionError(f"Matrix '{name}' does not match within tolerance.")
    else:
        print(f"✅ Matrix '{name}' passed (max abs diff: {max_abs:.2e}, max rel diff: {max_rel:.2e})")


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

print("Prior cov:\n", Pp)

S_yy = py_x
S_yx = C @ Pp  # shape (1,3)
S_xy = S_yx.T  # shape (3,1)
S_xx = Pp      # shape (3,3)

# Assemble full covariance
top = np.hstack([S_yy, S_yx])
bottom = np.hstack([S_xy, S_xx])
aug_cov = np.vstack([top, bottom])

print("Augmented Covariance:\n", aug_cov)

# eigvals, eigvecs = np.linalg.eigh(aug_cov)
# L = eigvecs @ np.diag(np.sqrt(np.clip(eigvals, 1e-12, None)))
# _, R = np.linalg.qr(L)
# S = R.T

eigvals, eigvecs = np.linalg.eigh(aug_cov)
D_sqrt = np.diag(np.sqrt(np.clip(eigvals, 1e-12, None)))
S = eigvecs @ D_sqrt

print("Recovered cov:\n", S@S.T)

assert np.allclose(S @ S.T, aug_cov, atol=1e-10)


assert np.allclose(S @ S.T, aug_cov, atol=1e-8), "2 doesn't work"

x = Gaussian.from_moment(x0, Pp)
C = np.array([[1, 1, 0]])
I = np.eye(3)


def h_augmented(x, return_grad):
    rv_mag = np.vstack([C @ x, x])
    rv_grad = np.vstack([C, I])
    return GaussianReturn(magnitude=rv_mag, grad_magnitude=rv_grad)

joint = x.affine_transform(h_augmented, mode=AffineMode.SQRT)

print("Affine Mean:\n", joint.mu)
print("Affine Cov:\n", joint.cov)

compare_matrices("Mean", joint.mu, aug_mu)
compare_matrices("Covariance", joint.cov, aug_cov)
