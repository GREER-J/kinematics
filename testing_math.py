import numpy as np

# --- Inputs ---

# State mean (2x1)
mux = np.array([
    [1.0],
    [2.0]
])

# State covariance (2x2)
Px = np.diag([1.0, 4.0])

# Measurement matrix H (1x2)
H = np.array([[2.0, 3.0]])

# Measurement noise covariance R (1x1)
R = np.array([[0.5]])


# --- Step 1: Moment-based prediction (for full covariance reference) ---
def moment_from(mux, H, Px, R):
    """
    Full moment calculation for [h(x); x].
    """
    muy = H @ mux  # Predicted measurement mean
    Py = H @ Px @ H.T + R  # Predicted measurement covariance
    Pyx = H @ Px  # Cross-covariance
    mu_aug = np.vstack([muy, mux])  # Augmented mean
    P_aug = np.block([
        [Py, Pyx],
        [Pyx.T, Px]
    ])  # Augmented covariance
    return muy, Py, Pyx, mu_aug, P_aug

# Compute using moment form
muy, Py, Pyx, mu_aug, P_aug = moment_from(mux, H, Px, R)

muy_expected = np.array([[8]])
Py_expected = np.array([[40.5]])
Pyx_expected = np.array([[2.0, 12.0]])
mu_aug_expected = np.array([[8.0], [1.0], [2.0]])

P_aug_expected = np.array([
    [40.5, 2.0, 12.0],
    [2.0, 1.0, 0.0],
    [12.0, 0.0, 4.0]
])

np.testing.assert_allclose(muy, muy_expected, atol=1e-10)
np.testing.assert_allclose(Py, Py_expected, atol=1e-10)
np.testing.assert_allclose(Pyx, Pyx_expected, atol=1e-10)
np.testing.assert_allclose(mu_aug, mu_aug_expected, atol=1e-10)
np.testing.assert_allclose(P_aug, P_aug_expected, atol=1e-10)

print("WINNER: Moment form confirmed!")

# Compute sqrt of Px and R
Sx = np.linalg.cholesky(Px).T  # (2x2)
np.testing.assert_allclose(Sx@Sx.T, Px, atol=1e-10)

SR = np.linalg.cholesky(R).T   # (1x1)
np.testing.assert_allclose(SR@SR.T, R, atol=1e-10)


# Build augmented Jacobian
# J_aug = [ d(h(x))/dx ; Identity ]
J_aug = np.vstack([
    H,            # (1x2)
    np.eye(2)     # (2x2)
])  # (3x2)

dhdx_expected = np.array([[2.0, 3.0],
                          [1.0, 0.0],
                          [0.0, 1.0]
                        ])
np.testing.assert_allclose(J_aug, dhdx_expected, atol=1e-10)

# Propagate state uncertainty
# --- Key: MATLAB does (obj.S * J.') meaning Sx @ J_aug.T
propagated_state = Sx @ J_aug.T  # (2x3)

np.testing.assert_allclose(propagated_state.T @ propagated_state, J_aug @ Px @ J_aug.T, atol=1e-10)

print("Propagated state works")

SR_padded = np.hstack([
    SR,                 # real measurement noise
    np.zeros((1, 2))     # zeros for the state parts
])  # (1,3)



# Transpose propagated uncertainty to (outputs x uncertainties)
propagated_state = propagated_state.T  # (3x2)
SR_padded = SR_padded.T

# print("Propagated state shape: ")
# print(propagated_state.shape)

# print("SR padded shape: ")
# print(SR_padded.shape)

# Stack vertically: [ propagated_state ; noise ]
stacked = np.hstack([
    propagated_state,  # (3x2)
    SR_padded          # (3x1)
])

P_aug_reconstructed = stacked @ stacked.T
print("P_aug_reconstructed: ")
print(P_aug_reconstructed)
np.testing.assert_allclose(P_aug_reconstructed, P_aug_expected, atol=1e-8)

# QR decomposition
Q, Rqr = np.linalg.qr(stacked.T, mode='reduced')

# Upper-triangular sqrt of augmented covariance
S_aug = Rqr.T

print("\nUpper-triangular S_aug (after QR):")
print(S_aug)

# Expand back to covariance
P_aug_from_sqrt = S_aug.T @ S_aug

print("\nExpanded covariance (P_aug_from_sqrt = S_aug @ S_aug.T):")
print(P_aug_from_sqrt)


# --- Final comparison ---
print("\nDifference (Expanded - Expected):")
print(P_aug_from_sqrt - P_aug_expected)

# Assert match
assert np.allclose(P_aug_from_sqrt, P_aug_expected, atol=1e-8), "Mismatch between sqrt-expanded and moment-expanded covariance!"

print("\n✅ SUCCESS: Square-root affine transform matches full moment calculation.")
