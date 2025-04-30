import numpy as np

# State mean
mux = np.array([
    [1.0],
    [2.0]
])

# State covariance
Px = np.diag([1.0, 4.0])

# # Measurement matrix H (Jacobian of h(x))
# H = np.array([[2.0, 3.0]])

# Measurement noise covariance
R = np.array([[0.5]])

# Augmented Jacobian
Ja = np.array([
    [2.0, 3.0],
    [1.0, 0.0],
    [0.0, 1.0]
])

# Predicted augmented mean
muy_aug = Ja @ mux

print("\nAugmented mean (muy_aug):")
print(muy_aug)

# Build augmented noise matrix
R_aug = np.zeros((3,3))
R_aug[0,0] = R  # Only measurement noise affects h(x)

# Augmented covariance
P_aug = Ja @ Px @ Ja.T + R_aug

print("\nAugmented covariance (P_aug):")
print(P_aug)
