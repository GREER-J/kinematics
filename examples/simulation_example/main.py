import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from itertools import combinations
from dataclasses import dataclass
from kinematics_library import homogeneous as htf
from kinematics_library.homogeneous_utils import get_basis_vectors, get_r_and_R_from_A, get_r_from_A

from src.camera_sensor import CameraSensor, CameraDirectionMeasurement
from src.camera_network import CameraNetwork


# Camera 1
cam_1 = CameraSensor(1, 100, 200, 45)
rC1Nn, Rc1n = get_r_and_R_from_A(cam_1.Apn)
c1_1, c1_2, c1_3 = get_basis_vectors(cam_1.Apn)

# Camera 2
cam_2 = CameraSensor(2, 400, 500, 0)
rC2Nn, Rc2n = get_r_and_R_from_A(cam_2.Apn)
c2_1, c2_2, c2_3 = get_basis_vectors(cam_2.Apn)

# Network
net = CameraNetwork([cam_1, cam_2])

# Target
target_E = 1000

zs = []

for k, target_N in enumerate(np.linspace(0, 500, 100)):
    rTNn = np.matrix([[target_N], [target_E], [200]])
    z = net.get_z(rTNn)

    zs.append(z)

    if k > 1:
        break


# Create a 2D plot
fig, ax = plt.subplots(figsize=(6, 6))

# Define the length of the basis vectors for visualization
vector_scale = 0.02  # Adjust as needed

    # Camera 2
ax.plot(rC2Nn[1], rC2Nn[0], 'o', label='Camera Position')

ax.quiver(
    rC2Nn[1], rC2Nn[0],           # Origin of the vector
    c2_2[1], c2_2[0],      # East and North components
    angles='xy',
    scale_units='xy',
    scale=vector_scale,
    color='blue',
    label='North (e1)'
)

ax.quiver(
    rC2Nn[1], rC2Nn[0],
    c2_1[1], c2_1[0],
    angles='xy',
    scale_units='xy',
    scale=vector_scale,
    color='green',
    label='East (e2)'
)

    # Camera 1
ax.plot(rC1Nn[1], rC1Nn[0], 'o', label='Camera 1 Position')

ax.quiver(
    rC1Nn[1], rC1Nn[0],           # Origin of the vector
    c1_2[1], c1_2[0],      # East and North components
    angles='xy',
    scale_units='xy',
    scale=vector_scale,
    color='blue',
    label='North (e1)'
)

ax.quiver(
    rC1Nn[1], rC1Nn[0],
    c1_1[1], c1_1[0],
    angles='xy',
    scale_units='xy',
    scale=vector_scale,
    color='green',
    label='East (e2)'
)

    # Target
ax.plot(1000 * np.ones((100, 1)), np.linspace(0, 500, 100), '--', label='Target Position')

for z in zs:
    ns = []
    es = []
    for n in z:
        ns.append(float(n[0]))
        es.append(float(n[1]))

    ax.scatter(es, ns)

ax.set_xlim(-500, 1500)
ax.set_ylim(-500, 1500)

# Set labels and title
ax.set_xlabel('East [m]')
ax.set_ylabel('North [m]')
ax.set_title('Camera Position and Orientation on NE Plane')

# Set equal aspect ratio
ax.set_aspect('equal')

# Add grid and legend
ax.grid(True)
ax.legend()

# Show the plot
plt.show()
