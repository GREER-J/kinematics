import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from kinematics_library import homogeneous as htf
from kinematics_library.homogeneous_utils import get_basis_vectors, get_r_and_R_from_A 
print("Hello world")

# Camera 1
cam_1_east = 100  # [m]
cam_1_north = 200  # [m]
cam_1_compass_bearing = np.deg2rad(45)  # [deg]

A1n = htf.transx(cam_1_north) @ htf.transy(cam_1_east) @ htf.rotz(cam_1_compass_bearing)  # NED
rC1Nn, Rc1n = get_r_and_R_from_A(A1n)
c1_1, c1_2, c1_3 = get_basis_vectors(A1n)


# Camera 2
cam_2_east = 400  # [m]
cam_2_north = 500  # [m]
cam_2_compass_bearing = np.deg2rad(0)  # [deg]

A2n = htf.transx(cam_2_north) @ htf.transy(cam_2_east) @ htf.rotz(cam_2_compass_bearing)  # NED
rC2Nn, Rc2n = get_r_and_R_from_A(A2n)
c2_1, c2_2, c2_3 = get_basis_vectors(A2n)

# Target

target_E = 1000


class CameraNetwork:
    def __init__(self):
        pass

def get_dir_vec_from_cam(vec3d, target_E, target_N):
    dz = np.array([[target_N], [target_E]]) - vec3d[0:2]
    dir_vec = dz / np.linalg.norm(dz)
    return dir_vec


def get_homogeneous_line_from_point_and_direction(point, direction):
    """
    Constructs a homogeneous line from a point and a direction vector.
    """
    # Convert the point to homogeneous coordinates
    point_h = np.array([point[0, 0], point[1, 0], 1.0])
    
    # Convert the direction vector to a point at infinity in homogeneous coordinates
    direction_h = np.array([direction[0, 0], direction[1, 0], 0.0])
    
    # The line is the cross product of the point and the direction
    line = np.cross(point_h, direction_h)
    return line

# Example usage
es = []
ns = []

for k, target_N in enumerate(np.linspace(0, 500, 100)):
    # Assume rC1Nn and rC2Nn are 3x1 numpy arrays representing camera positions
    y_c1 = get_dir_vec_from_cam(rC1Nn, target_E, target_N)
    y_c2 = get_dir_vec_from_cam(rC2Nn, target_E, target_N)

    # Construct homogeneous lines from camera positions and direction vectors
    h_c1 = get_homogeneous_line_from_point_and_direction(rC1Nn, y_c1)
    h_c2 = get_homogeneous_line_from_point_and_direction(rC2Nn, y_c2)

    # Calculate intersection point with the cross product
    p = np.cross(h_c1, h_c2)

    # Normalize if the third coordinate is not zero
    if p[2] != 0:
        p = p / p[2]

    es.append(p[1])
    ns.append(p[0])



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

ax.plot(es, ns, 'r--', label="Projected position?")

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

