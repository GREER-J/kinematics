import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from itertools import combinations
from dataclasses import dataclass
from kinematics_library import homogeneous as htf
from kinematics_library.homogeneous_utils import get_basis_vectors, get_r_and_R_from_A, get_r_from_A 


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


@dataclass
class CameraDirectionMeasurement:
    base_point: np.ndarray
    dir_vec: np.ndarray


class CameraSensor:
    def __init__(self, E: float, N: float, bearing_deg: float):
        self.Apn = htf.transx(N) @ htf.transy(E) @ htf.rotz(np.deg2rad(bearing_deg))  # NED
        self.false_target_per_timestep = 10

    def get_dir_vec(self, rTNn: np.ndarray) -> np.ndarray:
        rPNn = get_r_from_A(self.Apn)
        dr = rTNn - rPNn
        return dr / norm(dr)

    def get_false_targets(self, N: int):
        false_targets = []
        for _ in range(N):
            A = self.Apn@htf.roty(np.deg2rad(np.random.uniform(-45, 45))) @ htf.rotz(np.deg2rad(np.random.uniform(-45, 45))) @ htf.transx(10)
            r = get_r_from_A(A)
            false_targets.append(r)
        return false_targets

    def get_detections(self, rTNn) -> list[CameraDirectionMeasurement]:
        measurements = []

        rPNn = get_r_from_A(self.Apn)
        measurements.append(CameraDirectionMeasurement(rPNn, self.get_dir_vec(rTNn)))

        for false_target in self.get_false_targets(self.false_target_per_timestep):
            measurements.append(CameraDirectionMeasurement(rPNn, self.get_dir_vec(false_target)))
        return measurements
    
    def is_point_in_vis(self, r_vec) -> bool:
        # Rotate into maximal FOV limits and check e2 to see if it's positive or negative
        # Or just take angle via dot product and directly check FOV limits
        pass


class CameraNetwork:
    def __init__(self, cameras: list[CameraSensor]):
        self._cams = cameras

    def get_y(self, rTNn) -> list[CameraDirectionMeasurement]:
        y = []
        for cam in self._cams:
            y.extend(cam.get_detections(rTNn))
        return y

    def process_y(self, y: list[CameraDirectionMeasurement]):
        intersection_points = []
        for k, val in enumerate(combinations(y, 2)):
            y1, y2 = val
            # Construct homogeneous lines from camera positions and direction vectors
            h_c1 = get_homogeneous_line_from_point_and_direction(y1.base_point, y1.dir_vec)
            h_c2 = get_homogeneous_line_from_point_and_direction(y2.base_point, y2.dir_vec)

            # Calculate intersection point with the cross product
            p = np.cross(h_c1, h_c2)

            # Normalize if the third coordinate is not zero
            if p[2] != 0:
                p = p / p[2]

            # Can the point be seen by the cams?

            intersection_points.append(p[0:2])
        return intersection_points

    def get_z(self, rTNn):
        return net.process_y(net.get_y(rTNn))


# Camera 1
cam_1 = CameraSensor(100, 200, 45)
rC1Nn, Rc1n = get_r_and_R_from_A(cam_1.Apn)
c1_1, c1_2, c1_3 = get_basis_vectors(cam_1.Apn)

# Camera 2
cam_2 = CameraSensor(400, 500, 0)
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
