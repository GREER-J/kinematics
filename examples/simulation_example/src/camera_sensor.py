import numpy as np
from numpy.linalg import norm
from dataclasses import dataclass
from kinematics_library import homogeneous as htf
from kinematics_library.homogeneous_utils import get_basis_vectors, get_r_from_A


@dataclass
class CameraDirectionMeasurement:
    cam_id: int
    base_point: np.ndarray
    dir_vec: np.ndarray

    @property
    def x(self) -> np.ndarray:
        return np.ndarray([[self.cam_id], [self.cam_id], [self.cam_id]])


class CameraSensor:
    def __init__(self, cam_id: int, E: float, N: float, bearing_deg: float):
        self.cam_id = cam_id
        self.y_FOV = 80
        self.z_FOV = 170
        self.Apn = htf.transx(N) @ htf.transy(E) @ htf.rotz(np.deg2rad(bearing_deg))  # NED
        self.false_target_per_timestep = 10

    def get_dir_vec(self, rTNn: np.ndarray) -> np.ndarray:
        rPNn = get_r_from_A(self.Apn)
        dr = rTNn - rPNn
        return dr / norm(dr)

    def get_false_targets(self, N: int):
        false_targets = []
        for _ in range(N):
            A = self.Apn@ htf.rotz(np.deg2rad(np.random.uniform(-self.z_FOV/2, self.z_FOV/2)))@htf.roty(np.deg2rad(np.random.uniform(-self.y_FOV/2, self.y_FOV/2))) @ htf.transx(10)
            r = get_r_from_A(A)
            false_targets.append(r)
        return false_targets

    def get_detections(self, rTNn) -> list[CameraDirectionMeasurement]:
        measurements = []

        rPNn = get_r_from_A(self.Apn)
        measurements.append(CameraDirectionMeasurement(self.cam_id, rPNn, self.get_dir_vec(rTNn)))

        for false_target in self.get_false_targets(self.false_target_per_timestep):
            measurements.append(CameraDirectionMeasurement(self.cam_id, rPNn, self.get_dir_vec(false_target)))
        return measurements

    def is_point_in_vis(self, r_vec) -> bool:
        # Rotate into maximal FOV limits and check e2 to see if it's positive or negative
        # Or just take angle via dot product and directly check FOV limits
        u_r_vec = self.get_dir_vec(r_vec)
        e1, e2, _ = get_basis_vectors(self.Apn)

        # z rotation
        angle = self.get_angle_between_vecs(u_r_vec, e1)
        if angle > self.z_FOV:

            return False

        angle = self.get_angle_between_vecs(u_r_vec, e2)
        if angle > self.y_FOV:

            return False

        return True

    def get_angle_between_vecs(self, u_r_vec, e1):
        angle = np.acos(np.dot(e1.T, u_r_vec))
        return angle
