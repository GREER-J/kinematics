import numpy as np
from itertools import combinations
from src.camera_sensor import CameraSensor, CameraDirectionMeasurement


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

            if y1.cam_id == y2.cam_id:
                # It's the came camera
                continue
            # Construct homogeneous lines from camera positions and direction vectors
            h_c1 = get_homogeneous_line_from_point_and_direction(y1.base_point, y1.dir_vec)
            h_c2 = get_homogeneous_line_from_point_and_direction(y2.base_point, y2.dir_vec)

            # Calculate intersection point with the cross product
            p = np.cross(h_c1, h_c2)

            # Normalize if the third coordinate is not zero
            if p[2] != 0:
                p = p / p[2]

            # Create into position vector
            p[2] = 0  # no z component
            p = p.reshape((3,1))

            # Can the point be seen by the cams?
            point_in_all_cams_FOV = True
            for cam in self._cams:
                seen_by_cam = cam.is_point_in_vis(p)
                if not seen_by_cam:
                    point_in_all_cams_FOV = False
                    break

            if point_in_all_cams_FOV:
                intersection_points.append(p[0:2])
        return intersection_points

    def get_z(self, rTNn):
        return self.process_y(self.get_y(rTNn))
