from dataclasses import dataclass

import camtools as ct
import numpy as np

from lit.containers.base_container import BaseContainer


@dataclass
class SimFrame(BaseContainer):
    """
    Storing data for simulated frames.
    """

    frame_index: int = None  # Frame index in the scene.
    frame_pose: np.ndarray = None  # (4, 4) pose of the frame (vehicle)
    local_points: np.ndarray = None  # (N, 3) points in local coordinates.
    local_bboxes: np.ndarray = None  # (M, 7) bboxes in local coordinates.
    incident_angles: np.ndarray = None  # (N, ) incident angles of the points.

    def __post_init__(self):
        super().__post_init__()

    def to_dict(self):
        return {
            "frame_index": self.frame_index,
            "frame_pose": self.frame_pose,
            "local_points": self.local_points,
            "local_bboxes": self.local_bboxes,
            "incident_angles": self.incident_angles,
        }

    @classmethod
    def from_dict(cls, dict_data: dict):
        return cls(
            frame_index=(
                dict_data["frame_index"].item()
                if isinstance(dict_data["frame_index"], np.ndarray)
                else dict_data["frame_index"]
            ),
            frame_pose=dict_data["frame_pose"],
            local_points=dict_data["local_points"],
            local_bboxes=dict_data["local_bboxes"],
            incident_angles=dict_data["incident_angles"],
        )

    def get_ray_keep_inputs_with_lidar_center(self, lidar_center: np.ndarray):
        """
        Computes inputs for raydrop given the lidar center.

        Parameters:
        - lidar_center: numpy.ndarray of shape (3,) representing the LiDAR sensor's position.

        Returns:
        - ray_drop_input: (N, 5) array consisting of direction (dir_x, dir_y, dir_z), distance, and incident angle for each point.
        """
        # Adjust points relative to the lidar_center
        adjusted_points = self.local_points - lidar_center

        # Rotate points 90 degress around z-axis to align waymo to nuScenes
        rotation_matrix_90_deg_z = np.array(
            [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        adjusted_points = ct.transform.transform_points(
            adjusted_points, rotation_matrix_90_deg_z
        )

        # Normalize vectors for direction calculation
        normalized_vectors = adjusted_points / np.linalg.norm(
            adjusted_points, axis=1, keepdims=True
        )
        dir_x, dir_y, dir_z = normalized_vectors.T

        # Calculate the distance from the lidar to each point
        dist = np.linalg.norm(adjusted_points, axis=1)

        # Use the stored incident angles
        incident_angle = self.incident_angles

        # Construct the ray_drop input array (N, 5)
        ray_drop_input = np.column_stack((dir_x, dir_y, dir_z, dist, incident_angle))

        return ray_drop_input

    def get_ray_keep_inputs(self, lidar_to_vehicle_pose):
        """
        Returns (N, 5) array.

        dir_x, dir_y, dir_z, dist, incident_angle
        """
        C = ct.convert.pose_to_C(lidar_to_vehicle_pose)
        return self.get_ray_keep_inputs_with_lidar_center(C)
