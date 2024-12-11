import copy
from dataclasses import dataclass
from typing import List

import camtools as ct
import numpy as np

from lit.containers.base_container import BaseContainer


@dataclass
class Frame(BaseContainer):
    """
    TODO: rename points and bbox to local_points, etc.
    """

    scene_name: str  # Scene name
    frame_index: int  # Frame index in the scene
    frame_pose: np.ndarray  # (4, 4) pose of the frame (vehicle)
    lidar_to_vehicle_poses: List[np.ndarray]  # List of (4, 4) poses of the lidars
    num_points_of_each_lidar: List[int]  # After NLZ_flag filtering
    local_points: np.ndarray  # (N, 5) point cloud: (x, y, z, i, e)
    local_bboxes: np.ndarray  # (M, 8) boxes: (x, y, z, dx, dy, dz, heading, class)
    object_ids: List[str]  # (M,) object ids of the boxes

    def __post_init__(self):
        """
        Sanity checks for the frame data.
        """
        super().__post_init__()
        if len(self.num_points_of_each_lidar) != len(self.lidar_to_vehicle_poses):
            raise ValueError(
                f"len(num_points_of_each_lidar) != len(lidar_to_vehicle_poses): "
                f"{len(self.num_points_of_each_lidar)} != "
                f"{len(self.lidar_to_vehicle_poses)}"
            )
        if len(self.local_points) != np.sum(self.num_points_of_each_lidar):
            raise ValueError(
                f"len(points) != np.sum(num_points_of_each_lidar): "
                f"{len(self.local_points)} != {np.sum(self.num_points_of_each_lidar)}"
            )
        ct.sanity.assert_pose(self.frame_pose)
        for pose in self.lidar_to_vehicle_poses:
            ct.sanity.assert_pose(pose)

    def __str__(self):
        return f"Frame(scene_name={self.scene_name}, frame_index={self.frame_index})"

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return {
            "scene_name": self.scene_name,
            "frame_index": self.frame_index,
            "frame_pose": self.frame_pose,
            "lidar_to_vehicle_poses": self.lidar_to_vehicle_poses,
            "num_points_of_each_lidar": self.num_points_of_each_lidar,
            "local_points": self.local_points,
            "local_bboxes": self.local_bboxes,
            "object_ids": self.object_ids,
        }

    @classmethod
    def from_dict(cls, dict_data: dict):
        return cls(
            scene_name=dict_data["scene_name"],
            frame_index=dict_data["frame_index"],
            frame_pose=dict_data["frame_pose"],
            lidar_to_vehicle_poses=dict_data["lidar_to_vehicle_poses"],
            num_points_of_each_lidar=dict_data["num_points_of_each_lidar"],
            local_points=dict_data["local_points"],
            local_bboxes=dict_data["local_bboxes"],
            object_ids=dict_data["object_ids"],
        )

    def clone(self):
        """
        Clone the Frame.
        """
        return copy.deepcopy(self)

    def visualize(self, **kwargs):
        """
        Visualize the frame.
        """
        raise NotImplementedError
