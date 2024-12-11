from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import camtools as ct
import numpy as np
import open3d as o3d
from tqdm import tqdm

from lit.containers.base_container import BaseContainer
from lit.containers.fg_box import FGBox
from lit.containers.frame import Frame
from lit.recon_utils import (
    bboxes_to_lineset,
    get_indices_inside_bboxes,
    get_indices_outside_bboxes,
    remove_statistical_outlier,
)


@dataclass
class Scene(BaseContainer):
    """
    Class to handle a scene of frames. This includes:

    - Collecting frame data and into a scene data.
    - Saving the scene to disk.
    - Loading the scene from disk.
    - Scene processing, e.g. filtering, reconstruction, etc.

    Save as a .pth file with PyTorch.
    """

    scene_name: str = None
    frames: List[Frame] = field(default_factory=list)

    # Backup normalizer pose (not used).
    _normalizer_pose: np.ndarray = None
    _is_pose_normalized: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.frames is None:
            self.frames = []

    def to_dict(self):
        return {
            "scene_name": self.scene_name,
            "frames": [frame.to_dict() for frame in self.frames],
        }

    @classmethod
    def from_dict(cls, dict_data: dict):
        return cls(
            scene_name=dict_data["scene_name"],
            frames=[Frame.from_dict(frame) for frame in dict_data["frames"]],
        )

    def normalize_poses(self):
        """
        Normalize frames' poses for better reconstruction, and save the
        normalizer pose to self.normalizer_pose.
        """
        frame_poses = [frame.frame_pose for frame in self.frames]
        frame_poses = np.stack(frame_poses, axis=0)

        center_index = len(frame_poses) // 2
        center_pose = frame_poses[center_index]

        # Before: +x points to front.
        # After : +y points to front, +x points to right.
        rotate_x_y_R = ct.convert.euler_to_R(yaw=np.pi / 2, pitch=0, roll=0)
        rotate_x_y_pose = np.eye(4)
        rotate_x_y_pose[:3, :3] = rotate_x_y_R
        self._normalizer_pose = rotate_x_y_pose @ np.linalg.inv(center_pose)

        for frame in self.frames:
            frame.frame_pose = self._normalizer_pose @ frame.frame_pose
        self._is_pose_normalized = True

    def undo_normalize_poses(self):
        """
        Undo normalization of poses, by applying the inverse of the normalizer
        pose to all frames.
        """
        if not self._is_pose_normalized:
            raise ValueError("Poses are not normalized yet.")
        inv_normalizer_pose = np.linalg.inv(self._normalizer_pose)
        for frame in self.frames:
            frame.frame_pose = inv_normalizer_pose @ frame.frame_pose

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]

    def __iter__(self):
        return iter(self.frames)

    def __str__(self) -> str:
        return f"Scene({self.scene_name}, {len(self.frames)} frames)"

    def __repr__(self) -> str:
        return self.__str__()

    def is_frame_valid(self, index):
        """
        Check if frame at index is valid (same scene name and index).
        """
        return (
            self.frames[index].scene_name == self.scene_name
            and self.frames[index].frame_index == index
        )

    def sample_by_indices(self, frame_indices):
        """
        Sample a subset of frames by indices. This will modify the Scene.
        """
        self.frames = [self.frames[i] for i in frame_indices]

    def append_frame(self, frame: Frame, check_valid=True):
        """
        Append a frame to the scene.

        Args:
            frame: Frame to append.
            check_valid: Check frames's frame_index for sequential appending
                and check the scene name.
        """
        if check_valid:
            if frame.scene_name != self.scene_name:
                raise ValueError(
                    f"Scene name {frame.scene_name} does not match "
                    f"scene name {self.scene_name}"
                )
            if len(self.frames) != frame.frame_index:
                raise ValueError(
                    f"Frame index {frame.frame_index} does not match "
                    f"scene length {len(self.frames)}"
                )
        self.frames.append(frame)

    def extract_fg(
        self,
        select_labels: List[int] = None,
        verbose: bool = False,
    ):
        """
        Extract foreground from the scene.

        Args:
            select_labels: List of enabled labels. Set to None to enable all.

        Returns:
            [FGBox, FGBox, ...]  # A flat list of FGBoxes in all frames.
            Each FGBox already contains the scene_name and frame_index
            information so it knows which frame it belongs to.

        Notes on FGBox:
            FGBox(local_points, local_bbox, frame_pose, frame_index)
        """
        fg_boxes = []
        for frame in tqdm(
            self,
            desc="Extracting foreground points",
            disable=not verbose,
        ):
            # Hard-code: only the top lidar is used.
            top_lidar_start_idx = 0
            top_lidar_end_idx = frame.num_points_of_each_lidar[0]

            # Extract points.
            local_points = frame.local_points[top_lidar_start_idx:top_lidar_end_idx][
                :, :3
            ]

            # Extract bboxes by label.
            if select_labels is None:
                bbox_mask = np.ones_like(frame.local_bboxes[:, 7], dtype=np.bool)
            else:
                bbox_mask = np.isin(frame.local_bboxes[:, 7], select_labels)

            local_bboxes = frame.local_bboxes[bbox_mask]
            indices_inside_bboxes = get_indices_inside_bboxes(
                points=local_points, bboxes=local_bboxes
            )

            # Mask object_ids.
            assert len(frame.object_ids) == len(bbox_mask)
            object_ids = [
                object_id
                for (object_id, mask) in zip(frame.object_ids, bbox_mask)
                if mask
            ]

            # Prepare FGBoxes.
            assert len(local_bboxes) == len(indices_inside_bboxes) == len(object_ids)
            for (
                local_bbox,
                indices,
                object_id,
            ) in zip(
                local_bboxes,
                indices_inside_bboxes,
                object_ids,
            ):
                if len(indices) == 0:
                    continue
                fg_boxes.append(
                    FGBox(
                        scene_name=frame.scene_name,
                        frame_index=frame.frame_index,
                        frame_pose=frame.frame_pose,
                        object_id=object_id,
                        local_points=local_points[indices],
                        local_bbox=local_bbox,
                    )
                )

        return fg_boxes

    def extract_bg(
        self,
        enabled_lidars=(0,),
        remove_foreground=True,
        expand_box_ratio=1.0,
        raise_bbox=0.0,
        per_frame_rso_nb_neighbors=0,
        per_frame_rso_std_ratio=0.0,
        verbose=False,
    ):
        """
        Extract background from the scene by removing points within the
        annotated bounding boxes.

        Args:
            enabled_lidars: Indices of lidars to use.
                0: Top
                1: Front
                2: Side left
                3: Side right
                4: Rear
            remove_foreground: Remove foreground points within bboxes.
            expand_box_ratio: Expand the bbox by this ratio.
            raise_bbox: Raise the bbox z-axis by this amount (in meters) to keep
                points on the ground. This is useful only when remove_foreground
                is True.
            per_frame_rso_nb_neighbors:
                Number of neighbors to use for statistical outlier removal, per frame.
            per_frame_rso_std_ratio:
                Standard deviation ratio for statistical outlier removal, per frame.

        Returns:
            {
                "points": (N, 3) background points.
                "lidar_poses": (N, 4, 4) per-point lidar poses.
                "lidar_centers": (N, 3) per-point lidar centers.
                "linesets": List of Open3D line sets.
            }
        """
        if not isinstance(enabled_lidars, (list, tuple)):
            raise ValueError(
                f"enabled_lidars must be a list or tuple, got {enabled_lidars}"
            )

        all_points = []  # (N, 3) after concatenation.
        all_lidar_poses = []  # (N, 4, 4) after concatenation.
        all_linesets = []  # List of Open3D line sets. TODO: return all_bboxes.
        all_unique_lidar_poses = []  # (M, 4, 4) after concatenation.

        for frame in tqdm(
            self,
            desc="Extracting background points",
            disable=not verbose,
        ):
            # Compute lidar start/end indices.
            lidar_start_indices = np.cumsum([0] + frame.num_points_of_each_lidar[:-1])
            lidar_end_indices = np.cumsum(frame.num_points_of_each_lidar)
            frame_pose = frame.frame_pose

            local_points = []
            lidar_poses = []
            for lidar_idx in enabled_lidars:
                # Collect points
                start_idx = lidar_start_indices[lidar_idx]
                end_idx = lidar_end_indices[lidar_idx]
                lidar_local_points = frame.local_points[start_idx:end_idx][:, :3]

                # Remove foreground points.
                if remove_foreground:
                    local_bboxes = np.copy(frame.local_bboxes)
                    if expand_box_ratio != 1.0:
                        local_bboxes[:, 3:6] *= expand_box_ratio
                    if raise_bbox != 0:
                        local_bboxes[:, 2] += raise_bbox
                    outside_indices = get_indices_outside_bboxes(
                        lidar_local_points, local_bboxes
                    )
                    lidar_local_points = lidar_local_points[outside_indices]

                # Remove statistical outliers.
                if per_frame_rso_nb_neighbors != 0:
                    lidar_local_points = remove_statistical_outlier(
                        lidar_local_points,
                        nb_neighbors=per_frame_rso_nb_neighbors,
                        std_ratio=per_frame_rso_std_ratio,
                    )
                local_points.append(lidar_local_points)

                # Collect lidar poses.
                lidar_pose = frame_pose @ frame.lidar_to_vehicle_poses[lidar_idx]
                lidar_poses.append(
                    np.tile(lidar_pose, (len(lidar_local_points), 1, 1)),
                )
                all_unique_lidar_poses.append(lidar_pose)

            local_points = np.concatenate(local_points, axis=0)
            points = ct.transform.transform_points(local_points, frame_pose)
            all_points.append(points)

            lidar_poses = np.concatenate(lidar_poses, axis=0)
            all_lidar_poses.append(lidar_poses)

            # Bboxes.
            linesets = bboxes_to_lineset(
                bboxes=frame.local_bboxes,
                frame_pose=frame_pose,
            )
            all_linesets.append(linesets)

        all_points = np.concatenate(all_points, axis=0).astype(np.float32)
        all_lidar_poses = np.concatenate(all_lidar_poses, axis=0).astype(np.float32)

        bbox_lineset = o3d.geometry.LineSet()
        for lineset in all_linesets:
            bbox_lineset += lineset

        all_lidar_centers = all_lidar_poses[:, :3, 3]
        all_unique_lidar_poses = np.array(all_unique_lidar_poses).astype(np.float32)

        return {
            "points": all_points,
            "lidar_poses": all_lidar_poses,
            "lidar_centers": all_lidar_centers,
            "lineset": bbox_lineset,
            "unique_lidar_poses": all_unique_lidar_poses,
        }

    def visualize(self):
        # Extract background.
        bg_data = self.extract_bg(
            enabled_lidars=(0,),
            remove_foreground=True,
            raise_bbox=0.0,
        )

        # Extract foreground.
        fg_boxes = self.extract_fg(
            select_labels=(1,),
            verbose=False,
        )

        # Visualize.
        bg_points = bg_data["points"]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(bg_points)
        o3d.visualization.draw_geometries([pcd])


def test_load_scene():
    scene_dir = Path("/media/data/projects/lit/lit_data/nuscenes/scene")
    scene_name = "0ac05652a4c44374998be876ba5cd6fd.pkl"
    scene_path = scene_dir / scene_name
    scene = Scene.load(scene_path)
    print(f"Loaded scene: {scene} of {len(scene)} frames")


if __name__ == "__main__":
    test_load_scene()
