from dataclasses import dataclass, field
from typing import List

import numpy as np
import open3d as o3d

from lit.containers.base_container import BaseContainer
from lit.containers.fg_box import FGBox
from lit.containers.fg_object import FGObject
from lit.recon_utils import bbox_to_lineset


@dataclass
class FGScene(BaseContainer):
    """
    An FGScene contains a list of FGObjects. These FGObjects are from different
    frames of the same scene.

    With FGScene, one can extract the foreground mesh at a specific frame index.
    """

    fg_objects: List[FGObject] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        if self.fg_objects is None:
            self.fg_objects = []

    def __str__(self):
        return f"FGScene(with {len(self.fg_objects)} fg_objects)"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.fg_objects)

    def __getitem__(self, idx):
        return self.fg_objects[idx]

    def append(self, fg_object: FGObject):
        if not isinstance(fg_object, FGObject):
            raise ValueError(f"Type must be FGObject, but got {type(fg_object)}")
        self.fg_objects.append(fg_object)

    def to_dict(self):
        return {
            "fg_objects": [fg_object.to_dict() for fg_object in self.fg_objects],
        }

    @classmethod
    def from_dict(cls, dict_data: dict):
        return cls(
            fg_objects=[
                FGObject.from_dict(fg_object) for fg_object in dict_data["fg_objects"]
            ],
        )

    @classmethod
    def from_fg_boxes(cls, fg_boxes: List[FGBox]):
        """
        Group multiple foreground cub into multiple foreground groups.

        Args:
            fg_boxes: List of FGBox, the foreground boxes.
        """
        fg_objects = []
        for fg_box in fg_boxes:
            # Try to insert the fg_box in previous fg_objects.
            found_fg_object = False
            for fg_object in fg_objects:
                if fg_box.object_id == fg_object.object_id:
                    fg_object.insert(fg_box)
                    found_fg_object = True
                    break
            # If not found any fg_objects, create a new fg_object.
            if not found_fg_object:
                fg_object = FGObject()
                fg_object.insert(fg_box)
                fg_objects.append(fg_object)
        fg_scene = cls(fg_objects=fg_objects)
        return fg_scene

    def get_frame_mesh(
        self,
        frame_index: int,
        src_to_dst_scales: List[float] = None,
    ):
        """
        Get the fused foreground mesh in **world coord** at a specific frame.

        Args:
            frame_index: int, the frame index.
            src_to_dst_scales: List of 3 floats, the scale ratio from src to dst.

        Return:
            Open3D mesh, the fused foreground mesh in **world coord**.
        """
        all_group_fg_mesh = o3d.geometry.TriangleMesh()

        for fg_object in self.fg_objects:
            # Scale fg_object if needed.
            if src_to_dst_scales is not None:
                fg_object = fg_object.scale_by(src_to_dst_scales)

            # Append mesh.
            fg_mesh = fg_object.get_fg_mesh(frame_index)
            if fg_mesh is not None:
                all_group_fg_mesh += fg_mesh

        return all_group_fg_mesh

    def get_frame_ls(
        self,
        frame_index: int,
        src_to_dst_scales: List[float] = None,
    ):
        """
        Get the fused world-coord foreground line set at a specific frame index.

        Args:
            frame_index: int, the frame index.
            src_to_dst_scales: List of 3 floats, the scale ratio from src to dst.

        Return:
            Open3D line set, the fused foreground line set in **world coord**.
        """
        fg_ls = o3d.geometry.LineSet()

        for fg_object in self.fg_objects:
            # Scale fg_object if needed.
            if src_to_dst_scales is not None:
                fg_object = fg_object.scale_by(src_to_dst_scales)

            # Append ls.
            for fg_box in fg_object.fg_boxes:
                if fg_box.frame_index == frame_index:
                    fg_box_ls = bbox_to_lineset(
                        fg_box.local_bbox,
                        frame_pose=fg_box.frame_pose,
                    )
                    fg_ls += fg_box_ls

        return fg_ls

    def get_frame_local_bboxes(
        self,
        frame_index: int,
        src_to_dst_scales: List[float] = None,
    ):
        """
        Get the fused foreground local bboxes at a specific frame index.

        Args:
            frame_index: int, the frame index.
            src_to_dst_scales: List of 3 floats, the scale ratio from src to dst.

        Return:
            bboxes, shape (N, 8). BBoxes are in **local coordinates**.
            Each bbox is: (x, y, z, dx, dy, dz, heading, class)
        """
        local_bboxes = []
        for fg_object in self.fg_objects:
            # Scale fg_object if needed.
            if src_to_dst_scales is not None:
                fg_object = fg_object.scale_by(src_to_dst_scales)

            # Append bboxes.
            for fg_box in fg_object.fg_boxes:
                if fg_box.frame_index == frame_index:
                    assert fg_box.local_bbox.shape == (8,)
                    local_bboxes.append(fg_box.local_bbox)
        local_bboxes = np.asarray(local_bboxes).reshape((-1, 8)).astype(np.float32)

        return local_bboxes

    def get_frame_indices(self):
        """
        Get all sample indices of the fg_objects.
        """
        frame_indices = []
        for fg_object in self.fg_objects:
            for fg_box in fg_object.fg_boxes:
                frame_indices.append(fg_box.frame_index)
        frame_indices = sorted(list(set(frame_indices)))
        natural_indices = list(range(len(frame_indices)))
        if not np.all(np.array(frame_indices) == np.array(natural_indices)):
            raise ValueError("Sample indices are not continuous.")

        return frame_indices

    def discard_large_fg_objects(self, discard_ratio: float) -> None:
        """
        Discard large fg_objects by their volume of the reconstructed mesh's
        axis-aligned bounding box in canonical coordinates.

        Args:
            discard_ratio: float, the ratio of the largest fg_objects to be discarded.
        """
        num_discard = int(len(self.fg_objects) * discard_ratio)
        print(f"To discard {num_discard} out of {len(self.fg_objects)} fg_scene.")

        # Compute the volume of the axis-aligned bounding box of each fg_object.
        volumes = []
        for fg_object in self.fg_objects:
            min_bound = np.min(fg_object.mesh_vertices, axis=0)
            max_bound = np.max(fg_object.mesh_vertices, axis=0)
            volume = np.prod(max_bound - min_bound)
            volumes.append(volume)

        # Discard.
        keep_indices = np.argsort(volumes)[:-num_discard]
        self.fg_objects = [self.fg_objects[i] for i in keep_indices]
