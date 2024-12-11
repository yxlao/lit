from dataclasses import dataclass, field
from typing import List

import numpy as np
import open3d as o3d

from lit.containers.base_container import BaseContainer
from lit.containers.fg_box import FGBox
from lit.recon_utils import bbox_to_lineset


@dataclass
class FGObject(BaseContainer):
    """
    A group of FGBoxes that are of the same object.

    A FGObject contains FGBoxes from different frames, but they must have the
    same object_id. The FGObject also contains the reconstructed mesh of
    (mesh_vertices, mesh_triangles).
    """

    # Main data.
    object_id: str = None  # Object id of the group, all fg_boxes must have the same id
    fg_boxes: List[FGBox] = field(default_factory=list)  # len == # frame with object_id

    # Derived data.
    # For each foreground group, we have one mesh centered at the canonical
    # position. This mesh needs to be transformed to the correct position
    # according to each fg_box.
    mesh_vertices: np.ndarray = None  # (N, 3), vertices of the reconstructed mesh
    mesh_triangles: np.ndarray = None  # (M, 3), triangles of the reconstructed mesh

    def __post_init__(self):
        super().__post_init__()
        if self.fg_boxes is None:
            self.fg_boxes = []

    def __str__(self):
        return f"FGObject({len(self.fg_boxes)} fg_boxes)"

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, idx):
        return self.fg_boxes[idx]

    def __len__(self):
        return len(self.fg_boxes)

    def to_dict(self):
        return {
            "object_id": self.object_id,
            "fg_boxes": [fg_box.to_dict() for fg_box in self.fg_boxes],
            "mesh_vertices": self.mesh_vertices,
            "mesh_triangles": self.mesh_triangles,
        }

    @classmethod
    def from_dict(cls, dict_data: dict):
        return cls(
            object_id=dict_data["object_id"],
            fg_boxes=[FGBox.from_dict(fg_box) for fg_box in dict_data["fg_boxes"]],
            mesh_vertices=dict_data["mesh_vertices"],
            mesh_triangles=dict_data["mesh_triangles"],
        )

    def render_debug(self):
        """
        Render the mesh across frames.
        """
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.mesh_vertices)
        mesh.triangles = o3d.utility.Vector3iVector(self.mesh_triangles)
        mesh.compute_vertex_normals()

        skip_every = 4
        random_color = np.random.rand(3)
        fg_object_ls = o3d.geometry.LineSet()
        for fg_box in self.fg_boxes:
            if fg_box.frame_index % skip_every != 0:
                continue
            frame_pose = fg_box.frame_pose
            frame_ls = bbox_to_lineset(fg_box.local_bbox, frame_pose)
            frame_ls.paint_uniform_color(random_color)
            fg_object_ls += frame_ls

        o3d.visualization.draw_geometries([mesh])
        o3d.visualization.draw_geometries([fg_object_ls])

        return mesh

    def insert(self, fg_box: FGBox):
        """
        Insert a fg_box to group. The inserted fg_box must have the same object_id.

        Args:
            fg_box: FGBox, the fg_box to insert.
        """
        if not isinstance(fg_box, FGBox):
            raise ValueError(f"fg_box must be FGBox, got {type(fg_box)}")

        if self.object_id is None:
            self.object_id = fg_box.object_id
        elif self.object_id != fg_box.object_id:
            raise ValueError(
                f"Cannot insert fg_box with object_id {fg_box.object_id} to group "
                f"with object_id {self.object_id}."
            )
        self.fg_boxes.append(fg_box)

    def scale_by(self, src_to_dst_scales: List[float]):
        """
        Scale the FGObject by src_to_dst_scales.

        Args:
            src_to_dst_scales: List of 3 floats, the scale ratio from src to dst.
        """
        new_object_id = self.object_id
        new_fg_boxes = [fg_box.scale_by(src_to_dst_scales) for fg_box in self.fg_boxes]

        # As the mesh is centered at the canonical position, we scale the
        # vertices directly.
        new_mesh_vertices = self.mesh_vertices * src_to_dst_scales
        new_mesh_triangles = self.mesh_triangles.copy()

        return FGObject(
            object_id=new_object_id,
            fg_boxes=new_fg_boxes,
            mesh_vertices=new_mesh_vertices,
            mesh_triangles=new_mesh_triangles,
        )

    def get_fg_mesh(self, frame_index):
        """
        Get the foreground mesh in world coord at a specific frame index.

        In a FGObject, the could be 0 or 1 FGBox with the specified frame_index:
        - 0 FGBox with the specified frame_index, return None.
        - 1 FGBox with the specified frame_index, return the
          corresponding mesh in world coord.
        - More than 1 FGBox with the specified frame_index, raise ValueError.

        Returns the mesh in **world** coord.
        """
        fg_mesh = None

        num_fg_boxes_with_frame_index = 0
        for fg_box in self.fg_boxes:
            if fg_box.frame_index == frame_index:
                # The centered, canonical mesh.
                fg_mesh = o3d.geometry.TriangleMesh()
                fg_mesh.vertices = o3d.utility.Vector3dVector(self.mesh_vertices)
                fg_mesh.triangles = o3d.utility.Vector3iVector(self.mesh_triangles)

                # Convert to world coord.
                frame_pose = fg_box.frame_pose
                pseudo_pose = fg_box.compute_local_pseudo_pose()
                fg_mesh = fg_mesh.transform(frame_pose @ pseudo_pose)

                # Count.
                num_fg_boxes_with_frame_index += 1

        if not num_fg_boxes_with_frame_index in [0, 1]:
            raise ValueError(
                f"For each FGObject, there could only be 0 or 1 FGBox "
                f"in a particular frame. But got {num_fg_boxes_with_frame_index}."
            )

        return fg_mesh
