from dataclasses import dataclass
from typing import List

import camtools as ct
import numpy as np
import open3d as o3d

from lit.containers.base_container import BaseContainer
from lit.recon_utils import bbox_to_lineset, scale_points_with_bbox


@dataclass
class FGBox(BaseContainer):
    """
    Foreground box containing points, bbox, etc.
    """

    scene_name: str = None  # Scene name.
    frame_index: int = None  # Index of the frame.
    frame_pose: np.ndarray = None  # (4, 4) pose of the frame (vehicle).
    object_id: str = None  # Globally unique object id of the box.
    local_points: np.ndarray = None  # (N, 3), local-coord points.
    local_bbox: np.ndarray = None  # (8,) box: (x, y, z, dx, dy, dz, heading, class).

    def __post_init__(self):
        super().__post_init__()
        if self.object_id is None:
            raise ValueError("object_id must be provided.")

    def __str__(self):
        return f"FGBox(frame_index={self.frame_index}, object_id={self.object_id})"

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return {
            "scene_name": self.scene_name,
            "frame_index": self.frame_index,
            "frame_pose": self.frame_pose,
            "object_id": self.object_id,
            "local_points": self.local_points,
            "local_bbox": self.local_bbox,
        }

    @classmethod
    def from_dict(cls, dict_data: dict):
        return cls(
            scene_name=dict_data["scene_name"],
            frame_index=dict_data["frame_index"],
            frame_pose=dict_data["frame_pose"],
            object_id=dict_data["object_id"],
            local_points=dict_data["local_points"],
            local_bbox=dict_data["local_bbox"],
        )

    def scale_by(self, src_to_dst_scales: List[float]):
        """
        Scale the FGBox by src_to_dst_scales.
        """
        new_local_points, new_local_bbox = scale_points_with_bbox(
            points=self.local_points,
            bbox=self.local_bbox,
            src_to_dst_scales=src_to_dst_scales,
        )
        return FGBox(
            scene_name=self.scene_name,
            frame_index=self.frame_index,
            frame_pose=self.frame_pose,
            object_id=self.object_id,
            local_points=new_local_points,
            local_bbox=new_local_bbox,
        )

    def compute_local_pseudo_pose(self):
        """
        Compute the pseudo pose of the box, which is the pose of transforming
        a axis aligned box centered at (0, 0, 0) to the current box.

        Usage:
            ```python
            # Center points to the origin for reconstruction.
            pseudo_pose = self.compute_local_pseudo_pose()
            pseudo_T = ct.convert.pose_to_T(pseudo_pose)
            centered_points = ct.transform.transform_points(
                self.local_points, pseudo_T
            )

            # Recon.
            centered_pcd = o3d.geometry.PointCloud()
            centered_pcd.points = o3d.utility.Vector3dVector(centered_points)
            centered_mesh = (
                o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    centered_pcd, alpha=0.15
                )
            )

            # Transform back to the world coordinate.
            # mesh_vertices = ct.transform.transform_points(
            #     np.asarray(centered_mesh.vertices), self.frame_pose @ pseudo_pose
            # )
            # mesh_triangles = np.asarray(centered_mesh.triangles)
            # world_mesh = o3d.geometry.TriangleMesh()
            # world_mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
            # world_mesh.triangles = o3d.utility.Vector3iVector(mesh_triangles)
            world_mesh = centered_mesh.transform(self.frame_pose @ pseudo_pose)
            world_mesh.compute_vertex_normals()

            # Our usual way to get world lineset.
            world_ls = bbox_to_lineset(self.local_bbox, frame_pose=self.frame_pose)

            # Visualize.
            o3d.visualization.draw_geometries([world_ls, world_mesh])
            ```

        Notes:
            Current bbox:
                (x, y, z, dx, dy, dz, heading)
            Canonical bbox:
                Axis aligned bbox of the same size centered at the origin
                (0, 0, 0, dx, dy, dz, 0)
        """

        theta = self.local_bbox[6]
        R = np.array(
            [
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        R = R.T
        t = self.local_bbox[:3]
        pseudo_pose = np.eye(4)
        pseudo_pose[:3, :3] = R
        pseudo_pose[:3, 3] = t

        visualize = False
        if visualize:

            def visualization_01():
                """
                Transform a canonical bbox to the local bbox.
                """
                # Visualization 1: transform a canonical bbox to local bbox.
                # They shall overlap.
                local_ls = bbox_to_lineset(self.local_bbox)
                local_ls.paint_uniform_color([0, 0, 1])
                x, y, z, dx, dy, dz, heading = self.local_bbox[:7]
                canonical_bbox = np.array([0, 0, 0, dx, dy, dz, 0])
                canonical_ls = bbox_to_lineset(canonical_bbox, pseudo_pose)
                canonical_ls.paint_uniform_color([1, 0, 0])
                o3d.visualization.draw_geometries([local_ls, canonical_ls])

            def visualization_02():
                """
                Center the local points to origin for reconstruction.
                """
                centered_ls = bbox_to_lineset(self.local_bbox)
                centered_ls.paint_uniform_color([0, 0, 1])
                centered_ls.points = o3d.utility.Vector3dVector(
                    ct.transform.transform_points(
                        np.asarray(centered_ls.points), np.linalg.inv(pseudo_pose)
                    )
                )

                centered_points = np.copy(self.local_points)
                centered_points = ct.transform.transform_points(
                    centered_points, np.linalg.inv(pseudo_pose)
                )
                centered_pcd = o3d.geometry.PointCloud()
                centered_pcd.points = o3d.utility.Vector3dVector(centered_points)

                axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
                o3d.visualization.draw_geometries([axes, centered_pcd, centered_ls])

            def visualization_03():
                """
                A full recon pipeline:

                1. Transform bbox and points to centered canonical bbox.
                2. Recon mesh.
                3. Transform mesh back to the world coordinate.
                """

                # Center points to the origin for reconstruction.
                # pseudo_pose = self.compute_local_pseudo_pose() # Avoid recursion.
                pseudo_T = ct.convert.pose_to_T(pseudo_pose)
                centered_points = ct.transform.transform_points(
                    self.local_points, pseudo_T
                )

                # Recon.
                centered_pcd = o3d.geometry.PointCloud()
                centered_pcd.points = o3d.utility.Vector3dVector(centered_points)
                centered_mesh = (
                    o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                        centered_pcd, alpha=0.15
                    )
                )

                # Transform back to the world coordinate.
                # mesh_vertices = ct.transform.transform_points(
                #     np.asarray(centered_mesh.vertices), self.frame_pose @ pseudo_pose
                # )
                # mesh_triangles = np.asarray(centered_mesh.triangles)
                # world_mesh = o3d.geometry.TriangleMesh()
                # world_mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
                # world_mesh.triangles = o3d.utility.Vector3iVector(mesh_triangles)
                # world_mesh.compute_vertex_normals()
                world_mesh = centered_mesh.transform(self.frame_pose @ pseudo_pose)

                # Our usual way to get world lineset.
                world_ls = bbox_to_lineset(self.local_bbox, frame_pose=self.frame_pose)

                # Visualize.
                o3d.visualization.draw_geometries([world_ls, world_mesh])

            visualization_01()
            visualization_02()
            # visualization_03()

        return pseudo_pose
