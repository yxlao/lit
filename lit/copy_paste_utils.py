from pathlib import Path

import camtools as ct
import numpy as np
import open3d as o3d
import torch

from lit.containers.fg_scene import FGScene
from lit.lidar import KITTILidarIntrinsics, Lidar, NuScenesLidarIntrinsics
from lit.path_utils import get_lit_paths
from lit.raycast_engine_gpu import RaycastEngineGPU
from lit.recon_utils import (
    bbox_to_lineset,
    get_indices_inside_bbox,
    get_indices_outside_bboxes,
)

g_raycast_engine = RaycastEngineGPU()


def recompute_voxels(
    dataloader: torch.utils.data.DataLoader,
    batch_dict: dict,
) -> dict:
    """
    Given batch_dict, recompute voxels based on the points in batch_dict.

    This is useful if the points in batch_dict have been modified, e.g. by
    copy-pasting points from another dataset.

    This is essentially extracted steps in DatasetTemplate.prepare_data(), which
    is called by KittiDataset.__getitem__() and its friends.

    Args:
        dataloader: The dataloader object used to run feature encoder and
            data processor.
        batch_dict: dict. Must at least contain "points".
            - The batch_dict will be modified in-place!
            - Must have "points".
            - Batch size must be 1, i.e. "points" must be (N, 4) and
              points[:, 0] must be all 0.
            - Must be in numpy. Call recompute_voxels before passing to GPU.

    Returns:
        batch_dict: dict. The modified batch_dict.
    """

    before_points_shape = batch_dict["points"].shape
    before_voxels_shape = batch_dict["voxels"].shape
    before_voxel_coords_shape = batch_dict["voxel_coords"].shape
    before_voxel_num_points_shape = batch_dict["voxel_num_points"].shape

    # Check the dimensions of points.
    if batch_dict["points"].ndim != 2 or batch_dict["points"].shape[1] != 4:
        raise ValueError("points must be (N, 4)")

    # Check that batch size is 1.
    if not np.allclose(batch_dict["points"][:, 0], 0):
        raise ValueError("points[:, 0] must be all 0")

    # 1. Pop the keys that will be re-computed.
    for key in ["voxels", "voxel_coords", "voxel_num_points"]:
        batch_dict.pop(key, None)

    # 2. Strip the batch dimension.
    batch_dict["points"] = batch_dict["points"][:, 1:]

    # 3. Re-run feature encoder and data processor.
    batch_dict = dataloader.dataset.point_feature_encoder.forward(batch_dict)
    batch_dict = dataloader.dataset.data_processor.forward(batch_dict)

    # 4. Add back the batch dimension.
    batch_dict["points"] = np.concatenate(
        [np.zeros((len(batch_dict["points"]), 1)), batch_dict["points"]], axis=1
    )
    batch_dict["voxel_coords"] = np.concatenate(
        [np.zeros((len(batch_dict["voxel_coords"]), 1)), batch_dict["voxel_coords"]],
        axis=1,
    )

    after_points_shape = batch_dict["points"].shape
    after_voxels_shape = batch_dict["voxels"].shape
    after_voxel_coords_shape = batch_dict["voxel_coords"].shape
    after_voxel_num_points_shape = batch_dict["voxel_num_points"].shape

    # if (
    #     before_points_shape != after_points_shape
    #     or len(before_voxels_shape) != len(after_voxels_shape)
    #     or before_voxels_shape[1:] != after_voxels_shape[1:]
    #     or len(before_voxel_coords_shape) != len(after_voxel_coords_shape)
    #     or before_voxel_coords_shape[1:] != after_voxel_coords_shape[1:]
    #     or len(before_voxel_num_points_shape) != len(after_voxel_num_points_shape)
    # ):
    #     print("[Before]")
    #     print(f"- points.shape          : {before_points_shape}")
    #     print(f"- voxels.shape          : {before_voxels_shape}")
    #     print(f"- voxel_coords.shape    : {before_voxel_coords_shape}")
    #     print(f"- voxel_num_points.shape: {before_voxel_num_points_shape}")
    #     print("[After]")
    #     print(f"- points.shape          : {after_points_shape}")
    #     print(f"- voxels.shape          : {after_voxels_shape}")
    #     print(f"- voxel_coords.shape    : {after_voxel_coords_shape}")
    #     print(f"- voxel_num_points.shape: {after_voxel_num_points_shape}")
    #     raise ValueError("Recomputed voxels have invalid shapes.")

    return batch_dict


def compute_bbox_pseudo_pose(bbox: np.ndarray) -> np.ndarray:
    """
    Compute the bbox's pseudo pose, which is the pose of transforming the axis
    aligned box centered at [0, 0, 0] to the current bbox.
    """
    theta = bbox[6]
    R = np.array(
        [
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    R = R.T
    t = bbox[:3]
    pseudo_pose = np.eye(4)
    pseudo_pose[:3, :3] = R
    pseudo_pose[:3, 3] = t
    return pseudo_pose


def copy_paste_nuscenes_to_kitti(
    dataloader: torch.utils.data.DataLoader,
    batch_dict: dict,
    src_domain: str,
    dst_style: str,
    dst_bbox_size: str,
):
    """
    Replace points in batch_dict with points from nuScenes dataset.

    Args:
        dataloader: The dataloader object used to run feature encoder and
            data processor.
        batch_dict: dict. Must at least contain "points".
            - The batch_dict will be modified in-place!
            - Must have "points".
            - Batch size must be 1, i.e. "points" must be (N, 4) and
              points[:, 0] must be all 0.
            - Must be in numpy.
        src_domain: str. {"kitti_pcd", "nuscenes_mesh", "nuscenes_pcd"}
            - "nuscenes_mesh": Use points from the reconstructed nuScenes mesh.
            - "nuscenes_pcd": Use points from the raw fg nuScenes point cloud.
        dst_style: str. {"kitti", "nuscenes"}
            - "kitti": Use KITTI lidar parameters.
            - "nuscenes": Use nuScenes lidar parameters.
        dst_bbox_size: str. {"kitti", "nuscenes"}
            - "kitti": Keep KITTI bbox size.
            - "nuscenes": Replace bbox to match nuScenes bbox size.

    Return:
        batch_dict: dict. The modified batch_dict.
    """
    # Sanity checks.
    if src_domain not in ["kitti_pcd", "nuscenes_mesh", "nuscenes_pcd"]:
        raise ValueError(f"Invalid src_domain: {src_domain}")
    if dst_style not in ["kitti", "nuscenes"]:
        raise ValueError(f"Invalid dst_style: {dst_style}")
    if dst_bbox_size not in ["kitti", "nuscenes"]:
        raise ValueError(f"Invalid dst_bbox_size: {dst_bbox_size}")
    if batch_dict["points"].ndim != 2 or batch_dict["points"].shape[1] != 4:
        raise ValueError("points must be (N, 4)")
    if not np.allclose(batch_dict["points"][:, 0], 0):
        raise ValueError("points[:, 0] must be all 0")

    # Strip the batch dimension.
    points = batch_dict["points"][:, 1:4]
    kitti_bboxes = batch_dict["gt_boxes"][0]  # (1, N, 8) -> (N, 8)

    # Copy and paste.
    lit_paths = get_lit_paths(
        data_version="v1",
        data_domain="nuscenes",
    )
    num_points_before = len(points)
    assert src_domain in {"kitti_pcd", "nuscenes_mesh"}
    fg_scene_paths = sorted(list(lit_paths.fg_dir.glob("*.pkl")))
    dst_bboxes = []
    for kitti_bbox in kitti_bboxes:
        # Randomly pick an fg_scene, then randomly pick a fg_object (mesh).
        fg_scene_path = np.random.choice(fg_scene_paths)
        fg_scene = FGScene.load(fg_scene_path)
        fb_object = fg_scene[np.random.randint(len(fg_scene))]

        # Get vertices and triangles.
        # The vertices's bbox shall be centered according to fg_object.
        vertices = np.copy(fb_object.mesh_vertices)
        triangles = np.copy(fb_object.mesh_triangles)
        if not np.allclose(
            (np.min(vertices, axis=0) + np.max(vertices, axis=0)) / 2,
            [0, 0, 0],
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError("Vertices must be centered at [0, 0, 0].")

        # Lidar pose is only used if the src_domain is nuscenes_mesh and visualization.
        lidar_pose = np.eye(4)
        lidar_pose[:3, 3] = np.array(dataloader.dataset.dataset_cfg["SHIFT_COOR"])

        # Compute dst_bbox, and scale vertices to match dst_bbox size.
        _, _, _, kitti_dx, kitti_dy, kitti_dz, _, _ = kitti_bbox
        mesh_dx, mesh_dy, mesh_dz = np.max(vertices, axis=0) - np.min(vertices, axis=0)
        if dst_bbox_size == "kitti":
            dst_bbox = np.copy(kitti_bbox)
            vertices = vertices * np.array(
                [
                    kitti_dx / mesh_dx,
                    kitti_dy / mesh_dy,
                    kitti_dz / mesh_dz,
                ]
            )
        elif dst_bbox_size == "nuscenes":
            dst_bbox = np.copy(kitti_bbox)
            # Change the extents of the dst_bbox to match the mesh.
            dst_bbox[3:6] = np.array([mesh_dx, mesh_dy, mesh_dz])
            # Raise the center of the dst_bbox, such that its "ground plane"
            # still touches the ground.
            dst_bbox[2] = kitti_bbox[2] - kitti_dz / 2 + mesh_dz / 2
        else:
            raise ValueError(f"Invalid dst_bbox_size: {dst_bbox_size}")
        dst_bboxes.append(dst_bbox)

        # Compute new points.
        if src_domain == "kitti_pcd":
            if dst_style != "kitti":
                raise ValueError("Only dst_style=kitti is supported.")
            if dst_bbox_size != "nuscenes":
                raise ValueError("Only dst_bbox_size=nuscenes is supported.")

            inside_indices = get_indices_inside_bbox(points, kitti_bbox)
            if len(inside_indices) == 0:
                new_points = np.zeros((0, 3), dtype=np.float32)
            else:
                inside_points = points[inside_indices]

                # Center inside_points.
                pseudo_pose_kitti = compute_bbox_pseudo_pose(kitti_bbox)
                inside_points = ct.transform.transform_points(
                    inside_points, np.linalg.inv(pseudo_pose_kitti)
                )

                # Scale inside_points, while keeping it centered.
                inside_points = inside_points * np.array(
                    [
                        mesh_dx / kitti_dx,
                        mesh_dy / kitti_dy,
                        mesh_dz / kitti_dz,
                    ]
                )

                # Put the scaled inside_points back to the dst pose.
                pseudo_pose_dst = compute_bbox_pseudo_pose(dst_bbox)
                inside_points = ct.transform.transform_points(
                    inside_points, pseudo_pose_dst
                )
                new_points = inside_points
        elif src_domain == "nuscenes_mesh":
            # Compute the pseudo pose, which is the pose of transforming the axis
            # aligned box centered at [0, 0, 0] to the bbox.
            pseudo_pose = compute_bbox_pseudo_pose(dst_bbox)

            # Put the vertices to the bbox.
            vertices = ct.transform.transform_points(vertices, pseudo_pose)

            # Raycast the mesh to get the new points.
            if dst_style == "kitti":
                lidar_intrinsics = KITTILidarIntrinsics()
            elif dst_style == "nuscenes":
                lidar_intrinsics = NuScenesLidarIntrinsics()
            else:
                raise ValueError(f"Invalid dst_style: {dst_style}")

            lidar = Lidar(intrinsics=lidar_intrinsics, pose=lidar_pose)
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            mesh.compute_vertex_normals()
            try:
                new_points, _ = g_raycast_engine.lidar_intersect_mesh(
                    lidar=lidar, mesh=mesh
                )
            except RuntimeError as e:
                import pickle

                mesh_vertices = np.asarray(mesh.vertices)
                mesh_triangles = np.asarray(mesh.triangles)
                data_dict = {
                    "lidar": lidar,
                    "mesh_vertices": mesh_vertices,
                    "mesh_triangles": mesh_triangles,
                }
                pkl_path = Path("raycast_error_data.pkl")
                with open(pkl_path, "wb") as f:
                    pickle.dump(data_dict, f)
                print(f"RuntimeError: {e}")
                print(f"Saved debug data to: {pkl_path}")

                import ipdb

                ipdb.set_trace()
        else:
            raise ValueError(f"Unimplemented src_domain: {src_domain}")

        # Cut: Remove points inside KITTI bbox.
        outside_indices = get_indices_outside_bboxes(points, kitti_bbox[None])
        points = points[outside_indices]

        # Copy-Paste: Paste the new_points into points.
        points = np.concatenate([points, new_points], axis=0)

        # Visualize.
        visualize = False
        if visualize:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            kitti_bbox_ls = bbox_to_lineset(kitti_bbox).paint_uniform_color([0, 0, 1])
            dst_bbox_ls = bbox_to_lineset(dst_bbox).paint_uniform_color([1, 0, 0])
            lidar_frame = ct.camera.create_camera_frustums(
                Ks=None, Ts=[np.linalg.inv(lidar_pose)], size=1
            )
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)
            o3d.visualization.draw_geometries(
                [pcd, kitti_bbox_ls, dst_bbox_ls, lidar_frame, axes]
            )

    num_points_after = len(points)
    print(f"Copy-pasted: {num_points_before} -> {num_points_after} points")

    # Update batch_dict.
    points = np.concatenate([np.zeros((len(points), 1)), points], axis=1)
    batch_dict["points"] = points

    dst_bboxes = np.array(dst_bboxes).reshape((-1, 8))  # (N, 8), N can be 0
    if dst_bboxes[None].shape != batch_dict["gt_boxes"].shape:
        raise ValueError(f"{dst_bboxes[None].shape} != {batch_dict['gt_boxes'].shape}")
    batch_dict["gt_boxes"] = dst_bboxes[None]

    # Recompute voxel features.
    batch_dict = recompute_voxels(dataloader=dataloader, batch_dict=batch_dict)

    return batch_dict


def main():
    pass


if __name__ == "__main__":
    main()
