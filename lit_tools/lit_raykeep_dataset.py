import argparse
import time

import camtools as ct
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from pykdtree.kdtree import KDTree
from tqdm import tqdm

from lit.containers.scene import Scene
from lit.containers.sim_scene import SimScene
from lit.lidar import (
    Lidar,
    NuScenesLidarIntrinsics,
    WaymoLidarIntrinsics,
)
from lit.ray_keeper import GBMRayKeeper


def find_closest_directions(query_dirs, model_dirs):
    """
    Find the index of the closest direction in model_dirs for each direction in
    query_dirs using pykdtree.

    By law of cosines, d = sqrt(a^2 + b^2 - 2 * a * b * cos(theta)). Computing
    the minimal d is equivalent to finding the minimal theta between two
    directions.

    Args:
        query_dirs: [N, 3] array, N unit vectors representing directions.
        model_dirs: [M, 3] array, M unit vectors representing standard directions.

    Returns:
        indices: [N] array, indices of the closest direction in model_dirs for each
        direction in query_dirs.
    """
    # Check shape.
    assert query_dirs.ndim == 2 and query_dirs.shape[1] == 3
    assert model_dirs.ndim == 2 and model_dirs.shape[1] == 3

    # Normalize.
    query_dirs = query_dirs / np.linalg.norm(query_dirs, axis=1, keepdims=True)
    model_dirs = model_dirs / np.linalg.norm(model_dirs, axis=1, keepdims=True)

    # Create KDTree from model directions
    start = time.time()
    kdtree = KDTree(model_dirs)
    print(f"KDTree creation time: {time.time() - start:.3f}s")

    # Query the KDTree for the nearest neighbor of each query direction
    start = time.time()
    _, indices = kdtree.query(query_dirs, k=1)
    print(f"KDTree query time: {time.time() - start:.3f}s")

    return indices


def points_to_range_get_indices(points: np.ndarray, lidar: Lidar):
    """
    Compute the row and column indices in the range image for multiple points.

    Args:
        points: (N, 3) array, points in the lidar frame.
        lidar: Lidar object.

    Returns:
        Two arrays (row_indices, col_indices) indicating the positions of the
        points in the range image.
    """
    # Transform the points to the lidar's coordinate system.
    lidar_center = ct.convert.pose_to_C(lidar.pose)
    point_dirs = points - lidar_center
    point_dists = np.linalg.norm(point_dirs, axis=1, keepdims=True)
    point_dirs = point_dirs / point_dists

    # Get lidar directions.
    lidar_rays = lidar.get_rays()
    lidar_dirs = lidar_rays[:, 3:]

    # Find the closest directions for the points.
    ray_indices = find_closest_directions(
        query_dirs=point_dirs,
        model_dirs=lidar_dirs,
    )

    # Compute the row and column indices.
    H, W = lidar.intrinsics.vertical_res, lidar.intrinsics.horizontal_res
    row_indices, col_indices = np.divmod(ray_indices, W)

    return row_indices, col_indices


def points_to_range_with_nn(points: np.ndarray, lidar: Lidar):
    """
    Convert points to a range image with nearest neighboring ray. Smallest
    distance will be used when multiple points map to the same pixel. Points
    outside the lidar's field of view are considered out-of-bound and ignored.

    Args:
        points: (N, 3) array, points in the lidar frame.
        lidar: Lidar object.
    """
    # Constants for lidar's field of view (in radians).
    fov_up = np.radians(lidar.intrinsics.fov_up)
    fov_down = np.radians(lidar.intrinsics.fov_down)

    # Transform the points to the lidar's coordinate system.
    lidar_center = ct.convert.pose_to_C(lidar.pose)
    point_dirs = points - lidar_center
    point_dists = np.linalg.norm(point_dirs, axis=1)
    point_dirs /= point_dists[:, None]

    # Filter by "up-down" angles.
    xy_dists = np.linalg.norm(point_dirs[:, :2], axis=1)
    z_dists = point_dirs[:, 2]
    vertical_angles = np.arctan2(z_dists, xy_dists)
    valid_indices = np.where(
        (vertical_angles >= -fov_down) & (vertical_angles <= fov_up)
    )[0]
    print(f"Filter by angles: {len(points)} -> {len(valid_indices)}")
    point_dirs = point_dirs[valid_indices]
    point_dists = point_dists[valid_indices]

    # Filter by max_range.
    valid_indices = np.where(point_dists <= lidar.intrinsics.max_range)[0]
    print(f"Filter by max_range: {len(point_dirs)} -> {len(valid_indices)}")
    point_dirs = point_dirs[valid_indices]
    point_dists = point_dists[valid_indices]

    # Lidar directions are always ordered.
    lidar_rays = lidar.get_rays()
    lidar_dirs = lidar_rays[:, 3:]

    # For each valid point direction, find the closest direction in lidar_dirs.
    ray_indices = find_closest_directions(
        query_dirs=point_dirs,
        model_dirs=lidar_dirs,
    )

    # Create a range image.
    H, W = lidar.intrinsics.vertical_res, lidar.intrinsics.horizontal_res
    im_range = np.full((H, W), np.inf, dtype=np.float32)

    # Map 1D indices to 2D (row, column) indices.
    row_indices, col_indices = np.divmod(ray_indices, W)

    # Update the range image, keeping the smallest distance for each pixel.
    for row, col, dist in zip(row_indices, col_indices, point_dists):
        im_range[row, col] = min(im_range[row, col], dist)

    # Replace np.inf with zeros (or any other appropriate value).
    im_range[im_range == np.inf] = 0

    return im_range


def points_to_range_with_fov(points: np.ndarray, lidar: Lidar):
    """
    Convert lidar points to panoramic frame.
    Lidar points are in local coordinates.

    Args:
        points: (N, 3), float32, points in the lidar frame.
        lidar: Lidar object.

    Return:
        pano: (H, W), float32, panoramic image representing the depth.
    """
    fov_up = lidar.intrinsics.fov_up
    fov_down = lidar.intrinsics.fov_down
    total_fov = fov_up + fov_down
    lidar_H = lidar.intrinsics.vertical_res
    lidar_W = lidar.intrinsics.horizontal_res
    max_depth = lidar.intrinsics.max_range

    # Compute distances to lidar center.
    lidar_center = ct.convert.pose_to_C(lidar.pose)
    local_points = points - lidar_center
    dists = np.linalg.norm(local_points, axis=1)

    # Fill pano.
    pano = np.zeros((lidar_H, lidar_W))
    for local_point, dist in zip(local_points, dists):
        # Check max depth.
        if dist >= max_depth:
            continue

        x, y, z = local_point
        beta = np.pi - np.arctan2(y, x)
        alpha = np.arctan2(z, np.sqrt(x**2 + y**2)) + np.radians(fov_down)
        c = int(round(beta / (2 * np.pi / lidar_W)))
        r = int(round(lidar_H - alpha / (np.radians(total_fov) / lidar_H)))

        # Check out-of-bounds.
        if r >= lidar_H or r < 0 or c >= lidar_W or c < 0:
            continue

        # Set to min dist if not set or if current dist is smaller.
        if pano[r, c] == 0.0 or pano[r, c] > dist:
            pano[r, c] = dist

    return pano


def points_to_range_with_angles(points: np.ndarray, lidar: Lidar):
    """
    Convert lidar points to panoramic frame using specified vertical angles.
    Lidar points are in local coordinates.

    Args:
        points: (N, 3), float32, points in the lidar frame.
        lidar: Lidar object.

    Return:
        pano: (H, W), float32, panoramic image representing the depth.
    """
    lidar_H = lidar.intrinsics.vertical_res
    lidar_W = lidar.intrinsics.horizontal_res
    max_depth = lidar.intrinsics.max_range
    vertical_degrees = np.radians(lidar.intrinsics.vertical_degrees)

    # Compute distances to lidar center.
    lidar_center = ct.convert.pose_to_C(lidar.pose)
    local_points = points - lidar_center
    dists = np.linalg.norm(local_points, axis=1)

    # Fill pano.
    pano = np.zeros((lidar_H, lidar_W))
    for local_point, dist in zip(local_points, dists):
        # Check max depth.
        if dist >= max_depth:
            continue

        x, y, z = local_point
        beta = np.pi - np.arctan2(y, x)
        alpha = np.arctan2(z, np.sqrt(x**2 + y**2))

        c = int(round(beta / (2 * np.pi / lidar_W)))
        r = np.argmin(np.abs(vertical_degrees - alpha))

        # Check out-of-bounds.
        if r >= lidar_H or r < 0 or c >= lidar_W or c < 0:
            continue

        # Set to min dist if not set or if current dist is smaller.
        if pano[r, c] == 0.0 or pano[r, c] > dist:
            pano[r, c] = dist

    return pano


def points_to_range_indices(points: np.ndarray, lidar: Lidar):
    """
    Using the "angles" method.

    Convert lidar points to indices in the panoramic frame using specified
    vertical angles. Lidar points are in local coordinates.

    Args:
        points: (N, 3), float32, points in the lidar frame.
        lidar: Lidar object.

    Return:
        row_indices: (N,), int, row indices in the panoramic frame.
        col_indices: (N,), int, column indices in the panoramic frame.
    """
    lidar_H = lidar.intrinsics.vertical_res
    lidar_W = lidar.intrinsics.horizontal_res
    max_depth = lidar.intrinsics.max_range
    vertical_degrees = np.radians(lidar.intrinsics.vertical_degrees)

    # Transform the points to the lidar's coordinate system.
    lidar_center = ct.convert.pose_to_C(lidar.pose)
    local_points = points - lidar_center

    # Preallocate arrays for indices.
    row_indices = np.zeros(len(points), dtype=int)
    col_indices = np.zeros(len(points), dtype=int)

    for i, local_point in enumerate(local_points):
        x, y, z = local_point
        dist = np.linalg.norm(local_point)

        # Check max depth.
        if dist >= max_depth:
            continue

        beta = np.pi - np.arctan2(y, x)
        alpha = np.arctan2(z, np.sqrt(x**2 + y**2))

        col = int(round(beta / (2 * np.pi / lidar_W)))
        row = np.argmin(np.abs(vertical_degrees - alpha))

        # Check out-of-bounds.
        if row >= lidar_H or row < 0 or col >= lidar_W or col < 0:
            continue

        row_indices[i] = row
        col_indices[i] = col

    return row_indices, col_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--style",
        type=str,
        required=True,
        choices=("waymo", "nuscenes"),
        help="To create raykeep dataset, the src and dst style are the same.",
    )
    args = parser.parse_args()

    if args.style == "waymo":
        scene_dir = lit_paths.waymo.scene_dir
        sim_dir = lit_paths.waymo.to_waymo_sim_dir
        raykeep_dir = lit_paths.waymo.to_nuscenes_raykeep
        lidar_intrinsics = WaymoLidarIntrinsics()
    elif args.style == "nuscenes":
        scene_dir = lit_paths.nuscenes.scene_dir
        sim_dir = lit_paths.nuscenes.to_nuscenes_sim_dir
        raykeep_dir = lit_paths.nuscenes.to_nuscenes_raykeep
        lidar_intrinsics = NuScenesLidarIntrinsics()
    else:
        raise ValueError(f"Unknown style: {args.style}")

    # Check number of simulation scenes.
    sim_scene_paths = list(sorted(sim_dir.glob("*.pkl")))
    scene_names = [p.stem for p in sim_scene_paths]
    print(f"Found {len(scene_names)} simulation scenes.")

    # Randomly pick num_scenes scenes.
    np.random.seed(0)
    num_scenes = 50
    scene_names = np.random.choice(
        scene_names,
        size=num_scenes,
        replace=False,
    )

    # Inputs: dir_x, dir_y, dir_z, dist, incident_angle.
    # Output: 1 for keep, 0 for drop.
    network_inputs = []  # (N, 5)
    network_outputs = []  # (N,)
    for scene_name in tqdm(
        scene_names,
        desc="Collecting scenes",
    ):
        scene_path = scene_dir / f"{scene_name}.pkl"
        sim_scene_path = sim_dir / f"{scene_name}.pkl"
        scene = Scene.load(path=scene_path)
        sim_scene = SimScene.load(path=sim_scene_path)
        assert len(scene) == len(sim_scene)

        for rel_frame, sim_frame in tqdm(
            zip(scene, sim_scene),
            desc="Collecting frames",
            total=len(scene),
        ):
            assert sim_frame.frame_index == rel_frame.frame_index
            assert np.allclose(sim_frame.frame_pose, rel_frame.frame_pose)

            # Assumeing vehicle coordinates (vehicle is at (0, 0, 0)).
            lidar_pose = rel_frame.lidar_to_vehicle_poses[0]
            lidar_center = ct.convert.pose_to_C(lidar_pose)
            lidar = Lidar(
                intrinsics=lidar_intrinsics,
                pose=lidar_pose,
            )

            # Unpack data.
            H, W = lidar.intrinsics.vertical_res, lidar.intrinsics.horizontal_res
            sim_points = sim_frame.local_points
            rel_points = rel_frame.points

            sim_im_range = points_to_range_with_angles(
                points=sim_points,
                lidar=lidar,
            )
            rel_im_range = points_to_range_with_angles(
                points=rel_points,
                lidar=lidar,
            )

            # Find indices of sim rays in range image.
            row_indices, col_indices = points_to_range_indices(
                points=sim_points,
                lidar=lidar,
            )
            im = np.zeros((H, W), dtype=np.float32)
            im[row_indices, col_indices] = 1

            # Inputs (M, 5)
            # Each line is: dir_x, dir_y, dir_z, dist, incident_angle.
            dirs = sim_points - lidar_center
            dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
            dists = sim_im_range[row_indices, col_indices]
            incident_angles = sim_frame.incident_angles
            inputs = np.concatenate(
                (dirs, dists[:, None], incident_angles[:, None]), axis=1
            )
            network_inputs.append(inputs)

            # Output: (M,)
            # 1 for keep, 0 for drop.
            ray_keeps = rel_im_range[row_indices, col_indices]
            ray_keeps[ray_keeps > 0] = 1
            network_outputs.append(ray_keeps)

            # Replace points with raykeep points for debugging.
            overwrite_with_dropped_points = False
            if overwrite_with_dropped_points:
                ray_keeper = GBMRayKeeper.load()
                keep_probs = ray_keeper.predict(inputs)
                keep_masks = keep_probs > 0.5
                print(f"Keep ratio: {np.mean(keep_masks)}")
                sim_points = sim_points[keep_masks]

            visualize = False
            if visualize:
                # Create range images.
                sim_im_range_with_fov = points_to_range_with_fov(
                    points=sim_points,
                    lidar=lidar,
                )
                sim_im_range_with_angles = points_to_range_with_angles(
                    points=sim_points,
                    lidar=lidar,
                )
                sim_im_range_with_nn = points_to_range_with_nn(
                    points=sim_points,
                    lidar=lidar,
                )

                rel_im_range_with_fov = points_to_range_with_fov(
                    points=rel_points,
                    lidar=lidar,
                )
                rel_im_range_with_angles = points_to_range_with_angles(
                    points=rel_points,
                    lidar=lidar,
                )
                rel_im_range_with_nn = points_to_range_with_nn(
                    points=rel_points,
                    lidar=lidar,
                )
                num_sim_with_fov = np.count_nonzero(sim_im_range_with_fov)
                num_sim_with_angles = np.count_nonzero(sim_im_range_with_angles)
                num_sim_with_nn = np.count_nonzero(sim_im_range_with_nn)
                num_rel_with_fov = np.count_nonzero(rel_im_range_with_fov)
                num_rel_with_angles = np.count_nonzero(rel_im_range_with_angles)
                num_rel_with_nn = np.count_nonzero(rel_im_range_with_nn)

                # Plot sim to top, rel to down. Add title for each.
                fig, axes = plt.subplots(6, 1)
                axes[0].imshow(sim_im_range_with_fov)
                axes[0].set_title(f"Sim with fov ({num_sim_with_fov} rays)")
                axes[1].imshow(sim_im_range_with_angles)
                axes[1].set_title(
                    f"Sim with explicit angles ({num_sim_with_angles} rays)"
                )
                axes[2].imshow(sim_im_range_with_nn)
                axes[2].set_title(
                    f"Sim with nearest neighbors ({num_sim_with_nn} rays)"
                )

                axes[3].imshow(rel_im_range_with_fov)
                axes[3].set_title(f"Rel with fov ({num_rel_with_fov} rays)")
                axes[4].imshow(rel_im_range_with_angles)
                axes[4].set_title(
                    f"Rel with explicit angles ({num_rel_with_angles} rays)"
                )
                axes[5].imshow(rel_im_range_with_nn)
                axes[5].set_title(
                    f"Rel with nearest neighbors ({num_rel_with_nn} rays)"
                )

                for ax in axes:
                    ax.axis("off")
                plt.tight_layout()
                plt.show()

            visualize = False
            if visualize:
                rel_pcd = o3d.geometry.PointCloud()
                rel_pcd.points = o3d.utility.Vector3dVector(rel_points)
                rel_pcd.paint_uniform_color([0, 0, 1])
                sim_pcd = o3d.geometry.PointCloud()
                sim_pcd.points = o3d.utility.Vector3dVector(sim_points)
                sim_pcd.paint_uniform_color([1, 0, 0])

                axes = o3d.geometry.TriangleMesh.create_coordinate_frame()

                o3d.visualization.draw_geometries([rel_pcd, sim_pcd, axes])

    # Save to disk.
    network_inputs = np.concatenate(network_inputs, axis=0).astype(np.float32)
    network_outputs = np.concatenate(network_outputs, axis=0).astype(np.float32)
    raykeep_dir.mkdir(parents=True, exist_ok=True)
    raykeep_path = raykeep_dir / "raykeep_data.npz"
    np.savez_compressed(
        raykeep_path,
        network_inputs=network_inputs,
        network_outputs=network_outputs,
    )


if __name__ == "__main__":
    # test_range_image()
    main()
