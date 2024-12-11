"""
This implements lidar simulation on NuScenes or Waymo.
"""

import argparse
import copy
import json
import pickle
import time

import camtools as ct
import numpy as np
import open3d as o3d
from tqdm import tqdm

from lit.containers.fg_scene import FGScene
from lit.containers.scene import Scene
from lit.containers.sim_frame import SimFrame
from lit.containers.sim_scene import SimScene
from lit.fg_reconstructor import FGReconstructor
from lit.lidar import (
    KITTILidarIntrinsics,
    Lidar,
    NuScenesLidarIntrinsics,
    NuScenesVanillaLidarIntrinsics,
    WaymoLidarIntrinsics,
)
from lit.path_utils import LitPaths, get_lit_paths
from lit.raycast_engine import RaycastEngine
from lit.raycast_engine_cpu import RaycastEngineCPU
from lit.raycast_engine_gpu import RaycastEngineGPU
from lit.recon_utils import (
    bbox_to_corners,
    bboxes_to_lineset,
    incident_angles_to_colors,
)


class CanonicalMeshCache:
    """
    Shared mesh for ablation study.
    """

    def __init__(self, max_capacity: int):
        self.max_capacity = max_capacity
        self.list_vertices = []
        self.list_triangles = []

    def is_full(self):
        return len(self.list_vertices) >= self.max_capacity

    def insert(self, vertices, triangles):
        self.list_vertices.append(vertices)
        self.list_triangles.append(triangles)

    def get_random_mesh(self):
        if not self.is_full():
            raise ValueError(
                f"Cache is not full yet: {len(self.list_vertices)} "
                f"out of {self.max_capacity} filled."
            )
        idx = np.random.randint(len(self.list_vertices))
        vertices = self.list_vertices[idx]
        triangles = self.list_triangles[idx]

        return vertices, triangles


def run_sim(
    data_version: str,
    src_style: str,
    dst_style: str,
    lit_paths: LitPaths,
    scene_name: str,
    enable_scaling: bool,
    raycast_engine: RaycastEngine,
    skip_existing: bool = False,
    dry_run: bool = False,
    mesh_cache: CanonicalMeshCache = None,
    ablation_foreground_only: bool = False,
):
    # Get destination paths.
    if dst_style == "waymo":
        sim_scene_dir = lit_paths.sim_waymo_dir / scene_name
        lidar_intrinsics = WaymoLidarIntrinsics()
    elif dst_style == "nuscenes":
        sim_scene_dir = lit_paths.sim_nuscenes_dir / scene_name
        lidar_intrinsics = NuScenesLidarIntrinsics()
    elif dst_style == "nuscenes_vanilla":
        # Special case for vanilla nuscenes lidar without statistical modeling.
        sim_scene_dir = lit_paths.lit_data_root / "sim_nuscenes_vanilla" / scene_name
        lidar_intrinsics = NuScenesVanillaLidarIntrinsics()
    elif dst_style == "kitti":
        sim_scene_dir = lit_paths.sim_kitti_dir / scene_name
        lidar_intrinsics = KITTILidarIntrinsics()
    else:
        raise NotImplementedError(f"Unknown dst_style: {dst_style}")

    # Get src->dst scaling factors.
    # data/stats/src_to_dst_bbox_scales.json
    if enable_scaling:
        scales_path = lit_paths.data_root / "stats" / "src_to_dst_bbox_scales.json"
        with open(scales_path, "r") as f:
            scales_dict = json.load(f)
        src_to_dst_scales = np.array(
            scales_dict[f"{src_style}_to_{dst_style}_bbox_scale"]
        )
        print(f"Loaded src_to_dst_scales: {src_to_dst_scales}")
    else:
        src_to_dst_scales = None
        print("src_to_dst_scales is set to None.")

    # Skip if sim_dir exists (we won't check the content of sim_dir)
    if skip_existing and sim_scene_dir.exists():
        print(f"Skipping {scene_name} as it already exists.")
        return

    # Load scene.
    scene_path = lit_paths.scene_dir / f"{scene_name}.pkl"
    scene = Scene.load(scene_path)
    num_frames = len(scene)

    # Load fg groups.
    fg_path = lit_paths.fg_dir / f"{scene_name}.pkl"
    fg_scene = FGScene.load(fg_path)

    if mesh_cache is not None:
        for fg_object in fg_scene:
            if not mesh_cache.is_full():
                # Insert mesh into cache.
                vertices = fg_object.mesh_vertices
                triangles = fg_object.mesh_triangles
                mesh_cache.insert(vertices=vertices, triangles=triangles)
            else:
                # Retrieve random mesh from cache.
                vertices, triangles = mesh_cache.get_random_mesh()

                # Get avg_centered_corners for scaling.
                fused_centered_corners = []
                for fg_box in fg_object.fg_boxes:
                    pseudo_pose = fg_box.compute_local_pseudo_pose()
                    pseudo_T = ct.convert.pose_to_T(pseudo_pose)
                    local_corners = bbox_to_corners(fg_box.local_bbox)
                    centered_corners = ct.transform.transform_points(
                        local_corners, pseudo_T
                    )
                    fused_centered_corners.append(centered_corners)
                fused_centered_corners = np.array(fused_centered_corners)
                avg_centered_corners = np.mean(fused_centered_corners, axis=0)

                # Scale vertices with FGReconstructor.resize_mesh_to_fit_bbox.
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.triangles = o3d.utility.Vector3iVector(triangles)
                mesh = FGReconstructor.resize_mesh_to_fit_bbox(
                    mesh=mesh,
                    axis_aligned_centered_corners=avg_centered_corners,
                )
                new_vertices = np.asarray(mesh.vertices)
                new_triangles = np.asarray(mesh.triangles)

                # Apply replacement.
                fg_object.mesh_vertices = new_vertices
                fg_object.mesh_triangles = new_triangles

    # Load bg mesh.
    bg_mesh_path = lit_paths.bg_dir / f"{scene_name}.ply"
    bg_mesh = o3d.io.read_triangle_mesh(str(bg_mesh_path))
    if len(bg_mesh.vertices) == 0:
        print(f"[Warning] Empty bg mesh: {bg_mesh_path}, skipping.")

    # Simulate frame by frame.
    raycast_times = []
    sim_scene = SimScene()
    for frame_index in tqdm(
        range(num_frames), desc="Simulating frames", total=num_frames
    ):
        # Get poses.
        frame = scene[frame_index]
        frame_pose = frame.frame_pose
        lidar_to_vehicle_poses = frame.lidar_to_vehicle_poses

        # Get mesh.
        fg_mesh = fg_scene.get_frame_mesh(
            frame_index=frame_index,
            src_to_dst_scales=src_to_dst_scales,
        )
        if ablation_foreground_only:
            fused_mesh = fg_mesh
        else:
            fused_mesh = fg_mesh + bg_mesh

        # Get lidar.
        lidar_pose = frame_pose @ lidar_to_vehicle_poses[0]
        lidar = Lidar(intrinsics=lidar_intrinsics, pose=lidar_pose)

        # Get points.
        start_time = time.time()

        # Create a dummy mesh if mesh is empty.
        if len(fused_mesh.vertices) == 0:
            fused_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.0001)

        cast_result = raycast_engine.lidar_intersect_mesh(
            lidar=lidar,
            mesh=fused_mesh,
        )

        if isinstance(cast_result, tuple):
            points, incident_angles = cast_result
        else:
            points = cast_result
            incident_angles = np.zeros(len(points))
        raycast_times.append(time.time() - start_time)

        # Convert points from global to local coordinates with frame_pose.
        local_points = ct.transform.transform_points(
            points, ct.convert.pose_to_T(frame_pose)
        )

        # Get local bboxes.
        local_bboxes = fg_scene.get_frame_local_bboxes(
            frame_index=frame_index,
            src_to_dst_scales=src_to_dst_scales,
        )

        # Append.
        sim_frame = SimFrame(
            frame_index=frame_index,
            frame_pose=frame_pose,
            local_points=local_points,
            local_bboxes=local_bboxes,
            incident_angles=incident_angles,
        )
        sim_scene.append_frame(sim_frame)

        # Visualize.
        visualize = False
        if visualize:
            fg_frame_ls_unscaled = fg_scene.get_frame_ls(
                frame_index=frame_index,
            )
            fg_frame_ls_unscaled.paint_uniform_color([0, 0, 1])
            fg_frame_ls_scaled = bboxes_to_lineset(
                bboxes=local_bboxes,
                frame_pose=frame_pose,
            )
            fg_frame_ls_scaled.paint_uniform_color([1, 0, 0])

            fused_mesh.compute_vertex_normals()
            points_pcd = o3d.geometry.PointCloud()
            points_pcd.points = o3d.utility.Vector3dVector(points)
            colors = incident_angles_to_colors(incident_angles)
            points_pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries(
                [
                    fg_frame_ls_unscaled,
                    fg_frame_ls_scaled,
                    fused_mesh,
                    points_pcd,
                ]
            )

    if dry_run:
        total_raycast_time = sum(raycast_times)
        print(f"[Dry run] scene raycast time: {total_raycast_time}")
        return

    # Save as frames in local coordinates.
    sim_scene_dir.mkdir(parents=True, exist_ok=True)
    sim_scene.save_sim_frames(sim_scene_dir=sim_scene_dir)


def main():
    """
    # Example commands:
    CUDA_VISIBLE_DEVICES=0 python lit_04_sim.py --src_style waymo --dst_style kitti --skip_existing --data_version v0
    CUDA_VISIBLE_DEVICES=1 python lit_04_sim.py --src_style waymo --dst_style kitti --skip_existing --data_version v0 --reverse
    CUDA_VISIBLE_DEVICES=0 python lit_04_sim.py --src_style waymo --dst_style kitti --skip_existing --data_version v0 --shuffle
    CUDA_VISIBLE_DEVICES=1 python lit_04_sim.py --src_style waymo --dst_style kitti --skip_existing --data_version v0 --shuffle

    CUDA_VISIBLE_DEVICES=0 python lit_04_sim.py --src_style waymo --dst_style nuscenes --skip_existing --data_version v0
    CUDA_VISIBLE_DEVICES=1 python lit_04_sim.py --src_style waymo --dst_style nuscenes --skip_existing --data_version v0 --reverse
    CUDA_VISIBLE_DEVICES=0 python lit_04_sim.py --src_style waymo --dst_style nuscenes --skip_existing --data_version v0 --shuffle
    CUDA_VISIBLE_DEVICES=1 python lit_04_sim.py --src_style waymo --dst_style nuscenes --skip_existing --data_version v0 --shuffle

    CUDA_VISIBLE_DEVICES=0 python lit_04_sim.py --src_style nuscenes --dst_style kitti --skip_existing --data_version v0
    CUDA_VISIBLE_DEVICES=1 python lit_04_sim.py --src_style nuscenes --dst_style kitti --skip_existing --data_version v0 --reverse

    # Ablation
    CUDA_VISIBLE_DEVICES=0 python lit_04_sim.py --src_style nuscenes --dst_style kitti --skip_existing --data_version v5 --ablation_num_shared_meshes 1
    CUDA_VISIBLE_DEVICES=1 python lit_04_sim.py --src_style nuscenes --dst_style kitti --skip_existing --data_version v5 --ablation_num_shared_meshes 1 --reverse
    CUDA_VISIBLE_DEVICES=0 python lit_04_sim.py --src_style nuscenes --dst_style kitti --skip_existing --data_version v5 --ablation_num_shared_meshes 1 --shuffle
    CUDA_VISIBLE_DEVICES=0 python lit_04_sim.py --src_style nuscenes --dst_style kitti --skip_existing --data_version v6 --ablation_num_shared_meshes 50
    CUDA_VISIBLE_DEVICES=1 python lit_04_sim.py --src_style nuscenes --dst_style kitti --skip_existing --data_version v6 --ablation_num_shared_meshes 50 --reverse
    CUDA_VISIBLE_DEVICES=0 python lit_04_sim.py --src_style nuscenes --dst_style kitti --skip_existing --data_version v6 --ablation_num_shared_meshes 50 --shuffle
    CUDA_VISIBLE_DEVICES=1 python lit_04_sim.py --src_style nuscenes --dst_style kitti --skip_existing --data_version v6 --ablation_num_shared_meshes 50 --shuffle
    CUDA_VISIBLE_DEVICES=0 python lit_04_sim.py --src_style nuscenes --dst_style kitti --skip_existing --data_version v7 --ablation_foreground_only

    # Simulation for visualization
    python lit_04_sim.py --src_style waymo --dst_style nuscenes --data_version v8
    python lit_04_sim.py --src_style waymo --dst_style kitti --data_version v8

    # Self-translation
    CUDA_VISIBLE_DEVICES=0 python lit_04_sim.py --src_style waymo --dst_style waymo --skip_existing --data_version v9
    CUDA_VISIBLE_DEVICES=0 python lit_04_sim.py --src_style waymo --dst_style waymo --skip_existing --data_version v9 --reverse
    CUDA_VISIBLE_DEVICES=1 python lit_04_sim.py --src_style waymo --dst_style waymo --skip_existing --data_version v9 --shuffle
    CUDA_VISIBLE_DEVICES=1 python lit_04_sim.py --src_style waymo --dst_style waymo --skip_existing --data_version v9 --shuffle

    CUDA_VISIBLE_DEVICES=1 python lit_04_sim.py --src_style nuscenes --dst_style nuscenes --skip_existing --data_version v9
    CUDA_VISIBLE_DEVICES=1 python lit_04_sim.py --src_style nuscenes --dst_style nuscenes --skip_existing --data_version v9 --reverse
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_style",
        type=str,
        required=True,
        choices=("waymo", "nuscenes"),
    )
    parser.add_argument(
        "--dst_style",
        type=str,
        choices=["waymo", "nuscenes", "kitti", "nuscenes_vanilla"],
        required=True,
        help="lidar style, choose from nuscenes, kitti, waymo",
    )
    parser.add_argument(
        "--data_version",
        type=str,
        required=True,
        help="Version of the data, which determines the lit_paths.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the order of scenes.",
    )
    parser.add_argument(
        "--enable_scaling",
        action="store_true",
        help="Whether to enable src->dst shape size scaling.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse the order of scenes.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run without saving.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip existing scenes that have been processed.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU.",
    )
    parser.add_argument(
        "--ablation_num_shared_meshes",
        type=int,
        default=0,
        help="Number of shared meshes for ablation study.",
    )
    parser.add_argument(
        "--ablation_foreground_only",
        action="store_true",
        help="Only simulate foreground.",
    )
    parser.add_argument(
        "--scene_index",
        type=int,
        default=None,
        help="Scene index, when specified, only process this scene",
    )

    args = parser.parse_args()

    # Get project paths.
    lit_paths = get_lit_paths(
        data_version=args.data_version,
        data_domain=args.src_style,
    )

    # Select scenes if scene_index is specified.
    scene_names = copy.copy(lit_paths.scene_names)
    if args.scene_index is not None:
        assert isinstance(args.scene_index, int)
        assert (
            0 <= args.scene_index < len(scene_names)
        ), f"{args.scene_index} not in [0, {len(scene_names)})"
        print(
            f"Only extracting scene_index={args.scene_index}: "
            f"{scene_names[args.scene_index]}"
        )
        scene_names = [scene_names[args.scene_index]]

    # Find valid scene names that have both fg and bg.
    valid_scene_names = []
    for scene_name in scene_names:
        fg_path = lit_paths.fg_dir / f"{scene_name}.pkl"
        bg_path = lit_paths.bg_dir / f"{scene_name}.ply"
        if fg_path.exists() and bg_path.exists():
            valid_scene_names.append(scene_name)
    print(f"- # scenes: {len(scene_names)}")
    print(f"- # valid scenes: {len(valid_scene_names)}")
    scene_names = valid_scene_names

    if args.reverse:
        scene_names = scene_names[::-1]
        print("Reversed valid_scene_names")

    if args.shuffle:
        shuffle_indices = np.random.permutation(len(scene_names))
        scene_names = [scene_names[i] for i in shuffle_indices]
        print("Shuffled valid_scene_names")

    if args.ablation_num_shared_meshes > 0:
        mesh_cache = CanonicalMeshCache(max_capacity=args.ablation_num_shared_meshes)
    else:
        mesh_cache = None

    # Run simulation.
    if args.cpu:
        raycast_engine = RaycastEngineCPU()
    else:
        raycast_engine = RaycastEngineGPU()
    for scene_name in tqdm(scene_names, desc="Simulating scenes"):
        run_sim(
            data_version=args.data_version,
            src_style=args.src_style,
            dst_style=args.dst_style,
            lit_paths=lit_paths,
            scene_name=scene_name,
            raycast_engine=raycast_engine,
            enable_scaling=args.enable_scaling,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
            mesh_cache=mesh_cache,
            ablation_foreground_only=args.ablation_foreground_only,
        )


if __name__ == "__main__":
    main()
