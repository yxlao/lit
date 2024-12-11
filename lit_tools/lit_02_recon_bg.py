import argparse
import copy
from types import SimpleNamespace

import open3d as o3d
from tqdm import tqdm

from lit.bg_reconstructor import BGReconstructor
from lit.containers.scene import Scene
from lit.path_utils import LitPaths, get_lit_paths
from lit.recon_utils import remove_statistical_outlier

_waymo_args = SimpleNamespace()
_waymo_args.skip_every_n_frames = 2
_waymo_args.enabled_lidars = (0,)
_waymo_args.expand_box_ratio = 1.0
_waymo_args.raise_bbox = 0.4
_waymo_args.nksr_voxel_size = 0.25
_waymo_args.nksr_chunked = False
_waymo_args.per_frame_rso_nb_neighbors = 0
_waymo_args.per_frame_rso_std_ratio = 0.0
_waymo_args.rso_nb_neighbors = 0
_waymo_args.rso_std_ratio = 0.0

_nuscenes_args = SimpleNamespace()
_nuscenes_args.skip_every_n_frames = 1
_nuscenes_args.enabled_lidars = (0,)
_nuscenes_args.expand_box_ratio = 1.05
_nuscenes_args.raise_bbox = 0.0
_nuscenes_args.nksr_voxel_size = 0.25
_nuscenes_args.nksr_chunked = False
_nuscenes_args.per_frame_rso_nb_neighbors = 0
_nuscenes_args.per_frame_rso_std_ratio = 0.0
_nuscenes_args.rso_nb_neighbors = 0
_nuscenes_args.rso_std_ratio = 0.0


def recon_bg(
    scene_name: str,
    lit_paths: LitPaths,
    skip_existing=False,
    dray_run=False,
):
    if lit_paths.data_domain == "waymo":
        args = _waymo_args
    elif lit_paths.data_domain == "nuscenes":
        args = _nuscenes_args
    else:
        raise ValueError(f"Unknown dataset type: {lit_paths.data_domain}")

    # Paths of input and output.
    scene_path = lit_paths.scene_dir / f"{scene_name}.pkl"
    mesh_path = lit_paths.bg_dir / f"{scene_name}.ply"

    # Skip existing.
    if skip_existing and mesh_path.exists():
        print(f"Skipped {mesh_path}.")
        return

    # Load.
    scene = Scene.load(scene_path)
    scene.sample_by_indices(range(0, len(scene), args.skip_every_n_frames))
    bg_data = scene.extract_bg(
        enabled_lidars=args.enabled_lidars,
        remove_foreground=True,
        raise_bbox=args.raise_bbox,
        expand_box_ratio=args.expand_box_ratio,
        per_frame_rso_nb_neighbors=args.per_frame_rso_nb_neighbors,
        per_frame_rso_std_ratio=args.per_frame_rso_std_ratio,
    )
    points = bg_data["points"]
    lidar_centers = bg_data["lidar_centers"]

    # Remove statistical outlier.
    points, lidar_centers = remove_statistical_outlier(
        points=points,
        lidar_centers=lidar_centers,
        nb_neighbors=args.rso_nb_neighbors,
        std_ratio=args.rso_std_ratio,
    )

    visualize_points = False
    if visualize_points:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd])

    # Init reconstructor.
    bgr = BGReconstructor(
        voxel_size=args.nksr_voxel_size,
        chunked=args.nksr_chunked,
    )
    mesh = bgr.recon(
        points=points,
        lidar_centers=lidar_centers,
    )

    visualize_mesh = False
    if visualize_mesh:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh])

    # Save.
    if dray_run:
        print("Dry run. Not saving.")
    else:
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        print(f"Saved to {mesh_path}.")


def main():
    # CUDA_VISIBLE_DEVICES=0 python lit_02_recon_bg.py --data_domain waymo --data_version v0 --skip_existing
    # CUDA_VISIBLE_DEVICES=1 python lit_02_recon_bg.py --data_domain waymo --data_version v0 --reverse --skip_existing
    # CUDA_VISIBLE_DEVICES=0 python lit_02_recon_bg.py --data_domain nuscenes --data_version v0 --skip_existing
    # CUDA_VISIBLE_DEVICES=1 python lit_02_recon_bg.py --data_domain nuscenes --data_version v0 --reverse --skip_existing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_domain",
        type=str,
        required=True,
        choices=("waymo", "nuscenes"),
    )
    parser.add_argument(
        "--data_version",
        type=str,
        required=True,
        help="Version of the data, which determines the lit_paths.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse the order of scenes.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip existing scenes that have been processed.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run without saving.",
    )
    parser.add_argument(
        "--scene_index",
        type=int,
        default=None,
        help="Scene index, when specified, only process this scene",
    )

    args = parser.parse_args()

    # Get lit_paths.
    lit_paths = get_lit_paths(
        data_version=args.data_version,
        data_domain=args.data_domain,
    )

    # Get scenes to process.
    scene_names = copy.copy(lit_paths.scene_names)
    scene_paths = [
        lit_paths.scene_dir / f"{scene_name}.pkl" for scene_name in scene_names
    ]
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

    # Check scene_names have all been extracted.
    scene_paths_extracted = sorted(list(lit_paths.scene_dir.glob("*.pkl")))
    is_all_extracted = True
    for scene_path in scene_paths:
        if scene_path not in scene_paths_extracted:
            is_all_extracted = False
            print(f"{scene_path} has not been extracted.")
    if not is_all_extracted:
        raise ValueError("Not all scenes have been extracted. Aborting.")

    # Create output dir for reconstructed bg meshes.
    lit_paths.bg_dir.mkdir(exist_ok=True, parents=True)

    # Reverse the order of scenes.
    if args.reverse:
        scene_names = scene_names[::-1]

    # Process sequentially.
    for scene_name in tqdm(scene_names, desc="Reconstructing bg meshes"):
        recon_bg(
            scene_name=scene_name,
            lit_paths=lit_paths,
            skip_existing=args.skip_existing,
            dray_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
