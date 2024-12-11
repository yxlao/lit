import argparse
import copy
import pickle
import sys

import numpy as np
import open3d as o3d
from tqdm import tqdm

from lit.containers.fg_scene import FGScene
from lit.containers.scene import Scene
from lit.fg_reconstructor import FGReconstructor
from lit.path_utils import LitPaths, get_lit_paths
from lit.recon_utils import bbox_to_lineset
from lit_tools.global_configs import global_configs

sys.excepthook = lambda et, ev, tb: (
    None
    if issubclass(et, (KeyboardInterrupt, SystemExit))
    else (print(f"Unhandled exception: {ev}"), __import__("ipdb").post_mortem(tb))
)


def recon_fg(
    scene_name: str,
    fgr: FGReconstructor,
    lit_paths: LitPaths,
    skip_existing: bool = False,
    verbose: bool = False,
    dry_run: bool = False,
):
    # Maybe skip existing.
    fg_path = lit_paths.fg_dir / f"{scene_name}.pkl"
    if skip_existing and fg_path.exists():
        print(f"Skipping existing {fg_path}.")
        return

    # Load.
    scene_path = lit_paths.scene_dir / f"{scene_name}.pkl"
    scene = Scene.load(scene_path)

    # Extract foreground.
    if lit_paths.data_domain == "waymo":
        fg_boxes = scene.extract_fg(select_labels=[1], verbose=False)
    elif lit_paths.data_domain == "nuscenes":
        # These classes will be reconstructed and put back.
        fg_boxes = scene.extract_fg(
            select_labels=global_configs.nuscenes_class_labels_to_recon,
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown data_domain: {lit_paths.data_domain}")

    # Group fg_boxes.
    fg_scene = FGScene.from_fg_boxes(fg_boxes)
    print(f"Grouped {len(fg_boxes)} fg_boxes into {len(fg_scene)} groups.")

    # Visualize groups.
    # Each group is colored with a random color.
    visualize_groups = False
    if visualize_groups:
        all_fg_object_ls = o3d.geometry.LineSet()
        for fg_object in fg_scene:
            fg_object_color = np.random.rand(3)
            fg_object_ls = o3d.geometry.LineSet()
            for fg_box in fg_object.fg_boxes:
                ls_fg_box = bbox_to_lineset(
                    fg_box.local_bbox, frame_pose=fg_box.frame_pose
                )
                ls_fg_box.paint_uniform_color(fg_object_color)
                fg_object_ls += ls_fg_box
            all_fg_object_ls += fg_object_ls
        o3d.visualization.draw_geometries([all_fg_object_ls])

    # Reconstruct foreground mesh for each group.
    for fg_object in tqdm(
        fg_scene,
        desc="Reconstructing fg_objects",
        total=len(fg_scene),
        disable=not verbose,
    ):
        # Reconstruct.
        mesh = fgr.recon_fg_object(fg_object)
        fg_object.mesh_vertices = np.asarray(mesh.vertices).astype(np.float32)
        fg_object.mesh_triangles = np.asarray(mesh.triangles)

        # Visualize.
        visualize_group_mesh = False
        if visualize_group_mesh:
            fg_box = fg_object[0]  # They are all visualize the same.
            pseudo_pose = fg_box.compute_local_pseudo_pose()
            world_mesh = copy.deepcopy(mesh)
            world_mesh = world_mesh.transform(fg_box.frame_pose @ pseudo_pose)
            world_mesh.compute_vertex_normals()
            world_ls = bbox_to_lineset(
                fg_box.local_bbox,
                frame_pose=fg_box.frame_pose,
            )

            bbox_label = fg_box.local_bbox[7]
            title = f"label: {bbox_label}"
            if lit_paths.data_domain == "nuscenes":
                class_name = global_configs.nuscenes_extract_label_to_class_name[
                    bbox_label
                ]
                title += f", class: {class_name}"

            o3d.visualization.draw_geometries([world_mesh, world_ls], window_name=title)

    # Visualize all groups in a combined mesh.
    visualize_groups_mesh = False
    if visualize_groups_mesh:
        frame_index = 0

        groups_mesh = o3d.geometry.TriangleMesh()
        for fg_object in fg_scene:
            # Check if the group has a mesh in this frame.
            target_fg_box = None
            for fg_box in fg_object.fg_boxes:
                if fg_box.frame_index == frame_index:
                    target_fg_box = fg_box
                    break
            if target_fg_box is None:
                continue

            # Transform the mesh to the world frame.
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(fg_object.mesh_vertices)
            mesh.triangles = o3d.utility.Vector3iVector(fg_object.mesh_triangles)
            pseudo_pose = target_fg_box.compute_local_pseudo_pose()
            mesh = mesh.transform(target_fg_box.frame_pose @ pseudo_pose)
            groups_mesh += mesh

        groups_mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([groups_mesh])

    # Save the fg_scene.
    if dry_run:
        print(f"Dry run: Not saving {fg_path}.")
    else:
        fg_scene.save(fg_path)


def main():
    """
    CUDA_VISIBLE_DEVICES=0 python lit_03_recon_fg.py --data_domain waymo --data_version v0 --skip_existing
    CUDA_VISIBLE_DEVICES=1 python lit_03_recon_fg.py --data_domain waymo --data_version v0 --reverse --skip_existing
    CUDA_VISIBLE_DEVICES=0 python lit_03_recon_fg.py --data_domain waymo --data_version v0 --shuffle --skip_existing
    CUDA_VISIBLE_DEVICES=1 python lit_03_recon_fg.py --data_domain waymo --data_version v0 --shuffle --skip_existing

    CUDA_VISIBLE_DEVICES=0 python lit_03_recon_fg.py --data_domain nuscenes --data_version v0 --skip_existing
    CUDA_VISIBLE_DEVICES=1 python lit_03_recon_fg.py --data_domain nuscenes --data_version v0 --reverse --skip_existing
    CUDA_VISIBLE_DEVICES=0 python lit_03_recon_fg.py --data_domain nuscenes --data_version v0 --shuffle --skip_existing
    CUDA_VISIBLE_DEVICES=1 python lit_03_recon_fg.py --data_domain nuscenes --data_version v0 --shuffle --skip_existing
    """
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
        "--shuffle",
        action="store_true",
        help="Shuffle the order of scenes.",
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
        "--verbose",
        action="store_true",
        help="Print more information.",
    )
    parser.add_argument(
        "--scene_index",
        type=int,
        default=None,
        help="Scene index, when specified, only process this scene",
    )

    args = parser.parse_args()

    # Init paths.
    lit_paths = get_lit_paths(
        data_version=args.data_version,
        data_domain=args.data_domain,
    )
    lit_paths.fg_dir.mkdir(exist_ok=True, parents=True)

    # Get scene names.
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
    if args.reverse:
        scene_names = list(reversed(scene_names))
    if args.shuffle:
        np.random.shuffle(scene_names)
    print(f"Found {len(scene_names)} scenes.")

    # Determine whether to use DeepSDF.
    if lit_paths.data_version in {"v0", "v1"}:
        use_deepsdf = True
    else:
        raise ValueError(f"Unknown data_version: {lit_paths.data_version}")

    # Init FGReconstructor.
    fgr = FGReconstructor(
        use_deepsdf=use_deepsdf,
    )

    # Run foreground reconstruction.
    for scene_name in tqdm(scene_names, desc="Reconstructing fg_objects."):
        print(f"Processing {scene_name}...")
        recon_fg(
            scene_name,
            fgr=fgr,
            lit_paths=lit_paths,
            skip_existing=args.skip_existing,
            verbose=args.verbose,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
