import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm

from lit.containers.frame import Frame
from lit.containers.scene import Scene
from lit.path_utils import get_lit_paths
from lit.recon_utils import bbox_to_lineset
from lit_tools.global_configs import global_configs
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import NuScenesDataset, WaymoDataset
from pcdet.utils import common_utils


def extract_scene_waymo(args, cfg, logger, scene_index=None):
    dataset = WaymoDataset(
        dataset_cfg=cfg,
        class_names=["Vehicle", "Pedestrian", "Cyclist"],
        training=args.split == "train",
        logger=logger,
        allow_empty_gt_boxes=True,
        include_extras=True,
    )

    lit_paths = get_lit_paths(
        data_version=args.data_version,
        data_domain="waymo",
    )
    lit_paths.scene_dir.mkdir(parents=True, exist_ok=True)

    # Count number of frames per scene.
    map_scene_name_to_num_frames = OrderedDict()
    for info in dataset.infos:
        scene_name = info["point_cloud"]["lidar_sequence"]
        if scene_name not in map_scene_name_to_num_frames:
            map_scene_name_to_num_frames[scene_name] = 0
        map_scene_name_to_num_frames[scene_name] += 1

    # Compute scene starts and ends.
    print(f"Total num scenes: {len(map_scene_name_to_num_frames)}")
    print(f"Total num frames: {sum(map_scene_name_to_num_frames.values())}")
    scene_names = list(map_scene_name_to_num_frames.keys())
    scene_num_frames = list(map_scene_name_to_num_frames.values())
    scene_starts = np.cumsum([0] + scene_num_frames)[:-1]  # Inclusive.
    scene_ends = np.cumsum(scene_num_frames)  # Exclusive.

    # If scene_index is specified, only extract this scene.
    if scene_index is not None:
        assert isinstance(scene_index, int)
        assert (
            0 <= scene_index < len(scene_names)
        ), f"{scene_index} not in [0, {len(scene_names)})"
        print(f"Only extracting scene_index={scene_index}: {scene_names[scene_index]}")
        scene_names = [scene_names[scene_index]]
        scene_starts = [scene_starts[scene_index]]
        scene_ends = [scene_ends[scene_index]]

    # Extract scenes, save each scene as a pkl file.
    for scene_name, scene_start, scene_end in tqdm(
        zip(scene_names, scene_starts, scene_ends),
        desc="Extracting scenes",
        total=len(scene_names),
    ):
        # Init.
        scene = Scene(scene_name=scene_name)
        scene_path = lit_paths.scene_dir / f"{scene_name}.pkl"

        # Skip existing.
        if args.skip_existing and scene_path.exists():
            try:
                scene = Scene.load(scene_path)
            except Exception as e:
                print(f"{scene_path} exists but failed to load, continue processing.")
            else:
                print(f"{scene_name} of {len(scene_names)} frames exists. Skipping.")
                continue

        # Append frames.
        for frame_index, global_frame_index in enumerate(range(scene_start, scene_end)):
            frame_dict = dataset[global_frame_index]

            # Unpack indices.
            frame_index = frame_dict["sample_idx"]  # e.g 15

            # Unpack data.
            points = frame_dict["points"][:]  # (x, y, z, i, e)
            gt_boxes = frame_dict["gt_boxes"].astype(np.float32)  # (N, 8)
            pose = frame_dict["pose"]  # (4, 4)
            num_points_of_each_lidar = frame_dict["num_points_of_each_lidar"]  # (5,)
            lidar_to_vehicle_poses = frame_dict["lidar_to_vehicle_poses"]  # (5, 4, 4)

            # Check data.
            if len(points) != np.sum(num_points_of_each_lidar):
                raise ValueError(
                    f"In {scene_name}, "
                    f"len(points) != np.sum(num_points_of_each_lidar): "
                    f"{len(points)} != {np.sum(num_points_of_each_lidar)}"
                )

            # Print warnings if the frame's gt_boxes are empty.
            if len(gt_boxes) == 0:
                print(f"Warning: {scene_name} frame {frame_index} has no gt_boxes.")

            # Object ids.
            object_ids = frame_dict["obj_ids"]
            if not len(object_ids) == len(gt_boxes):
                raise ValueError(
                    f"len(object_ids) != len(gt_boxes): "
                    f"{len(object_ids)} != {len(gt_boxes)}"
                )

            # Append.
            scene.append_frame(
                Frame(
                    scene_name=scene_name,
                    frame_index=frame_index,
                    frame_pose=pose,
                    lidar_to_vehicle_poses=lidar_to_vehicle_poses,
                    num_points_of_each_lidar=num_points_of_each_lidar,
                    local_points=points,
                    local_bboxes=gt_boxes,
                    object_ids=object_ids,
                ),
                check_valid=True,
            )

            # Visualize frame.
            visualize_frame = False
            if visualize_frame:
                # Visualize.
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points[:, :3])
                ls = o3d.geometry.LineSet()
                for gt_box in gt_boxes:
                    ls += bbox_to_lineset(gt_box)
                coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
                o3d.visualization.draw_geometries([pcd, ls, coord])

        # Save.
        scene.save(path=scene_path, verbose=False)


def extract_scene_nuscenes(args, cfg, logger, scene_index=None):
    # # KITTI
    # all_kitti_class_names = [
    #     "Car",
    #     "Van",
    #     "Truck",
    #     "Pedestrian",
    #     "Person",
    #     "Cyclist",
    #     "Tram",
    #     "Misc",
    # ]
    dataset = NuScenesDataset(
        dataset_cfg=cfg,
        class_names=global_configs.nuscenes_extract_class_names,
        training=args.split == "train",
        logger=logger,
        include_extras=True,
    )

    # Make sure that there are no SHIFT_COOR applied. This way, the
    # reconstructed geometry has the same coordinate as the raw dataset.
    # See README_coordinates.md for more details.
    if "SHIFT_COOR" in dataset.dataset_cfg:
        assert np.allclose(dataset.dataset_cfg["SHIFT_COOR"], [0, 0, 0])

    lit_paths = get_lit_paths(
        data_version=args.data_version,
        data_domain="nuscenes",
    )
    lit_paths.scene_dir.mkdir(parents=True, exist_ok=True)

    # Count the number of frames per scene. This can also be read from
    # the scene.json file or the related APIs. However, we directly get this
    # info from the dataset.info as we need to get them in a specific order.
    map_scene_name_to_num_frames = OrderedDict()
    for info in dataset.infos:
        sample_dict = dataset.nusc.get("sample", info["token"])
        scene_dict = dataset.nusc.get("scene", sample_dict["scene_token"])
        scene_name = scene_dict["token"]
        if scene_name not in map_scene_name_to_num_frames:
            map_scene_name_to_num_frames[scene_name] = 0
        map_scene_name_to_num_frames[scene_name] += 1
    # Then, we cross validate this result wih the one retrieved from the API.
    for scene_name in map_scene_name_to_num_frames:
        scene_dict = dataset.nusc.get("scene", scene_name)
        assert map_scene_name_to_num_frames[scene_name] == scene_dict["nbr_samples"]

    # Compute scene starts and ends.
    print(f"Total num scenes: {len(map_scene_name_to_num_frames)}")
    print(f"Total num frames: {sum(map_scene_name_to_num_frames.values())}")

    scene_names = list(map_scene_name_to_num_frames.keys())
    scene_num_frames = list(map_scene_name_to_num_frames.values())
    scene_starts = np.cumsum([0] + scene_num_frames)[:-1]  # Inclusive.
    scene_ends = np.cumsum(scene_num_frames)  # Exclusive.

    # If scene_index is specified, only extract this scene.
    if scene_index is not None:
        assert isinstance(scene_index, int)
        assert (
            0 <= scene_index < len(scene_names)
        ), f"{scene_index} not in [0, {len(scene_names)})"
        print(f"Only extracting scene_index={scene_index}: {scene_names[scene_index]}")
        scene_names = [scene_names[scene_index]]
        scene_starts = [scene_starts[scene_index]]
        scene_ends = [scene_ends[scene_index]]

    # Extract scenes, save each scene as a pkl file.
    for scene_name, scene_start, scene_end in tqdm(
        zip(scene_names, scene_starts, scene_ends),
        desc="Extracting scenes",
        total=len(scene_names),
    ):
        # Init.
        scene_is_valid = True
        scene = Scene(scene_name=scene_name)
        scene_path = lit_paths.scene_dir / f"{scene_name}.pkl"

        # Skip existing.
        if args.skip_existing and scene_path.exists():
            scene = Scene.load(scene_path)
            print(f"{scene_name} of {len(scene_names)} frames exists. Skipping.")

        # Append frames.
        for proposed_frame_index, global_frame_index in enumerate(
            range(scene_start, scene_end)
        ):
            frame_dict = dataset[global_frame_index]

            # Unpack indices.
            frame_index = frame_dict["sample_idx"]

            # Unpack data.
            points = frame_dict["points"][:]  # (x, y, z)
            gt_boxes = frame_dict["gt_boxes"].astype(
                np.float32
            )  # (x, y, z, dx, dy, dz, heading, label)
            pose = frame_dict["pose"]  # (4, 4)
            num_points_of_each_lidar = frame_dict["num_points_of_each_lidar"]  # (1,)
            lidar_to_vehicle_poses = frame_dict["lidar_to_vehicle_poses"]  # (1, 4, 4)

            ####################################################################
            # Special treatment for NuScenes pose
            ####################################################################
            # The points we get here are in the lidar coordinate. However, the
            # points we get for Waymo is in the vehicle (ego) coordinate.
            # We have two options:
            #
            # 1. Transform points.
            #    Transform points to vehicle coordinate and use this new points
            #    from now on. The problem with this option is that we also need
            #    to transform the gt_boxes, which is not trivial.
            # 2. Transform poses (selected).
            #    - frame_pose            <- frame_pose @ lidar_to_vehicle_pose
            #    - lidar_to_vehicle_pose <- identity
            #    This way, we can use the original points from now on. We use
            #    this option.
            assert len(lidar_to_vehicle_poses) == 1
            pose = (pose @ lidar_to_vehicle_poses[0]).astype(np.float32)
            lidar_to_vehicle_poses[0] = np.eye(4, dtype=np.float32)
            ####################################################################

            # Check data.
            if len(points) != np.sum(num_points_of_each_lidar):
                raise ValueError(
                    f"In {scene_name}, "
                    f"len(points) != np.sum(num_points_of_each_lidar): "
                    f"{len(points)} != {np.sum(num_points_of_each_lidar)}"
                )

            # Print warnings if the frame's gt_boxes are empty.
            if len(gt_boxes) == 0:
                print(f"Warning: {scene_name} frame {frame_index} has no gt_boxes.")

            # Object ids.
            object_ids = frame_dict["obj_ids"]
            if not len(object_ids) == len(gt_boxes):
                raise ValueError(
                    f"len(object_ids) != len(gt_boxes): "
                    f"{len(object_ids)} != {len(gt_boxes)}"
                )

            # Append.
            if proposed_frame_index != frame_index:
                print(
                    f"[Invalid Scene] {scene_name} frame frame_index={frame_index}, "
                    f"but proposed_frame_index={proposed_frame_index}. "
                )
                scene_is_valid = False
                break

            # Append.
            scene.append_frame(
                Frame(
                    scene_name=scene_name,
                    frame_index=frame_index,
                    frame_pose=pose,
                    lidar_to_vehicle_poses=lidar_to_vehicle_poses,
                    num_points_of_each_lidar=num_points_of_each_lidar,
                    local_points=points,
                    local_bboxes=gt_boxes,
                    object_ids=object_ids,
                ),
                check_valid=True,
            )

            # Visualize frame.
            visualize_frame = False
            if visualize_frame:
                # Print a summary of number of box per class.
                print(f"Total number of boxes: {len(gt_boxes)}")
                print("Number of boxes per class:")
                for class_name in global_configs.nuscenes_extract_class_names:
                    num_boxes = np.sum(
                        gt_boxes[:, -1]
                        == global_configs.nuscenes_extract_class_names.index(class_name)
                        + 1
                    )
                    print(f"  - {class_name}: {num_boxes}")

                # Visualize.
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points[:, :3])
                ls = o3d.geometry.LineSet()
                for gt_box in gt_boxes:
                    ls += bbox_to_lineset(gt_box)
                coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
                o3d.visualization.draw_geometries([pcd, ls, coord])

        # Visualize scene.
        visualize = False
        if visualize:
            scene.visualize()

        # Save scene.
        if scene_is_valid:
            scene.save(path=scene_path, verbose=False)


def main():
    # python lit_01_extract_scene.py --cfg_file ../tools/cfgs/dataset_configs/waymo_dataset_extract.yaml --split train
    # python lit_01_extract_scene.py --cfg_file ../tools/cfgs/dataset_configs/waymo_dataset_extract.yaml --split valid
    # python lit_01_extract_scene.py --cfg_file ../tools/cfgs/dataset_configs/nuscenes_dataset_extract.yaml --split train
    # python lit_01_extract_scene.py --cfg_file ../tools/cfgs/dataset_configs/nuscenes_dataset_extract.yaml --split valid
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_file",
        type=str,
        default=None,
        help="Config path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split of the dataset",
        choices=["train", "valid"],
    )
    parser.add_argument(
        "--data_version",
        type=str,
        default="v1",
        help="Data version",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip existing files",
    )
    parser.add_argument(
        "--scene_index",
        type=int,
        default=None,
        help="Scene index, when specified, only extract this scene",
    )
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    np.random.seed(1024)

    logger = common_utils.create_logger("/tmp/st3d_extract.log")

    # Make sure we don't use the wrong config file
    # - Pre-processing shall be disabled.
    # - Shuffle shall be disabled.
    # - Batch size shall be 1.
    # - Augmentation shall be disabled.
    assert Path(args.cfg_file).name in [
        "waymo_dataset_extract.yaml",
        "waymo_dataset_info.yaml",
        "nuscenes_dataset_extract.yaml",
    ]

    if cfg["DATASET"] == "WaymoDataset":
        extract_scene_waymo(args, cfg, logger, scene_index=args.scene_index)
    elif cfg["DATASET"] == "NuScenesDataset":
        extract_scene_nuscenes(args, cfg, logger, scene_index=args.scene_index)
    else:
        raise ValueError(f"Unknown dataset: {cfg['DATASET']}")


if __name__ == "__main__":
    main()
