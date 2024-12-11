# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.


import os
import pickle

import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils, range_image_utils, transform_utils

from pcdet.utils import common_utils

try:
    tf.enable_eager_execution()
except:
    pass

WAYMO_CLASSES = ["unknown", "Vehicle", "Pedestrian", "Sign", "Cyclist"]


def generate_labels(frame):
    obj_name, difficulty, dimensions, locations, heading_angles = [], [], [], [], []
    tracking_difficulty, speeds, accelerations, obj_ids = [], [], [], []
    num_points_in_gt = []
    laser_labels = frame.laser_labels
    for i in range(len(laser_labels)):
        box = laser_labels[i].box
        class_ind = laser_labels[i].type
        loc = [box.center_x, box.center_y, box.center_z]
        heading_angles.append(box.heading)
        obj_name.append(WAYMO_CLASSES[class_ind])
        difficulty.append(laser_labels[i].detection_difficulty_level)
        tracking_difficulty.append(laser_labels[i].tracking_difficulty_level)
        dimensions.append(
            [box.length, box.width, box.height]
        )  # lwh in unified coordinate of OpenPCDet
        locations.append(loc)
        obj_ids.append(laser_labels[i].id)
        num_points_in_gt.append(laser_labels[i].num_lidar_points_in_box)

    annotations = {}
    annotations["name"] = np.array(obj_name)
    annotations["difficulty"] = np.array(difficulty)
    annotations["dimensions"] = np.array(dimensions)
    annotations["location"] = np.array(locations)
    annotations["heading_angles"] = np.array(heading_angles)

    annotations["obj_ids"] = np.array(obj_ids)
    annotations["tracking_difficulty"] = np.array(tracking_difficulty)
    annotations["num_points_in_gt"] = np.array(num_points_in_gt)

    annotations = common_utils.drop_info_with_name(annotations, name="unknown")
    if annotations["name"].__len__() > 0:
        gt_boxes_lidar = np.concatenate(
            [
                annotations["location"],
                annotations["dimensions"],
                annotations["heading_angles"][..., np.newaxis],
            ],
            axis=1,
        )
    else:
        gt_boxes_lidar = np.zeros((0, 7))
    annotations["gt_boxes_lidar"] = gt_boxes_lidar
    return annotations


def convert_range_image_to_point_cloud(
    frame, range_images, camera_projections, range_image_top_pose, ri_index=0
):
    """
    Modified from the codes of Waymo Open Dataset.
    Convert range images to point cloud.
    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        camera_projections: A dict of {laser_name,
            [camera_projection_from_first_return, camera_projection_from_second_return]}.
        range_image_top_pose: range image pixel pose for top lidar.
        ri_index: 0 for the first return, 1 for the second return.

    Returns:
        points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        cp_points: {[N, 6]} list of camera projections of length 5 (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    points_NLZ = []
    points_intensity = []
    points_elongation = []

    frame_pose = tf.convert_to_tensor(
        np.reshape(np.array(frame.pose.transform), [4, 4])
    )
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims
    )
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0],
        range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2],
    )
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation, range_image_top_pose_tensor_translation
    )
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
            beam_inclinations = range_image_utils.compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0],
            )
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims
        )
        pixel_pose_local = None
        frame_pose_local = None
        if c.name == dataset_pb2.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0)
        range_image_mask = range_image_tensor[..., 0] > 0
        range_image_NLZ = range_image_tensor[..., 3]
        range_image_intensity = range_image_tensor[..., 1]
        range_image_elongation = range_image_tensor[..., 2]
        range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(range_image_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local,
        )

        range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
        points_tensor = tf.gather_nd(range_image_cartesian, tf.where(range_image_mask))
        points_NLZ_tensor = tf.gather_nd(
            range_image_NLZ, tf.compat.v1.where(range_image_mask)
        )
        points_intensity_tensor = tf.gather_nd(
            range_image_intensity, tf.compat.v1.where(range_image_mask)
        )
        points_elongation_tensor = tf.gather_nd(
            range_image_elongation, tf.compat.v1.where(range_image_mask)
        )
        cp = camera_projections[c.name][0]
        cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims)
        cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))
        points.append(points_tensor.numpy())
        cp_points.append(cp_points_tensor.numpy())
        points_NLZ.append(points_NLZ_tensor.numpy())
        points_intensity.append(points_intensity_tensor.numpy())
        points_elongation.append(points_elongation_tensor.numpy())

    return points, cp_points, points_NLZ, points_intensity, points_elongation


def save_lidar_points(frame, cur_save_path):
    parsed_frame = frame_utils.parse_range_image_and_camera_projection(frame)

    if len(parsed_frame) == 4:
        # New API.
        (
            range_images,
            camera_projections,
            seg_labels,
            range_image_top_pose,
        ) = parsed_frame
    else:
        # Old API.
        (
            range_images,
            camera_projections,
            range_image_top_pose,
        ) = parsed_frame

    (
        points,
        cp_points,
        points_in_NLZ_flag,
        points_intensity,
        points_elongation,
    ) = convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose
    )

    # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)
    points_in_NLZ_flag = np.concatenate(points_in_NLZ_flag, axis=0).reshape(-1, 1)
    points_intensity = np.concatenate(points_intensity, axis=0).reshape(-1, 1)
    points_elongation = np.concatenate(points_elongation, axis=0).reshape(-1, 1)

    num_points_of_each_lidar = [point.shape[0] for point in points]
    save_points = np.concatenate(
        [points_all, points_intensity, points_elongation, points_in_NLZ_flag], axis=-1
    ).astype(np.float32)

    np.save(cur_save_path, save_points)
    # print('saving to ', cur_save_path)
    return num_points_of_each_lidar


def process_single_sequence(
    scene_file,
    save_path,
    sampled_interval,
    has_label=True,
    enable_only_save_lidar_poses=False,
):
    # Complete separate code path for only saving lidar poses.
    # This will always run without checking if the scene has been processed.
    if enable_only_save_lidar_poses:
        return process_single_sequence_only_save_lidar_poses(
            scene_file=scene_file,
            save_path=save_path,
            sampled_interval=sampled_interval,
            has_label=has_label,
        )

    sequence_name = os.path.splitext(os.path.basename(scene_file))[0]

    # print('Load record (sampled_interval=%d): %s' % (sampled_interval, sequence_name))
    if not scene_file.exists():
        print("NotFoundError: %s" % scene_file)
        return []

    dataset = tf.data.TFRecordDataset(str(scene_file), compression_type="")
    cur_save_dir = save_path / sequence_name
    cur_save_dir.mkdir(parents=True, exist_ok=True)
    pkl_file = cur_save_dir / ("%s.pkl" % sequence_name)

    scene_infos = []
    if pkl_file.exists():
        scene_infos = pickle.load(open(pkl_file, "rb"))
        print("Skip scene since it has been processed before: %s" % pkl_file)
        return scene_infos

    for cnt, data in enumerate(dataset):
        if cnt % sampled_interval != 0:
            continue
        # print(sequence_name, cnt)
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        info = {}
        pc_info = {
            "num_features": 5,
            "lidar_sequence": sequence_name,
            "sample_idx": cnt,
        }
        info["point_cloud"] = pc_info

        info["frame_id"] = sequence_name + ("_%03d" % cnt)
        image_info = {}
        for j in range(5):
            width = frame.context.camera_calibrations[j].width
            height = frame.context.camera_calibrations[j].height
            image_info.update({"image_shape_%d" % j: (height, width)})
        info["image"] = image_info

        # Save the intrinsics, 4x4 extrinsic matrix, width, and height of each camera.
        save_waymo_calibrations = False
        if save_waymo_calibrations:
            clib_dict = dict()
            for c in frame.context.camera_calibrations:
                cam_name_str = dataset_pb2.CameraName.Name.Name(c.name)
                clib_dict[f"CAM_{cam_name_str}_INTRINSIC"] = np.array(
                    c.intrinsic, np.float32
                )
                clib_dict[f"CAM_{cam_name_str}_EXTRINSIC"] = np.reshape(
                    np.array(c.extrinsic.transform, np.float32), [4, 4]
                )
                clib_dict[f"CAM_{cam_name_str}_WIDTH"] = np.array(c.width)
                clib_dict[f"CAM_{cam_name_str}_HEIGHT"] = np.array(c.height)
                clib_dict[f"CAM_{cam_name_str}_ROLLING_SHUTTER_DIRECTION"] = np.array(
                    c.rolling_shutter_direction
                )
            for l in frame.context.laser_calibrations:
                lidar_name_str = dataset_pb2.LaserName.Name.Name(l.name)
                clib_dict[f"LIDAR_{lidar_name_str}_EXTRINSIC"] = np.reshape(
                    np.array(l.extrinsic.transform, np.float32), [4, 4]
                )

            save_path = "waymo_calibrations.pkl"
            pickle.dump(clib_dict, open(save_path, "wb"))
            print(f"Calibrations are saved to {save_path}")
            exit(0)

        pose = np.array(frame.pose.transform, dtype=np.float32).reshape(4, 4)
        info["pose"] = pose

        if has_label:
            annotations = generate_labels(frame)
            info["annos"] = annotations

        num_points_of_each_lidar = save_lidar_points(
            frame, cur_save_dir / ("%04d.npy" % cnt)
        )
        info["num_points_of_each_lidar"] = num_points_of_each_lidar

        scene_infos.append(info)

    with open(pkl_file, "wb") as f:
        pickle.dump(scene_infos, f)

    print("Infos are saved to (sampled_interval=%d): %s" % (sampled_interval, pkl_file))
    return scene_infos


def process_single_sequence_only_save_lidar_poses(
    scene_file,
    save_path,
    sampled_interval,
    has_label=True,
):
    """
    Extra info will be save to:
        f"{save_path}_extra" / f"{sequence_name}.pkl"
    """

    # Paths.
    sequence_name = os.path.splitext(os.path.basename(scene_file))[0]
    pkl_dir = save_path.parent / f"{save_path.name}_extra"
    pkl_path = save_path.parent / f"{save_path.name}_extra" / f"{sequence_name}.pkl"
    pkl_dir.mkdir(parents=True, exist_ok=True)

    # Check input.
    if not scene_file.exists():
        print("NotFoundError: %s" % scene_file)
        return []

    # Check output.
    scene_infos = []
    if pkl_path.exists():
        scene_infos = pickle.load(open(pkl_path, "rb"))
        print("Skip scene since it has been processed before: %s" % pkl_path)
        return scene_infos

    dataset = tf.data.TFRecordDataset(str(scene_file), compression_type="")

    for cnt, data in enumerate(dataset):
        if cnt % sampled_interval != 0:
            continue

        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        # TOP: 1, FRONT: 2, SIDE_LEFT: 3, SIDE_RIGHT: 4, REAR: 5
        calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
        lidar_poses = [
            np.reshape(np.array(c.extrinsic.transform, np.float32), [4, 4])
            for c in calibrations
        ]
        frame_pose = np.array(frame.pose.transform, np.float32).reshape(4, 4)

        info = {}
        info["sequence_name"] = sequence_name
        info["sample_idx"] = cnt
        info["frame_id"] = sequence_name + ("_%03d" % cnt)
        info["lidar_to_vehicle_poses"] = lidar_poses
        info["frame_pose"] = frame_pose

        scene_infos.append(info)

    with open(pkl_path, "wb") as f:
        pickle.dump(scene_infos, f)

    return scene_infos
