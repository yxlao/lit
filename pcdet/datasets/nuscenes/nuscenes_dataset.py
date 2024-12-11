import copy
import pickle
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from nuscenes.nuscenes import NuScenes, transform_matrix
from pyquaternion import Quaternion
from tqdm import tqdm

from lit.path_utils import get_lit_paths
from lit.recon_utils import bboxes_to_lineset, scale_bboxes_by_domain
from pcdet.datasets.dataset import DatasetTemplate
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import box_utils, common_utils, self_training_utils


def path_to_scene_and_timestamp(path):
    """
    Args:
        path: samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin

    Return:
        ("n008-2018-08-01-15-16-36-0400", "1533151603547590")
    """
    path = Path(path)
    path_stem = Path(path.stem).stem
    tokens = path_stem.split("__")
    if len(tokens) != 3:
        raise ValueError(f"Invalid path: {path}")
    scene = tokens[0]
    timestamp = tokens[2]
    return scene, timestamp


class NuScenesDataset(DatasetTemplate):
    def __init__(
        self,
        dataset_cfg,
        class_names,
        training=True,
        root_path=None,
        logger=None,
        include_extras=False,  # Include obj_ids
    ):
        self.include_extras = include_extras
        root_path = (
            root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)
        ) / dataset_cfg.VERSION
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
        )
        self.infos = []
        self.include_nuscenes_data(self.mode)
        if self.training and self.dataset_cfg.get("BALANCED_RESAMPLING", False):
            self.infos = self.balanced_infos_resampling(self.infos)

        # Load the official NuScenes object for tokens mappings. E.g., we need
        # to get the instance_token given a annotation token. The hierarchy for
        # nuScenes is:
        # - log                       # log.json (we don't need to care)
        #   - scene                   # scene.json
        #     - sample                # sample.json
        #         - sample_annotation # sample_annotation.json
        #
        # Our goal is to:
        #     1) Extract scenes, each scene has list of Frames.
        #     2) Each Frame contains:
        #        - [ ] scene_name
        #        - [ ] sample_idx
        #        - [ ] points
        #        - [ ] num_points_of_each_lidar
        #        - [ ] gt_boxes
        #        - [ ] obj_ids
        #        - [ ] frame_pose
        #        - [ ] lidar_to_vehicle_poses
        #
        # sample_annotation.json: format of a box's annotation
        #     {
        #         # box token, e.g. infos["gt_boxes_token"][0]
        #         "token": "173a50411564442ab195e132472fde71",
        #         # token of the frame"
        #         "sample_token": "e93e98b63d3b40209056d129dc53ceee",
        #         # object token, i.e our obj_id
        #         "instance_token": "5e2b6fd1fab74d04a79eefebbec357bb",
        #         "visibility_token": "4",
        #         "attribute_tokens": [],
        #         "translation": [994.031, 612.51, 0.728],
        #         "size": [0.3, 0.291, 0.734],
        #         "rotation": [-0.04208490861058176, 0.0, 0.0, 0.9991140377690821],
        #         "prev": "",
        #         "next": "35034272eb1f413187ae7b6affb6ec7a",
        #         "num_lidar_pts": 2,
        #         "num_radar_pts": 0,
        #     }
        #
        # instance.json: format of an instance object.
        #    {
        #        "token": "5e2b6fd1fab74d04a79eefebbec357bb",
        #        "category_token": "85abebdccd4d46c7be428af5a6173947",
        #        "nbr_annotations": 13,
        #        "first_annotation_token": "173a50411564442ab195e132472fde71",
        #        "last_annotation_token": "2cd832644d09479389ed0785e5de85c9",
        #    }
        #
        # Ref:
        # https://www.nuscenes.org/tutorials/nuscenes_tutorial.html
        self.nusc = NuScenes(
            version=self.dataset_cfg.VERSION,
            dataroot=str(self.root_path),
            verbose=True,
        )

        # Check if we shall use simulated data.
        # E.g., set dataset_cfg.DST_STYLE to "kitti" to use simulated
        # nuScenes data with KITTI style.
        self.dst_style = self.dataset_cfg.get("DST_STYLE", None)

        # Replace info["lidar_path"] with simulated lidar_path.
        # Not all training frames have simulated data, thus the self.infos may
        # not have the same length as before.
        if self.dst_style == "kitti":
            new_infos = []

            # Ref: init_paths.py
            lit_paths = get_lit_paths(
                data_version=self.dataset_cfg.DATA_VERSION,
                data_domain="nuscenes",
            )
            sim_frame_dir = lit_paths.to_kitti_sim_frame_dir

            for info in self.infos:
                sample_dict = self.nusc.get("sample", info["token"])

                scene_name = sample_dict["scene_token"]
                if scene_name in lit_paths.scene_list:
                    sample_tokens_in_scene = self.get_sample_tokens_from_scene_token(
                        scene_name
                    )
                    frame_index = sample_tokens_in_scene.index(info["token"])
                    lidar_path = sim_frame_dir / scene_name / f"{frame_index:04d}.npz"
                    if lidar_path.is_file():
                        info["lidar_path"] = str(lidar_path)
                        new_infos.append(info)
                    else:
                        print(
                            f"[WARNING] "
                            f"{lidar_path} does not exist but it "
                            f"is in scene_list for data version "
                            f"{lit_paths.data_version}"
                        )
            print(
                f"Found {len(new_infos)} frames with simulated data "
                f"out of {len(self.infos)}."
            )
            self.infos = new_infos

        # For debugging, clip the info to first 128.
        # self.infos = self.infos[:128]
        # self.infos = self.infos[:int(len(self.infos) * 0.1)]

    def include_nuscenes_data(self, mode):
        self.logger.info("Loading NuScenes dataset")
        nuscenes_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, "rb") as f:
                infos = pickle.load(f)
                nuscenes_infos.extend(infos)

        self.infos.extend(nuscenes_infos)
        self.logger.info(
            "Total samples for NuScenes dataset: %d" % (len(nuscenes_infos))
        )

        # Sample lidar paths
        sample_lidar_paths = sorted([info["lidar_path"] for info in self.infos])
        if len(sample_lidar_paths) != len(self.infos):
            raise ValueError(
                f"sample_lidar_paths({len(sample_lidar_paths)}) != infos({len(self.infos)})"
            )
        sample_logs_timestamps = [
            path_to_scene_and_timestamp(p) for p in sample_lidar_paths
        ]
        sample_log_set = set([s for s, _ in sample_logs_timestamps])

        # Sweep lidar paths
        # data/nuscenes/v1.0-mini/sweeps/LIDAR_TOP
        sweep_dir = self.root_path / "sweeps" / "LIDAR_TOP"
        sweep_lidar_paths = sorted(
            [str(p.relative_to(self.root_path)) for p in sweep_dir.glob("*.bin")]
        )
        sweep_log_timestamps = [
            path_to_scene_and_timestamp(p) for p in sweep_lidar_paths
        ]
        # Filter sweep_scenes_timestamps with sample_scenes_set
        sweep_log_timestamps = [
            (s, t) for s, t in sweep_log_timestamps if s in sample_log_set
        ]

        # Build log_map_timestamp_to_id
        #   - log_map_timestamp_to_id["scene_x"]["1533151603547590"] = 0
        #   - log_map_timestamp_to_id["scene_x"]["1533151604048025"] = 1
        #   - ...
        #   - log_map_timestamp_to_id["scene_x"]["1533151604548459"] = 100
        #   - log_map_timestamp_to_id["scene_y"]["1538984233547259"] = 0
        #   - log_map_timestamp_to_id["scene_y"]["1538984234047694"] = 1
        #   - ...
        #   - log_map_timestamp_to_id["scene_y"]["1538984234548129"] = 100
        all_log_timestamps = sample_logs_timestamps + sweep_log_timestamps
        all_log_timestamps = sorted(all_log_timestamps)

        log_map_timestamp_to_id = {}
        for log, timestamp in all_log_timestamps:
            if log not in log_map_timestamp_to_id:
                log_map_timestamp_to_id[log] = {}
            if timestamp in log_map_timestamp_to_id[log]:
                raise ValueError(f"Duplicate timestamp({timestamp}) in scene({log})")
            log_map_timestamp_to_id[log][timestamp] = len(log_map_timestamp_to_id[log])

        # Print stats
        # self.logger.info("log, num_samples, num_sweeps, num_total")
        # for log in sorted(log_map_timestamp_to_id.keys()):
        #     num_samples = len([t for s, t in sample_logs_timestamps if s == log])
        #     num_sweeps = len([t for s, t in sweep_log_timestamps if s == log])
        #     num_total = len(log_map_timestamp_to_id[log])
        #     if num_samples + num_sweeps != num_total:
        #         raise ValueError(
        #             f"num_samples({num_samples}) + num_sweeps({num_sweeps}) != num_total({num_total})"
        #         )
        #     self.logger.info(f"{log}, {num_samples}, {num_sweeps}, {num_total}")

        self.log_map_timestamp_to_id = log_map_timestamp_to_id

    def balanced_infos_resampling(self, infos):
        """
        Class-balanced sampling of nuScenes dataset from https://arxiv.org/abs/1908.09492
        """
        if self.class_names is None:
            return infos

        cls_infos = {name: [] for name in self.class_names}
        for info in infos:
            for name in set(info["gt_names"]):
                if name in self.class_names:
                    cls_infos[name].append(info)

        duplicated_samples = sum([len(v) for _, v in cls_infos.items()])
        cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()}

        sampled_infos = []

        frac = 1.0 / len(self.class_names)
        ratios = [frac / v for v in cls_dist.values()]

        for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios):
            sampled_infos += np.random.choice(
                cur_cls_infos, int(len(cur_cls_infos) * ratio)
            ).tolist()
        self.logger.info(
            "Total samples after balanced resampling: %s" % (len(sampled_infos))
        )

        cls_infos_new = {name: [] for name in self.class_names}
        for info in sampled_infos:
            for name in set(info["gt_names"]):
                if name in self.class_names:
                    cls_infos_new[name].append(info)

        cls_dist_new = {
            k: len(v) / len(sampled_infos) for k, v in cls_infos_new.items()
        }

        return sampled_infos

    @staticmethod
    def remove_ego_points(points, center_radius=1.0):
        # By default,
        # +x (red)  : right
        # +y (green): front
        # +z (blue) : up
        mask = ~(
            (np.abs(points[:, 0]) < center_radius)
            & (np.abs(points[:, 1]) < center_radius * 1.5)
        )
        return points[mask]

    def get_sweep(self, sweep_info):
        """
        This function is typically not used for domain adaptation, as we always
        set max_sweeps=1 for train/eval.
        """
        lidar_path = self.root_path / sweep_info["lidar_path"]
        points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape(
            [-1, 5]
        )[:, :4]
        points_sweep = self.remove_ego_points(points_sweep).T
        if sweep_info["transform_matrix"] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweep_info["transform_matrix"].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points)))
            )[:3, :]

        cur_times = sweep_info["time_lag"] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T

    def get_lidar_with_sweeps(self, index, max_sweeps=1, force_nearest_sweeps=False):
        info = self.infos[index]
        if Path(info["lidar_path"]).suffix == ".bin":
            lidar_path = self.root_path / info["lidar_path"]
            # Load the raw nuScenes bin file.
            points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape(
                [-1, 5]
            )[:, :4]
        elif Path(info["lidar_path"]).suffix == ".npz":
            lidar_path = info["lidar_path"]  # Already the full relative path
            # Load "points" from npz (N, 3).
            points = np.load(lidar_path)["local_points"]
            # Append zeros as intensity -> (N, 4).
            points = np.concatenate(
                (
                    points,
                    np.zeros((points.shape[0], 1), dtype=np.float32),
                ),
                axis=1,
            ).astype(np.float32)
        else:
            raise ValueError(f"Unknown lidar_path.suffix: {lidar_path}")

        points = self.remove_ego_points(points, center_radius=1.5)
        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        # Get sweeps before the current key frame and after the previous key frame.
        # If max_sweeps=1, then this has no effect.
        if force_nearest_sweeps:
            ks = range(max_sweeps - 1)
        else:
            ks = np.random.choice(len(info["sweeps"]), max_sweeps - 1, replace=False)
        for k in ks:
            points_sweep, times_sweep = self.get_sweep(info["sweeps"][k])
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.concatenate((points, times), axis=1)

        return points

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def get_sample_tokens_from_scene_token(self, scene_token):
        """
        Return a list of ordered sample tokens for a scene token.

        https://github.com/nutonomy/nuscenes-devkit/issues/713#issuecomment-1030722858
        """
        scene_dict = self.nusc.get("scene", scene_token)

        curr_token = scene_dict["first_sample_token"]
        keep_looping = True
        sample_tokens = []
        while keep_looping:
            sample_tokens.append(curr_token)
            if curr_token == scene_dict["last_sample_token"]:
                keep_looping = False
            sample_dict = self.nusc.get("sample", curr_token)
            next_token = sample_dict["next"]
            curr_token = next_token

        assert len(sample_tokens) == scene_dict["nbr_samples"]

        return sample_tokens

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])

        force_nearest_sweeps = self.dataset_cfg.get("FORCE_NEAREST_SWEEPS", False)
        # self.logger.info(f"max_sweeps: {self.dataset_cfg.MAX_SWEEPS}")
        # self.logger.info(f"force_nearest_sweeps: {force_nearest_sweeps}")

        points = self.get_lidar_with_sweeps(
            index,
            max_sweeps=self.dataset_cfg.MAX_SWEEPS,
            force_nearest_sweeps=force_nearest_sweeps,
        )

        # Noise with std 0.01
        # points[:, :3] += np.random.normal(0, 0.01, points[:, :3].shape)

        # Noise with std 0.1
        # points[:, :3] += np.random.normal(0, 0.1, points[:, :3].shape)

        # Random ray drop with ratio 0.1
        # mask = np.random.rand(points.shape[0]) > 0.1
        # points = points[mask]

        # Random ray drop with ratio 0.2
        # mask = np.random.rand(points.shape[0]) > 0.2
        # points = points[mask]

        # Get the frame index within the scene.
        # - sample_dict["scene_token"]: folder name
        # - frame_index: xxxx.npy
        sample_dict = self.nusc.get("sample", info["token"])
        sample_tokens_in_scene = self.get_sample_tokens_from_scene_token(
            sample_dict["scene_token"]
        )
        frame_index = sample_tokens_in_scene.index(info["token"])

        vis_points_sim = False
        if vis_points_sim:
            sim_frame_root = (
                Path.home() / "research/lit/data/nuscenes/08_kitti_sim_frame"
            )
            sim_frame_path = (
                sim_frame_root / sample_dict["scene_token"] / f"{frame_index:04d}.npy"
            )
            points_sim = np.load(sim_frame_path)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            pcd.paint_uniform_color([0, 0, 1])

            pcd_sim = o3d.geometry.PointCloud()
            pcd_sim.points = o3d.utility.Vector3dVector(points_sim)
            pcd_sim.paint_uniform_color([1, 0, 0])

            coords = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

            o3d.visualization.draw_geometries([pcd, pcd_sim, coords])

        if self.dataset_cfg.get("SHIFT_COOR", None):
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)

        # For purposes of getting points and poses, we only care about LIDAR_TOP.
        lidar_top_sample_data = self.nusc.get(
            "sample_data", sample_dict["data"]["LIDAR_TOP"]
        )
        lidar_top_ego_pose = self.nusc.get(
            "ego_pose", lidar_top_sample_data["ego_pose_token"]
        )
        lidar_top_calibrated_sensor = self.nusc.get(
            "calibrated_sensor", lidar_top_sample_data["calibrated_sensor_token"]
        )
        lidar_top_to_vehicle_pose = transform_matrix(
            lidar_top_calibrated_sensor["translation"],
            Quaternion(lidar_top_calibrated_sensor["rotation"]),
            inverse=False,
        )
        lidar_to_vehicle_poses = [lidar_top_to_vehicle_pose]

        # Frame pose (ego pose).
        frame_pose = transform_matrix(
            lidar_top_ego_pose["translation"],
            Quaternion(lidar_top_ego_pose["rotation"]),
            inverse=False,
        )

        sanity_check_different_ego_pose = False
        if sanity_check_different_ego_pose:
            # Note that the number of ego_pose records in our loaded database is
            # the same as the number of sample_data records. These two records
            # exhibit a one-to-one correspondence. All the sensors on the ego do
            # not necessarily take their respective readings at the exact same
            # time. So to be very accurate, every sensor reading has an ego_pose
            # associated with the timestamp the reading was taken at.
            #
            # This means that the "ego_pose" retrieved from "CAM_BACK" and
            # "LIDAR_TOP" will have different tokens but the value are very
            # similar.
            #
            # See: https://github.com/nutonomy/nuscenes-devkit/issues/744
            cam_back_sample_data = self.nusc.get(
                "sample_data", sample_dict["data"]["CAM_BACK"]
            )
            cam_back_ego_pose = self.nusc.get(
                "ego_pose", cam_back_sample_data["ego_pose_token"]
            )
            cam_back_calibrated_sensor = self.nusc.get(
                "calibrated_sensor", cam_back_sample_data["calibrated_sensor_token"]
            )
            np.testing.assert_allclose(
                lidar_top_ego_pose["rotation"],
                cam_back_ego_pose["rotation"],
                atol=1e-2,
                rtol=1e-3,
            )
            np.testing.assert_allclose(
                lidar_top_ego_pose["translation"],
                cam_back_ego_pose["translation"],
                atol=1e-2,
                rtol=1e-3,
            )

        input_dict = {
            "points": points,
            "frame_id": Path(info["lidar_path"]).stem,
            "metadata": {"token": info["token"]},
            "sample_idx": frame_index,  # Consistent with WaymoDataset.
        }

        if "gt_boxes" in info:
            # For each gt_box, get the instance token to identify the object.
            # The gt_box may be filtered by nuscenes_utils.py::fill_trainval_infos.
            instance_tokens = [
                self.nusc.get("sample_annotation", sample_annotation_token)[
                    "instance_token"
                ]
                for sample_annotation_token in info["gt_boxes_token"]
            ]
            # To be consistent to Waymo, we call it obj_ids.
            # Convert to numpy array first, for easy indexing.
            obj_ids = np.array(instance_tokens, dtype=np.str_)
            assert len(obj_ids) == len(info["gt_boxes"])

            if self.dataset_cfg.get("FILTER_MIN_POINTS_IN_GT", False):
                mask = (
                    info["num_lidar_pts"] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1
                )
            else:
                mask = None

            if mask is None:
                input_dict["gt_names"] = info["gt_names"]
                input_dict["gt_boxes"] = info["gt_boxes"]
                input_dict["obj_ids"] = obj_ids
            else:
                input_dict["gt_names"] = info["gt_names"][mask]
                input_dict["gt_boxes"] = info["gt_boxes"][mask]
                input_dict["obj_ids"] = obj_ids[mask]
                assert len(input_dict["gt_boxes"]) == len(input_dict["obj_ids"])

            if self.dataset_cfg.get("SHIFT_COOR", None):
                input_dict["gt_boxes"][:, 0:3] += self.dataset_cfg.SHIFT_COOR

            if self.dataset_cfg.get("USE_PSEUDO_LABEL", None) and self.training:
                input_dict["gt_boxes"] = None

            # for debug only
            # gt_boxes_mask = np.array([n in self.class_names for n in input_dict['gt_names']], dtype=np.bool_)
            # debug_dict = {'gt_boxes': copy.deepcopy(input_dict['gt_boxes'][gt_boxes_mask])}

        if self.dataset_cfg.get("FOV_POINTS_ONLY", None):
            input_dict["points"] = self.extract_fov_data(
                input_dict["points"],
                self.dataset_cfg.FOV_DEGREE,
                self.dataset_cfg.FOV_ANGLE,
            )
            if input_dict["gt_boxes"] is not None:
                fov_gt_flag = self.extract_fov_gt(
                    input_dict["gt_boxes"],
                    self.dataset_cfg.FOV_DEGREE,
                    self.dataset_cfg.FOV_ANGLE,
                )
                input_dict.update(
                    {
                        "gt_names": input_dict["gt_names"][fov_gt_flag],
                        "gt_boxes": input_dict["gt_boxes"][fov_gt_flag],
                        "obj_ids": input_dict["obj_ids"][fov_gt_flag],
                    }
                )
                assert len(input_dict["gt_boxes"]) == len(input_dict["obj_ids"])

        if self.dataset_cfg.get("USE_PSEUDO_LABEL", None) and self.training:
            self.fill_pseudo_labels(input_dict)

        if self.dataset_cfg.get(
            "SET_NAN_VELOCITY_TO_ZEROS", False
        ) and not self.dataset_cfg.get("USE_PSEUDO_LABEL", None):
            gt_boxes = input_dict["gt_boxes"]
            gt_boxes[np.isnan(gt_boxes)] = 0
            input_dict["gt_boxes"] = gt_boxes

        if (
            not self.dataset_cfg.PRED_VELOCITY
            and "gt_boxes" in input_dict
            and not self.dataset_cfg.get("USE_PSEUDO_LABEL", None)
        ):
            input_dict["gt_boxes"] = input_dict["gt_boxes"][:, [0, 1, 2, 3, 4, 5, 6]]

        # Insert properties first, as self.prepare_data may recursively call __getitem__.
        input_dict["pose"] = frame_pose
        input_dict["lidar_to_vehicle_poses"] = lidar_to_vehicle_poses

        data_dict = self.prepare_data(data_dict=input_dict)

        # Unlike waymo we do this after prepare_data, as we hard-code to use the
        # top lidar.
        data_dict["num_points_of_each_lidar"] = [len(data_dict["points"])]

        # Extra info.
        if self.include_extras:
            obj_ids = [str(obj_id) for obj_id in data_dict["obj_ids"]]
            data_dict["obj_ids"] = obj_ids
        else:
            if "obj_ids" in data_dict:
                data_dict.pop("obj_ids")

        # Rotate from y pointing to front to x pointing to front.
        # - Rotate points
        # - Rotate gt_boxes

        return data_dict

    def generate_prediction_dicts(
        self, batch_dict, pred_dicts, class_names, output_path=None
    ):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:
        Returns:
        """

        def get_template_prediction(num_samples):
            ret_dict = {
                "name": np.zeros(num_samples),
                "score": np.zeros(num_samples),
                "boxes_lidar": np.zeros([num_samples, 7]),
                "pred_labels": np.zeros(num_samples),
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict["pred_scores"].cpu().numpy()
            pred_boxes = box_dict["pred_boxes"].cpu().numpy()
            pred_labels = box_dict["pred_labels"].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            if self.dataset_cfg.get("SHIFT_COOR", None):
                pred_boxes[:, 0:3] -= self.dataset_cfg.SHIFT_COOR

            pred_dict["name"] = np.array(class_names)[pred_labels - 1]
            pred_dict["score"] = pred_scores
            pred_dict["boxes_lidar"] = pred_boxes
            pred_dict["pred_labels"] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict["frame_id"] = batch_dict["frame_id"][index]
            single_pred_dict["metadata"] = batch_dict["metadata"][index]
            annos.append(single_pred_dict)

        return annos

    def kitti_eval(self, eval_det_annos, eval_gt_annos, class_names):
        from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

        map_name_to_kitti = {
            "car": "Car",
            "pedestrian": "Pedestrian",
            "truck": "Truck",
        }

        def transform_to_kitti_format(annos, info_with_fakelidar=False, is_gt=False):
            for anno in annos:
                if "name" not in anno:
                    anno["name"] = anno["gt_names"]
                    anno.pop("gt_names")

                for k in range(anno["name"].shape[0]):
                    if anno["name"][k] in map_name_to_kitti:
                        anno["name"][k] = map_name_to_kitti[anno["name"][k]]
                    else:
                        anno["name"][k] = "Person_sitting"

                if "boxes_lidar" in anno:
                    gt_boxes_lidar = anno["boxes_lidar"].copy()
                else:
                    gt_boxes_lidar = anno["gt_boxes"].copy()

                # filter by fov
                if is_gt and self.dataset_cfg.get("GT_FILTER", None):
                    if self.dataset_cfg.GT_FILTER.get("FOV_FILTER", None):
                        fov_gt_flag = self.extract_fov_gt(
                            gt_boxes_lidar,
                            self.dataset_cfg["FOV_DEGREE"],
                            self.dataset_cfg["FOV_ANGLE"],
                        )
                        gt_boxes_lidar = gt_boxes_lidar[fov_gt_flag]
                        anno["name"] = anno["name"][fov_gt_flag]

                anno["bbox"] = np.zeros((len(anno["name"]), 4))
                anno["bbox"][:, 2:4] = 50  # [0, 0, 50, 50]
                anno["truncated"] = np.zeros(len(anno["name"]))
                anno["occluded"] = np.zeros(len(anno["name"]))

                if len(gt_boxes_lidar) > 0:
                    if info_with_fakelidar:
                        gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(
                            gt_boxes_lidar
                        )

                    gt_boxes_lidar[:, 2] -= gt_boxes_lidar[:, 5] / 2
                    anno["location"] = np.zeros((gt_boxes_lidar.shape[0], 3))
                    anno["location"][:, 0] = -gt_boxes_lidar[:, 1]  # x = -y_lidar
                    anno["location"][:, 1] = -gt_boxes_lidar[:, 2]  # y = -z_lidar
                    anno["location"][:, 2] = gt_boxes_lidar[:, 0]  # z = x_lidar
                    dxdydz = gt_boxes_lidar[:, 3:6]
                    anno["dimensions"] = dxdydz[:, [0, 2, 1]]  # lwh ==> lhw
                    anno["rotation_y"] = -gt_boxes_lidar[:, 6] - np.pi / 2.0
                    anno["alpha"] = (
                        -np.arctan2(-gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0])
                        + anno["rotation_y"]
                    )
                else:
                    anno["location"] = anno["dimensions"] = np.zeros((0, 3))
                    anno["rotation_y"] = anno["alpha"] = np.zeros(0)

        transform_to_kitti_format(eval_det_annos)
        transform_to_kitti_format(eval_gt_annos, info_with_fakelidar=False, is_gt=True)

        kitti_class_names = []
        for x in class_names:
            if x in map_name_to_kitti:
                kitti_class_names.append(map_name_to_kitti[x])
            else:
                kitti_class_names.append("Person_sitting")
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
            gt_annos=eval_gt_annos,
            dt_annos=eval_det_annos,
            current_classes=kitti_class_names,
        )
        return ap_result_str, ap_dict

    def nuscene_eval(self, det_annos, class_names, **kwargs):
        import json

        from nuscenes.nuscenes import NuScenes

        from pcdet.datasets.nuscenes import nuscenes_utils

        nusc = NuScenes(
            version=self.dataset_cfg.VERSION, dataroot=str(self.root_path), verbose=True
        )
        nusc_annos = nuscenes_utils.transform_det_annos_to_nusc_annos(det_annos, nusc)
        nusc_annos["meta"] = {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }

        output_path = Path(kwargs["output_path"])
        output_path.mkdir(exist_ok=True, parents=True)
        res_path = str(output_path / "results_nusc.json")
        with open(res_path, "w") as f:
            json.dump(nusc_annos, f)

        self.logger.info(f"The predictions of NuScenes have been saved to {res_path}")

        if self.dataset_cfg.VERSION == "v1.0-test":
            return "No ground-truth annotations for evaluation", {}

        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import NuScenesEval

        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
            "v1.0-test": "test",
        }
        try:
            eval_version = "detection_cvpr_2019"
            eval_config = config_factory(eval_version)
        except:
            eval_version = "cvpr_2019"
            eval_config = config_factory(eval_version)

        nusc_eval = NuScenesEval(
            nusc,
            config=eval_config,
            result_path=res_path,
            eval_set=eval_set_map[self.dataset_cfg.VERSION],
            output_dir=str(output_path),
            verbose=True,
        )
        metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)

        with open(output_path / "metrics_summary.json", "r") as f:
            metrics = json.load(f)

        result_str, result_dict = nuscenes_utils.format_nuscene_results(
            metrics, self.class_names, version=eval_version
        )

        return result_str, result_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        if kwargs["eval_metric"] == "kitti":
            eval_det_annos = copy.deepcopy(det_annos)
            eval_gt_annos = copy.deepcopy(self.infos)
            return self.kitti_eval(eval_det_annos, eval_gt_annos, class_names)
        elif kwargs["eval_metric"] == "nuscenes":
            return self.nuscene_eval(det_annos, class_names, **kwargs)
        else:
            raise NotImplementedError

    def create_groundtruth_database(self, used_classes=None, max_sweeps=10):
        import torch

        database_save_path = self.root_path / f"gt_database_{max_sweeps}sweeps_withvelo"
        db_info_save_path = (
            self.root_path / f"nuscenes_dbinfos_{max_sweeps}sweeps_withvelo.pkl"
        )

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        for idx in tqdm(range(len(self.infos))):
            sample_idx = idx
            info = self.infos[idx]
            points = self.get_lidar_with_sweeps(idx, max_sweeps=max_sweeps)
            gt_boxes = info["gt_boxes"]
            gt_names = info["gt_names"]

            box_idxs_of_pts = (
                roiaware_pool3d_utils.points_in_boxes_gpu(
                    torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                    torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda(),
                )
                .long()
                .squeeze(dim=0)
                .cpu()
                .numpy()
            )

            for i in range(gt_boxes.shape[0]):
                filename = "%s_%s_%d.bin" % (sample_idx, gt_names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, "w") as f:
                    gt_points.tofile(f)

                if (used_classes is None) or gt_names[i] in used_classes:
                    db_path = str(
                        filepath.relative_to(self.root_path)
                    )  # gt_database/xxxxx.bin
                    db_info = {
                        "name": gt_names[i],
                        "path": db_path,
                        "image_idx": sample_idx,
                        "gt_idx": i,
                        "box3d_lidar": gt_boxes[i],
                        "num_points_in_gt": gt_points.shape[0],
                    }
                    if gt_names[i] in all_db_infos:
                        all_db_infos[gt_names[i]].append(db_info)
                    else:
                        all_db_infos[gt_names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print("Database %s: %d" % (k, len(v)))

        with open(db_info_save_path, "wb") as f:
            pickle.dump(all_db_infos, f)


def create_nuscenes_info(version, data_path, save_path, max_sweeps=10):
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits

    from pcdet.datasets.nuscenes import nuscenes_utils

    data_path = data_path / version
    save_path = save_path / version

    assert version in ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
    if version == "v1.0-trainval":
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == "v1.0-test":
        train_scenes = splits.test
        val_scenes = []
    elif version == "v1.0-mini":
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise NotImplementedError

    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    available_scenes = nuscenes_utils.get_available_scenes(nusc)
    available_scene_names = [s["name"] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set(
        [
            available_scenes[available_scene_names.index(s)]["token"]
            for s in train_scenes
        ]
    )
    val_scenes = set(
        [available_scenes[available_scene_names.index(s)]["token"] for s in val_scenes]
    )

    print(
        "%s: train scene(%d), val scene(%d)"
        % (version, len(train_scenes), len(val_scenes))
    )

    train_nusc_infos, val_nusc_infos = nuscenes_utils.fill_trainval_infos(
        data_path=data_path,
        nusc=nusc,
        train_scenes=train_scenes,
        val_scenes=val_scenes,
        test="test" in version,
        max_sweeps=max_sweeps,
    )

    if version == "v1.0-test":
        print("test sample: %d" % len(train_nusc_infos))
        with open(save_path / f"nuscenes_infos_{max_sweeps}sweeps_test.pkl", "wb") as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print(
            "train sample: %d, val sample: %d"
            % (len(train_nusc_infos), len(val_nusc_infos))
        )
        with open(
            save_path / f"nuscenes_infos_{max_sweeps}sweeps_train.pkl", "wb"
        ) as f:
            pickle.dump(train_nusc_infos, f)
        with open(save_path / f"nuscenes_infos_{max_sweeps}sweeps_val.pkl", "wb") as f:
            pickle.dump(val_nusc_infos, f)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import yaml
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file", type=str, default=None, help="specify the config of dataset"
    )
    parser.add_argument("--func", type=str, default="create_nuscenes_infos", help="")
    parser.add_argument("--version", type=str, default="v1.0-trainval", help="")
    args = parser.parse_args()

    if args.func == "create_nuscenes_infos":
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / "../../../").resolve()
        dataset_cfg.VERSION = args.version
        create_nuscenes_info(
            version=dataset_cfg.VERSION,
            data_path=ROOT_DIR / "data" / "nuscenes",
            save_path=ROOT_DIR / "data" / "nuscenes",
            max_sweeps=dataset_cfg.MAX_SWEEPS,
        )

        nuscenes_dataset = NuScenesDataset(
            dataset_cfg=dataset_cfg,
            class_names=None,
            root_path=ROOT_DIR / "data" / "nuscenes",
            logger=common_utils.create_logger(),
            training=True,
        )
        nuscenes_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS)
        nuscenes_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS)
