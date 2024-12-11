# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.

import argparse
import copy
import multiprocessing
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lit.path_utils import get_lit_paths
from pcdet.datasets.dataset import DatasetTemplate
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import box_utils, common_utils


class WaymoDataset(DatasetTemplate):
    _all_lidar_names = ["TOP", "FRONT", "SIDE_LEFT", "SIDE_RIGHT", "REAR"]

    def __init__(
        self,
        dataset_cfg,
        class_names,
        training=True,
        root_path=None,
        logger=None,
        allow_empty_gt_boxes=False,
        include_extras=False,
    ):
        """
        Args:
            include_extras: include extra info in __getitem__ which may not be
                able to transfer to CUDA. This is only used for extracting
                Scene data.
        """
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
            allow_empty_gt_boxes=allow_empty_gt_boxes,
        )
        self.include_extras = include_extras

        self.data_path = self.root_path / self.dataset_cfg.PROCESSED_DATA_TAG
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]

        split_dir = (
            self.root_path.parent.parent
            / "data_split"
            / "waymo"
            / (self.split + ".txt")
        )
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]

        self.infos = []
        self.extra_infos = []
        self.include_waymo_data(self.mode)

        # Check if we shall use simulated data.
        # E.g., set dataset_cfg.DST_STYLE to "kitti" to use simulated
        # nuScenes data with KITTI style.
        self.dst_style = self.dataset_cfg.get("DST_STYLE", None)

        # Get Waymo dataset paths.
        if self.dst_style is not None:
            lit_paths = get_lit_paths(
                data_version=self.dataset_cfg.DATA_VERSION,
                data_domain="waymo",
            )

        # Not all training frames have simulated data, thus the self.infos may
        # not have the same length as before.
        if self.dst_style == "kitti":
            new_infos = []

            # Ref init_paths.py
            self.dst_data_path = lit_paths.to_kitti_sim_frame_dir

            # Only a subset of self.infos has a corresponding simulation in
            # the sim_frame_dir. That is, we shall check
            # self.infos[i]["point_cloud"]["lidar_sequence"]
            # and see if it has a corresponding folder in sim_frame_dir.
            new_infos = []
            new_extra_infos = []
            for info, extra_info in zip(self.infos, self.extra_infos):
                sequence_name = info["point_cloud"]["lidar_sequence"]
                if sequence_name in lit_paths.scene_list:
                    sim_dir = self.dst_data_path / sequence_name
                    if not sim_dir.exists():
                        print(
                            f"[WARNING] {sim_dir} does not exist but it "
                            f"is in scene_list for data version "
                            f"{lit_paths.data_version}"
                        )
                    new_infos.append(info)
                    new_extra_infos.append(extra_info)

            print(f"Selected {len(new_infos)} from {len(self.infos)} infos")
            self.infos = new_infos
            self.extra_infos = new_extra_infos

        elif self.dst_style == "nuscenes":
            new_infos = []

            # Ref init_paths.py
            self.dst_data_path = lit_paths.to_nuscenes_sim_frame_dir

            # Only a subset of self.infos has a corresponding simulation in
            # the sim_frame_dir. That is, we shall check
            # self.infos[i]["point_cloud"]["lidar_sequence"]
            # and see if it has a corresponding folder in sim_frame_dir.
            new_infos = []
            new_extra_infos = []
            for info, extra_info in zip(self.infos, self.extra_infos):
                sequence_name = info["point_cloud"]["lidar_sequence"]
                if sequence_name in lit_paths.scene_list:
                    sim_dir = self.dst_data_path / sequence_name
                    if not sim_dir.exists():
                        print(
                            f"[WARNING] "
                            f"{sim_dir} does not exist but it "
                            f"is in scene_list for data version "
                            f"{lit_paths.data_version}"
                        )
                    new_infos.append(info)
                    new_extra_infos.append(extra_info)

            print(f"Selected {len(new_infos)} from {len(self.infos)} infos")
            self.infos = new_infos
            self.extra_infos = new_extra_infos

        elif self.dst_style == "v0":
            self.dst_data_path = None
            # print(f"self.info before subsampling: {len(self.infos)}")
            # self.infos = self.infos[::5]
            # self.extra_infos = self.extra_infos[::5]
            # print(f"self.info after subsampling : {len(self.infos)}")

        elif self.dst_style == None:
            self.dst_data_path = None

        else:
            raise ValueError(f"dst_style {self.dst_style} not implemented")

        # For debugging, clip the info to first 128.
        # self.infos = self.infos[:128]
        # self.extra_infos = self.extra_infos[:128]
        # self.infos = self.infos[: int(len(self.infos) * 0.1)]
        # self.extra_infos = self.extra_infos[: int(len(self.extra_infos) * 0.1)]

    def set_split(self, split):
        """
        Only needed if you want to change the split after initialization.
        """
        super().__init__(
            dataset_cfg=self.dataset_cfg,
            class_names=self.class_names,
            training=self.training,
            root_path=self.root_path,
            logger=self.logger,
        )
        self.split = split
        split_dir = (
            self.root_path.parent.parent / "data_split" / "waymo" / f"{self.split}.txt"
        )
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        self.infos = []
        self.extra_infos = []
        self.include_waymo_data(self.mode)

    def include_waymo_data(self, mode):
        self.logger.info("Loading Waymo dataset")
        waymo_infos = []
        extra_waymo_infos = []

        num_skipped_infos = 0
        for k in range(len(self.sample_sequence_list)):
            # Load infos.
            sequence_name = os.path.splitext(self.sample_sequence_list[k])[0]
            info_path = self.data_path / sequence_name / ("%s.pkl" % sequence_name)
            info_path = self.check_sequence_name_with_all_version(info_path)
            if not info_path.exists():
                num_skipped_infos += 1
                continue  # This also skips the extra infos.
            with open(info_path, "rb") as f:
                infos = pickle.load(f)
                waymo_infos.extend(infos)

            # Load extra infos.
            # For simplicity, we assume that the extra infos are always available.
            extra_info_dir = self.data_path.parent / f"{self.data_path.name}_extra"
            extra_info_path = extra_info_dir / f"{sequence_name}.pkl"
            if not extra_info_path.exists():
                # TODO: enable this check after pre-processing all data.
                # raise FileNotFoundError(
                #     f"{info_path} exists but {extra_info_path} does not exist."
                # )
                continue
            with open(extra_info_path, "rb") as f:
                extra_infos = pickle.load(f)
                extra_waymo_infos.extend(extra_infos)

        self.infos.extend(waymo_infos[:])
        self.extra_infos.extend(extra_waymo_infos[:])

        self.logger.info("Total skipped info %s" % num_skipped_infos)
        self.logger.info("Total samples for Waymo dataset: %d" % (len(waymo_infos)))

        if self.dataset_cfg.SAMPLED_INTERVAL[mode] > 1:
            sampled_waymo_infos = []
            for k in range(0, len(self.infos), self.dataset_cfg.SAMPLED_INTERVAL[mode]):
                sampled_waymo_infos.append(self.infos[k])
            self.infos = sampled_waymo_infos
            self.logger.info(
                "Total sampled samples for Waymo dataset: %d" % len(self.infos)
            )

            sampled_extra_waymo_infos = []
            for k in range(
                0, len(self.extra_infos), self.dataset_cfg.SAMPLED_INTERVAL[mode]
            ):
                sampled_extra_waymo_infos.append(self.extra_infos[k])
            self.extra_infos = sampled_extra_waymo_infos
            self.logger.info(
                "Total sampled extra samples for Waymo dataset: %d"
                % len(self.extra_infos)
            )

    @staticmethod
    def check_sequence_name_with_all_version(sequence_file):
        if (
            "_with_camera_labels" not in str(sequence_file)
            and not sequence_file.exists()
        ):
            sequence_file = Path(
                str(sequence_file)[:-9] + "_with_camera_labels.tfrecord"
            )
        if "_with_camera_labels" in str(sequence_file) and not sequence_file.exists():
            sequence_file = Path(str(sequence_file).replace("_with_camera_labels", ""))

        return sequence_file

    def get_infos(
        self,
        raw_data_path,
        save_path,
        num_workers=None,
        has_label=True,
        sampled_interval=1,
        enable_only_save_lidar_poses=False,
    ):
        """
        num_workers: int, number of threads to be used for data preprocessing.
            - None: use all CPU cores.
            - 0: no parallelization.
            - 1: parallelization with one thread, which is different from 0.
        """
        import concurrent.futures as futures
        from functools import partial

        from pcdet.datasets.waymo import waymo_utils

        print(
            "---------------The waymo sample interval is %d, total sequecnes is %d-----------------"
            % (sampled_interval, len(self.sample_sequence_list))
        )

        process_single_sequence = partial(
            waymo_utils.process_single_sequence,
            save_path=save_path,
            sampled_interval=sampled_interval,
            has_label=has_label,
            enable_only_save_lidar_poses=enable_only_save_lidar_poses,
        )
        sample_sequence_file_list = [
            self.check_sequence_name_with_all_version(raw_data_path / sequence_file)
            for sequence_file in self.sample_sequence_list
        ]

        if num_workers is None:
            num_workers = multiprocessing.cpu_count()

        if num_workers > 0:
            with futures.ThreadPoolExecutor(num_workers) as executor:
                sequence_infos = list(
                    tqdm(
                        executor.map(
                            process_single_sequence, sample_sequence_file_list
                        ),
                        total=len(sample_sequence_file_list),
                    )
                )
            # Equivalent to:
            # for infos in sequence_infos:
            #    all_sequences_infos.extend(infos)
            all_sequences_infos = [item for infos in sequence_infos for item in infos]
        else:
            # Single thread.
            all_sequences_infos = []
            for sequence_file in tqdm(sample_sequence_file_list):
                sequence_info = process_single_sequence(sequence_file)
                all_sequences_infos.extend(sequence_info)

        return all_sequences_infos

    def get_lidar(self, sequence_name, sample_idx, info):
        """
        Returns:
            points_all: (N, 5) [x, y, z, intensity, elongation]
            num_points_of_each_lidar: (5,), the updated info["num_points_of_each_lidar"]
        """
        if self.dst_data_path is None:
            # (N, 6): [x, y, z, intensity, elongation, NLZ_flag]
            lidar_file = self.data_path / sequence_name / ("%04d.npy" % sample_idx)

            point_features = np.load(lidar_file)

            # For each lidar, compute start/end indices.
            num_lidars = len(self._all_lidar_names)
            num_points_of_each_lidar = info["num_points_of_each_lidar"]
            assert len(num_points_of_each_lidar) == num_lidars
            end_indices = np.cumsum(num_points_of_each_lidar)  # Exclusive.
            start_indices = np.concatenate([[0], end_indices[:-1]])  # Inclusive.

            # For each lidar, compute the number of points with NLZ_flag == -1
            points_all, NLZ_flag = point_features[:, 0:5], point_features[:, 5]
            num_points_of_each_lidar = [
                np.sum(
                    NLZ_flag[start_indices[lidar_idx] : end_indices[lidar_idx]] == -1
                )
                for lidar_idx in range(num_lidars)
            ]

            # Filter all with NLZ_flag == -1
            points_all = points_all[NLZ_flag == -1]
            points_all[:, 3] = np.tanh(points_all[:, 3])

            if len(points_all) != np.sum(num_points_of_each_lidar):
                raise ValueError(
                    f"len(points_all) != np.sum(num_points_of_each_lidar): "
                    f"{len(points_all)} != {np.sum(num_points_of_each_lidar)}"
                )

        else:
            lidar_file_npy = (
                self.dst_data_path / sequence_name / ("%04d.npy" % sample_idx)
            )
            lidar_file_npz = (
                self.dst_data_path / sequence_name / ("%04d.npz" % sample_idx)
            )

            if lidar_file_npy.is_file():
                # (N, 3)
                point_features = np.load(lidar_file_npy)
            elif lidar_file_npz.is_file():
                # (N, 3)
                point_features = np.load(lidar_file_npz)["local_points"]
            else:
                raise (f"Both {lidar_file_npy} and {lidar_file_npz} are not found")

            # (N, 3) -> (N, 5) by appending intensity and elongation as zeros.
            points_all = np.concatenate(
                [point_features, np.zeros([len(point_features), 2])], axis=1
            )

            # All points are from the top lidar.
            num_points_of_each_lidar = [len(points_all), 0, 0, 0, 0]

        return points_all, num_points_of_each_lidar

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        extra_info = copy.deepcopy(self.extra_infos[index])

        pc_info = info["point_cloud"]
        sequence_name = pc_info["lidar_sequence"]
        sample_idx = pc_info["sample_idx"]
        points, num_points_of_each_lidar = self.get_lidar(
            sequence_name, sample_idx, info
        )

        # Sanity checks.
        assert pc_info["lidar_sequence"] == extra_info["sequence_name"]
        assert pc_info["sample_idx"] == extra_info["sample_idx"]
        np.testing.assert_allclose(info["pose"], extra_info["frame_pose"])

        # Per-lidar pose.
        num_lidars = len(num_points_of_each_lidar)
        lidar_to_vehicle_poses = extra_info["lidar_to_vehicle_poses"]
        assert len(lidar_to_vehicle_poses) == num_lidars

        input_dict = {
            "points": points,
            "frame_id": info["frame_id"],
            "sample_idx": sample_idx,
        }

        if "annos" in info:
            annos = info["annos"]
            annos = common_utils.drop_info_with_name(annos, name="unknown")

            if self.dataset_cfg.get("INFO_WITH_FAKELIDAR", False):
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(
                    annos["gt_boxes_lidar"]
                )
            else:
                gt_boxes_lidar = annos["gt_boxes_lidar"]

            input_dict.update(
                {
                    "gt_names": annos["name"],
                    "gt_boxes": gt_boxes_lidar,
                    "num_points_in_gt": annos.get("num_points_in_gt", None),
                }
            )

            if self.dataset_cfg.get("USE_PSEUDO_LABEL", None) and self.training:
                input_dict["gt_boxes"] = None

            if self.dataset_cfg.get("USE_PSEUDO_LABEL", None) and self.training:
                input_dict["gt_boxes"] = None

            # for debug only
            # gt_boxes_mask = np.array([n in self.class_names for n in input_dict['gt_names']], dtype=np.bool_)
            # debug_dict = {'gt_boxes': copy.deepcopy(gt_boxes_lidar[gt_boxes_mask])}

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
                        "num_points_in_gt": (
                            input_dict["num_points_in_gt"][fov_gt_flag]
                            if input_dict["num_points_in_gt"] is not None
                            else None
                        ),
                    }
                )

        # load saved pseudo label for unlabeled data
        if self.dataset_cfg.get("USE_PSEUDO_LABEL", None) and self.training:
            self.fill_pseudo_labels(input_dict)

        # Insert properties first, as self.prepare_data may recursively call __getitem__.
        input_dict["sequence_name"] = sequence_name
        input_dict["pose"] = info["pose"]
        input_dict["lidar_to_vehicle_poses"] = lidar_to_vehicle_poses
        input_dict["num_points_of_each_lidar"] = num_points_of_each_lidar  # Updated
        input_dict["obj_ids"] = annos["obj_ids"]

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict["metadata"] = info.get("metadata", info["frame_id"])
        data_dict.pop("num_points_in_gt", None)

        # Extra info.
        if self.include_extras:
            obj_ids = [str(obj_id) for obj_id in data_dict["obj_ids"]]
            data_dict["obj_ids"] = obj_ids
        else:
            if "obj_ids" in data_dict:
                data_dict.pop("obj_ids")
            if "metadata" in data_dict:
                data_dict.pop("metadata")
            if "sequence_name" in data_dict:
                data_dict.pop("sequence_name")

        return data_dict

    @staticmethod
    def generate_prediction_dicts(
        batch_dict, pred_dicts, class_names, output_path=None
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
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict["pred_scores"].cpu().numpy()
            pred_boxes = box_dict["pred_boxes"].cpu().numpy()
            pred_labels = box_dict["pred_labels"].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict["name"] = np.array(class_names)[pred_labels - 1]
            pred_dict["score"] = pred_scores
            pred_dict["boxes_lidar"] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict["frame_id"] = batch_dict["frame_id"][index]
            single_pred_dict["metadata"] = batch_dict["metadata"][index]
            annos.append(single_pred_dict)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if "annos" not in self.infos[0].keys():
            return "No ground-truth boxes for evaluation", {}

        def kitti_eval(eval_det_annos, eval_gt_annos):
            from pcdet.datasets.kitti import kitti_utils
            from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

            map_name_to_kitti = {
                "Vehicle": "Car",
                "Pedestrian": "Pedestrian",
                "Cyclist": "Cyclist",
                "Sign": "Sign",
                "Car": "Car",
            }
            kitti_utils.transform_annotations_to_kitti_format(
                eval_det_annos, map_name_to_kitti=map_name_to_kitti
            )
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos,
                map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get("INFO_WITH_FAKELIDAR", False),
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos,
                dt_annos=eval_det_annos,
                current_classes=kitti_class_names,
            )
            return ap_result_str, ap_dict

        def waymo_eval(eval_det_annos, eval_gt_annos):
            from pcdet.datasets.waymo.waymo_eval import (
                OpenPCDetWaymoDetectionMetricsEstimator,
            )

            eval = OpenPCDetWaymoDetectionMetricsEstimator()

            ap_dict = eval.waymo_evaluation(
                eval_det_annos,
                eval_gt_annos,
                class_name=class_names,
                distance_thresh=1000,
                fake_gt_infos=self.dataset_cfg.get("INFO_WITH_FAKELIDAR", False),
            )
            ap_result_str = "\n"
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0]
                ap_result_str += "%s: %.4f \n" % (key, ap_dict[key])

            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info["annos"]) for info in self.infos]

        if kwargs["eval_metric"] == "kitti":
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos)
        elif kwargs["eval_metric"] == "waymo":
            ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict

    def create_groundtruth_database(
        self,
        info_path,
        save_path,
        used_classes=None,
        split="train",
        sampled_interval=10,
        processed_data_tag=None,
    ):
        database_save_path = save_path / (
            "pcdet_gt_database_%s_sampled_%d" % (split, sampled_interval)
        )
        db_info_save_path = save_path / (
            "pcdet_waymo_dbinfos_%s_sampled_%d.pkl" % (split, sampled_interval)
        )

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, "rb") as f:
            infos = pickle.load(f)

        for k in range(0, len(infos), sampled_interval):
            print("gt_database sample: %d/%d" % (k + 1, len(infos)))
            info = infos[k]

            pc_info = info["point_cloud"]
            sequence_name = pc_info["lidar_sequence"]
            sample_idx = pc_info["sample_idx"]
            points = self.get_lidar(sequence_name, sample_idx)

            annos = info["annos"]
            names = annos["name"]
            difficulty = annos["difficulty"]
            gt_boxes = annos["gt_boxes_lidar"]

            num_obj = gt_boxes.shape[0]

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

            for i in range(num_obj):
                filename = "%s_%04d_%s_%d.bin" % (
                    sequence_name,
                    sample_idx,
                    names[i],
                    i,
                )
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]
                gt_points[:, :3] -= gt_boxes[i, :3]

                if (used_classes is None) or names[i] in used_classes:
                    with open(filepath, "w") as f:
                        gt_points.tofile(f)

                    db_path = str(
                        filepath.relative_to(self.root_path)
                    )  # gt_database/xxxxx.bin
                    db_info = {
                        "name": names[i],
                        "path": db_path,
                        "sequence_name": sequence_name,
                        "sample_idx": sample_idx,
                        "gt_idx": i,
                        "box3d_lidar": gt_boxes[i],
                        "num_points_in_gt": gt_points.shape[0],
                        "difficulty": difficulty[i],
                    }
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print("Database %s: %d" % (k, len(v)))

        with open(db_info_save_path, "wb") as f:
            pickle.dump(all_db_infos, f)


class MixedWaymoDataset(DatasetTemplate):
    """
    Two Waymo Datasets mixed together.

    - dataset_a: The base dataset. This dataset will be randomly sampled
                 according to the MIX_RATIO_A_TO_B.
    - dataset_b: The main dataset. Each epoch will make sure that all samples
                 of dataset_b are used exactly once.
    """

    def __init__(
        self,
        dataset_cfg,
        class_names,
        training=True,
        root_path=None,
        logger=None,
        allow_empty_gt_boxes=False,
        include_extras=False,
    ):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
            allow_empty_gt_boxes=allow_empty_gt_boxes,
        )

        dst_style = self.dataset_cfg.get("DST_STYLE", None)

        # To fill
        self.dataset_a = None
        self.dataset_b = None
        self.dataset_a_sampled_len = None
        self.dataset_len = None

        if dst_style == "v0_kitti":
            dataset_cfg_a = copy.deepcopy(dataset_cfg)
            dataset_cfg_a.DST_STYLE = "v0"
            self.dataset_a = WaymoDataset(
                dataset_cfg=dataset_cfg_a,
                class_names=class_names,
                training=training,
                root_path=root_path,
                logger=logger,
                allow_empty_gt_boxes=allow_empty_gt_boxes,
                include_extras=include_extras,
            )

            dataset_cfg_b = copy.deepcopy(dataset_cfg)
            dataset_cfg_b.DST_STYLE = "kitti"
            self.dataset_b = WaymoDataset(
                dataset_cfg=dataset_cfg_b,
                class_names=class_names,
                training=training,
                root_path=root_path,
                logger=logger,
                allow_empty_gt_boxes=allow_empty_gt_boxes,
                include_extras=include_extras,
            )

            # Compute length of the mixed dataset
            mix_ratio_a_to_b = self.dataset_cfg.get("MIX_RATIO_A_TO_B", None)
            if mix_ratio_a_to_b is None:
                raise ValueError("DATA_CONFIG.MIX_RATIO_A_TO_B is not set")
            self.dataset_a_sampled_len = int(len(self.dataset_b) * mix_ratio_a_to_b)
            self.dataset_len = self.dataset_a_sampled_len + len(self.dataset_b)

            print(f"Initialized MixedWaymoDataset:")
            print(f"  - dst_style             : {dst_style}")
            print(f"  - Ratio A:B             : {mix_ratio_a_to_b}")
            print(f"  - Samples from A (v0)   : {len(self.dataset_a)}")
            print(f"    Sampled for A         : {self.dataset_a_sampled_len}")
            print(f"  - Samples from B (KITTI): {len(self.dataset_b)}")
            print(f"  - Mixed Dataset Length  : {self.dataset_len}")
        else:
            raise NotImplementedError(f"dst_style {self.dst_style} not implemented")

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        if index < self.dataset_a_sampled_len:
            index_a = np.random.randint(0, len(self.dataset_a))
            # print(f"dataset[{index}]: dataset_a[{index_a}]")
            return self.dataset_a[index_a]
        else:
            index_b = index - self.dataset_a_sampled_len
            # print(f"dataset[{index}]: dataset_b[{index_b}]")
            return self.dataset_b[index_b]


class WaymoDatasetInfo(WaymoDataset):
    def __init__(
        self,
        dataset_cfg,
        class_names,
        training=True,
        root_path=None,
        logger=None,
    ):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
        )

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        pc_info = info["point_cloud"]
        sequence_name = pc_info["lidar_sequence"]
        sample_idx = pc_info["sample_idx"]

        input_dict = {
            "frame_id": info["frame_id"],
            "sample_idx": sample_idx,
            "sequence_name": sequence_name,
        }
        input_dict["metadata"] = info.get("metadata", info["frame_id"])
        return input_dict


def create_waymo_infos(
    dataset_cfg,
    class_names,
    data_path,
    save_path,
    raw_data_tag="raw_data",
    processed_data_tag="waymo_processed_data",
    workers=None,
    disable_train_info=False,
    disable_val_info=False,
    disable_gt_database=False,
    enable_only_save_lidar_poses=False,
):
    dataset = WaymoDataset(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=data_path,
        training=False,
        logger=common_utils.create_logger(),
    )
    train_split, val_split = "train", "val"

    if enable_only_save_lidar_poses:
        train_filename = save_path / f"waymo_infos_{train_split}_extra.pkl"
        val_filename = save_path / f"waymo_infos_{val_split}_extra.pkl"
    else:
        train_filename = save_path / f"waymo_infos_{train_split}.pkl"
        val_filename = save_path / f"waymo_infos_{val_split}.pkl"

    ############################################################################
    # Per-frame points (npy), per-sequence info (pkl), per-split info (pkl),
    # for training dataset.
    ############################################################################
    if not disable_train_info:
        print("===============================================================")
        print("Generating per-frame points, per-seq info, per-split info (train)")
        print("===============================================================")
        dataset.set_split(train_split)
        waymo_infos_train = dataset.get_infos(
            raw_data_path=data_path / raw_data_tag,
            save_path=save_path / processed_data_tag,
            num_workers=workers,
            has_label=True,
            sampled_interval=1,
            enable_only_save_lidar_poses=enable_only_save_lidar_poses,
        )
        with open(train_filename, "wb") as f:
            pickle.dump(waymo_infos_train, f)
        print(f"Waymo info train file is saved to {train_filename}")
    else:
        print("===============================================================")
        print("Skip generating per-frame points, per-seq info, per-split info (train)")
        print("===============================================================")

    ############################################################################
    # Per-frame points (npy), per-sequence info (pkl), per-split info (pkl),
    # for validation dataset.
    ############################################################################
    if not disable_val_info:
        print("===============================================================")
        print("Generating per-frame points, per-seq info, per-split info (val)")
        print("===============================================================")
        dataset.set_split(val_split)
        waymo_infos_val = dataset.get_infos(
            raw_data_path=data_path / raw_data_tag,
            save_path=save_path / processed_data_tag,
            num_workers=workers,
            has_label=True,
            sampled_interval=1,
            enable_only_save_lidar_poses=enable_only_save_lidar_poses,
        )
        with open(val_filename, "wb") as f:
            pickle.dump(waymo_infos_val, f)
        print(f"Waymo info val file is saved to {val_filename}")
    else:
        print("===============================================================")
        print("Skip generating per-frame points, per-seq info, per-split info (val)")
        print("===============================================================")

    ############################################################################
    # Ground-truth database for data augmentation.
    ############################################################################
    if not disable_gt_database and not enable_only_save_lidar_poses:
        print("===============================================================")
        print("Generating ground-truth database (skip-10) for data augmentation")
        print("===============================================================")
        dataset.set_split(train_split)
        dataset.create_groundtruth_database(
            info_path=train_filename,
            save_path=save_path,
            split="train",
            sampled_interval=10,
            used_classes=["Vehicle", "Pedestrian", "Cyclist"],
        )
    else:
        print("===============================================================")
        print("Skip generating ground-truth database (skip-10) for data augmentation")
        print("===============================================================")

    print("===============================================================")
    print("Data preparation Done")
    print("===============================================================")


def main():
    parser = argparse.ArgumentParser(description="arg parser")

    # Default arguments.
    parser.add_argument(
        "--cfg_file",
        type=str,
        default=None,
        help="specify the config of dataset",
    )
    parser.add_argument(
        "--func",
        type=str,
        default="create_waymo_infos",
        help="",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="None for using all CPU cores; 0 for disable parallel",
    )

    # Arguments to disable part of the pipeline.
    parser.add_argument(
        "--disable_train_info",
        action="store_true",
        help="disable generating training info",
    )
    parser.add_argument(
        "--disable_val_info",
        action="store_true",
        help="disable generating validation info",
    )
    parser.add_argument(
        "--disable_gt_database",
        action="store_true",
        help="disable generating gt database (skip-10) for augmentation",
    )

    # Arguments for extracting additional info.
    # We only do addition info extraction, and will not replace the original info.
    parser.add_argument(
        "--enable_only_save_lidar_poses",
        action="store_true",
        help="Extract lidar-to-vehicle pose only. All other processing are skipped. "
        "Will save to waymo/waymo_processed_data_extra/seq_name.pkl",
    )

    args = parser.parse_args()

    if args.func == "create_waymo_infos":
        import yaml
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / "../../../").resolve()
        create_waymo_infos(
            dataset_cfg=dataset_cfg,
            class_names=["Vehicle", "Pedestrian", "Cyclist"],
            data_path=ROOT_DIR / "data" / "waymo",
            save_path=ROOT_DIR / "data" / "waymo",
            raw_data_tag="raw_data",
            processed_data_tag=dataset_cfg.PROCESSED_DATA_TAG,
            workers=args.workers,
            disable_train_info=args.disable_train_info,
            disable_val_info=args.disable_val_info,
            disable_gt_database=args.disable_gt_database,
            enable_only_save_lidar_poses=args.enable_only_save_lidar_poses,
        )
    else:
        raise NotImplementedError(f"function {args.func} not implemented")


if __name__ == "__main__":
    main()
