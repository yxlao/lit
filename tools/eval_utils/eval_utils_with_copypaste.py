import copy
import pickle
import time
from pathlib import Path

import camtools as ct
import numpy as np
import open3d as o3d
import torch
import tqdm

from lit.copy_paste_utils import copy_paste_nuscenes_to_kitti, recompute_voxels
from lit.recon_utils import bboxes_to_lineset
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.models import load_data_to_gpu
from pcdet.models.model_utils.dsnorm import set_ds_target
from pcdet.utils import common_utils


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric["recall_roi_%s" % str(cur_thresh)] += ret_dict.get(
            "roi_%s" % str(cur_thresh), 0
        )
        metric["recall_rcnn_%s" % str(cur_thresh)] += ret_dict.get(
            "rcnn_%s" % str(cur_thresh), 0
        )
    metric["gt_num"] += ret_dict.get("gt", 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict["recall_%s" % str(min_thresh)] = "(%d, %d) / %d" % (
        metric["recall_roi_%s" % str(min_thresh)],
        metric["recall_rcnn_%s" % str(min_thresh)],
        metric["gt_num"],
    )


def eval_one_epoch(
    cfg,
    model,
    dataloader,
    epoch_id,
    logger,
    dist_test=False,
    save_to_file=False,
    result_dir=None,
    copy_paste_with=None,
    args=None,
):
    """
    Args:
        copy_paste_with: Replace foreground points with points from another
            dataset or config. When this option is enabled, the batch_size must
            be 1.
    """
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / "final_result" / "data"
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        "gt_num": 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric["recall_roi_%s" % str(cur_thresh)] = 0
        metric["recall_rcnn_%s" % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    # dataset.kitti_infos = dataset.kitti_infos[:50]
    class_names = dataset.class_names

    logger.info("*************** EPOCH %s EVALUATION *****************" % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], broadcast_buffers=False
        )
    model.eval()

    if cfg.get("SELF_TRAIN", None) and cfg.SELF_TRAIN.get("DSNORM", None):
        model.apply(set_ds_target)

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(
            total=len(dataloader), leave=True, desc="eval", dynamic_ncols=True
        )

    if isinstance(dataset, KittiDataset) and dataloader.batch_size != 1:
        raise ValueError("Only support batch size 1 for customized eval.")

    enable_kitti_stats_and_exit = False
    kitti_stats = []

    det_annos = []
    gt_annos = []
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        if enable_kitti_stats_and_exit:
            points = batch_dict["points"][:, 1:4]
            points = points - dataset.dataset_cfg["SHIFT_COOR"]
            kitti_stats.append({"points": points})
            progress_bar.update()
            continue

        if copy_paste_with is not None:
            if dataloader.batch_size != 1:
                raise ValueError("Only support batch size 1 for copy_paste_with")
            print(f"Using copy_paste_with {copy_paste_with}")
            if copy_paste_with == "kitti_pcd_kitti_lidar_kitti_size":
                # Just to validate recompute_voxels.
                batch_dict = recompute_voxels(dataloader, batch_dict)
            elif copy_paste_with == "kitti_pcd_kitti_lidar_nuscenes_size":
                batch_dict = copy_paste_nuscenes_to_kitti(
                    dataloader=dataloader,
                    batch_dict=batch_dict,
                    src_domain="kitti_pcd",
                    dst_style="kitti",
                    dst_bbox_size="nuscenes",
                )
            elif copy_paste_with == "nuscenes_mesh_nuscenes_lidar_kitti_size":
                batch_dict = copy_paste_nuscenes_to_kitti(
                    dataloader=dataloader,
                    batch_dict=batch_dict,
                    src_domain="nuscenes_mesh",
                    dst_style="nuscenes",
                    dst_bbox_size="kitti",
                )
            elif copy_paste_with == "nuscenes_mesh_nuscenes_lidar_nuscenes_size":
                batch_dict = copy_paste_nuscenes_to_kitti(
                    dataloader=dataloader,
                    batch_dict=batch_dict,
                    src_domain="nuscenes_mesh",
                    dst_style="nuscenes",
                    dst_bbox_size="nuscenes",
                )
            elif copy_paste_with == "nuscenes_mesh_kitti_lidar_kitti_size":
                batch_dict = copy_paste_nuscenes_to_kitti(
                    dataloader=dataloader,
                    batch_dict=batch_dict,
                    src_domain="nuscenes_mesh",
                    dst_style="kitti",
                    dst_bbox_size="kitti",
                )
            elif copy_paste_with == "nuscenes_mesh_kitti_lidar_nuscenes_size":
                batch_dict = copy_paste_nuscenes_to_kitti(
                    dataloader=dataloader,
                    batch_dict=batch_dict,
                    src_domain="nuscenes_mesh",
                    dst_style="kitti",
                    dst_bbox_size="nuscenes",
                )
            else:
                raise ValueError(f"Unknown copy_paste_with {copy_paste_with}")

        # Backup the contents of batch_dict on CPU.
        cpu_batch_dict = dict()
        cpu_batch_dict["gt_boxes"] = copy.deepcopy(batch_dict["gt_boxes"])

        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        visualize = False
        if visualize:
            points = batch_dict["points"][:, 1:4].cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            gt_boxes = batch_dict["gt_boxes"][0].cpu().numpy()
            gt_ls = bboxes_to_lineset(gt_boxes, frame_pose=np.eye(4))
            gt_ls.paint_uniform_color([0, 0, 1])

            pd_boxes = pred_dicts[0]["pred_boxes"].cpu().numpy()
            pd_ls = bboxes_to_lineset(pd_boxes, frame_pose=np.eye(4))
            pd_ls.paint_uniform_color([1, 0, 0])

            if Path("kitti_eval_info.pkl").is_file():
                with open("kitti_eval_info.pkl", "rb") as f:
                    kitti_eval_info = pickle.load(f)
                eval_det_annos = kitti_eval_info["eval_det_annos"]
                eval_gt_annos = kitti_eval_info["eval_gt_annos"]

                assert len(eval_det_annos) == len(eval_gt_annos)

                if len(eval_det_annos) != len(dataloader):
                    print(f"len(eval_det_annos)!= len(dataloader), skip stats")
                else:
                    # Ref: code from KittiDataset.evaluation.
                    eval_det_annos = copy.deepcopy(eval_det_annos)
                    eval_gt_annos = copy.deepcopy(eval_gt_annos)
                    eval_det_annos = [eval_det_annos[i]]
                    eval_gt_annos = [eval_gt_annos[i]]

                    from pcdet.datasets.kitti.kitti_object_eval_python import (
                        eval as kitti_eval,
                    )

                    ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                        eval_gt_annos, eval_det_annos, class_names
                    )
                    print(f"# Frame {i} results ####################")
                    print(ap_result_str)
                    print(f"#######################################")

            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

            o3d.visualization.draw_geometries([pcd, gt_ls, pd_ls, axes])

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict,
            pred_dicts,
            class_names,
            output_path=final_output_dir if save_to_file else None,
        )

        if isinstance(dataset, KittiDataset):
            # Make a fake gt_dict with gt_boxes.
            # We already assume batch_size = 1.
            pred_dict = pred_dicts[0]
            device = pred_dict["pred_boxes"].device
            gt_dict = {}

            gt_boxes = cpu_batch_dict["gt_boxes"][0]
            num_gt_boxes = len(gt_boxes)

            gt_dict["pred_boxes"] = torch.tensor(gt_boxes).to(device)
            gt_dict["pred_scores"] = -torch.ones(num_gt_boxes).to(device)
            gt_dict["pred_labels"] = torch.ones(num_gt_boxes).to(device).long()
            gt_dict["pred_cls_scores"] = torch.zeros(num_gt_boxes).to(device)
            gt_dict["pred_iou_scores"] = torch.zeros(num_gt_boxes).to(device)
            gt_batch_annos = dataset.generate_prediction_dicts(
                batch_dict,
                [gt_dict],
                class_names,
                output_path=None,
                force_no_filter=True,
            )
            gt_annos += gt_batch_annos

        det_annos += annos

        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if enable_kitti_stats_and_exit:
        script_dir = Path(__file__).parent.absolute()
        lit_root = script_dir.parent.parent
        kitti_stats_path = lit_root / "data" / "test_data" / "kitti_stats.pkl"
        with open(kitti_stats_path, "wb") as f:
            pickle.dump(kitti_stats, f)
        print(f"Saved kitti stats to {kitti_stats_path}")
        exit(0)

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(
            det_annos, len(dataset), tmpdir=result_dir / "tmpdir"
        )
        metric = common_utils.merge_results_dist(
            [metric], world_size, tmpdir=result_dir / "tmpdir"
        )

    logger.info("*************** Performance of EPOCH %s *****************" % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info(
        "Generate label finished(sec_per_example: %.4f second)." % sec_per_example
    )

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric["gt_num"]
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric["recall_roi_%s" % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric["recall_rcnn_%s" % str(cur_thresh)] / max(
            gt_num_cnt, 1
        )
        logger.info("recall_roi_%s: %f" % (cur_thresh, cur_roi_recall))
        logger.info("recall_rcnn_%s: %f" % (cur_thresh, cur_rcnn_recall))
        ret_dict["recall/roi_%s" % str(cur_thresh)] = cur_roi_recall
        ret_dict["recall/rcnn_%s" % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno["name"].__len__()
    logger.info(
        "Average predicted number of objects(%d samples): %.3f"
        % (len(det_annos), total_pred_objects / max(1, len(det_annos)))
    )

    with open(result_dir / "result.pkl", "wb") as f:
        pickle.dump(det_annos, f)

    # Original non-modified GT.
    result_str, result_dict = dataset.evaluation(
        det_annos,
        class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir,
    )
    print("Eval with original annos ################################")
    logger.info(result_str)
    print("##########################################################")

    # Copy-pasted GT evaluation.
    if isinstance(dataset, KittiDataset):
        result_str, result_dict = dataset.evaluation_with_custom_annos(
            det_annos,
            gt_annos,
            class_names,
            eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
            output_path=final_output_dir,
        )
        print("Eval with custom gt annos ################################")
        logger.info(result_str)
        print("##########################################################")

    ret_dict.update(result_dict)

    # add avg predicted number of objects to tensorboard log
    ret_dict["eval_avg_pred_bboxes"] = total_pred_objects / max(1, len(det_annos))

    logger.info("Result is save to %s" % result_dir)
    logger.info("****************Evaluation done.*****************")
    return ret_dict


if __name__ == "__main__":
    pass
