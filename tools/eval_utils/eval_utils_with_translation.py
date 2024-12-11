import pickle
import time

import numpy as np
import torch
import tqdm
from visual_utils import open3d_vis_utils as V

from lidartranslator.api import nuscenes_to_waymo
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
    args=None,
):
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
    class_names = dataset.class_names
    det_annos = []

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
    start_time = time.time()

    # TODO: only support batch_size=1 for now. This is just for convenience
    # for debugging. We should support batch_size > 1 in the future.
    if dataloader.batch_size != 1:
        raise NotImplementedError("batch_size > 1 is not supported for now.")

    all_num_points = []
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            print('src["points"]          :', batch_dict["points"].shape)
            print('src["frame_id"]        :', batch_dict["frame_id"])
            print('src["metadata"]        :', batch_dict["metadata"])
            print('src["gt_boxes"]        :', batch_dict["gt_boxes"].shape)
            print('src["use_lead_xyz"]    :', batch_dict["use_lead_xyz"])
            print('src["voxels"]          :', batch_dict["voxels"].shape)
            print('src["voxel_coords"]    :', batch_dict["voxel_coords"].shape)
            print('src["voxel_num_points"]:', batch_dict["voxel_num_points"].shape)
            print('src["batch_size"]      :', batch_dict["batch_size"])

            if args.translator == "nksr" or args.translator == "pointersect":
                src_points = batch_dict["points"][:, 1:].cpu().numpy()
                data_dict = nuscenes_to_waymo(
                    src_points,
                    translator=args.translator,
                    pointersect_k=args.pointersect_k,
                    visualize=args.visualize,
                )
                dst_points = data_dict["points"]

                if args.recompute_voxels:
                    print("Recompute voxels!!!!!!!!!!!!!!!!")
                    # 1. convert all tensors to numpy arrays
                    for key in batch_dict.keys():
                        if isinstance(batch_dict[key], torch.Tensor):
                            batch_dict[key] = batch_dict[key].cpu().numpy()

                    # 2. strip the batch dimension, strip unused keys
                    batch_dict["points"] = batch_dict["points"][:, 1:]
                    if key in ["voxels", "voxel_coords", "voxel_num_points"]:
                        batch_dict.pop(key)

                    # 3. update batch_dict
                    batch_dict["points"] = dst_points

                    # 4. run preprocessing
                    batch_dict = dataloader.dataset.point_feature_encoder.forward(
                        batch_dict
                    )
                    batch_dict = dataloader.dataset.data_processor.forward(batch_dict)

                    # 5. convert all numpy arrays to torch tensors
                    for key in batch_dict.keys():
                        if isinstance(batch_dict[key], np.ndarray) and np.issubdtype(
                            batch_dict[key].dtype, np.number
                        ):
                            batch_dict[key] = torch.tensor(batch_dict[key]).cuda()

                    # 6. add batch dimension back
                    points = batch_dict["points"].cpu().numpy()
                    points = np.concatenate(
                        (
                            np.zeros((points.shape[0], 1), dtype=np.float32),
                            points,
                        ),
                        axis=1,
                    )
                    batch_dict["points"] = torch.tensor(points).cuda()

                    voxel_coords = batch_dict["voxel_coords"].cpu().numpy()
                    voxel_coords = np.concatenate(
                        (
                            np.zeros((voxel_coords.shape[0], 1), dtype=np.int32),
                            voxel_coords,
                        ),
                        axis=1,
                    )
                    batch_dict["voxel_coords"] = torch.tensor(voxel_coords).cuda()

                else:
                    dst_points = np.concatenate(
                        (
                            np.zeros((dst_points.shape[0], 1), dtype=np.float32),
                            dst_points,
                        ),
                        axis=1,
                    )
                    batch_dict["points"] = torch.tensor(dst_points).cuda()

            else:
                if args.recompute_voxels:
                    print("Recompute voxels!!!!!!!!!!!!!!!!")
                    # 1. convert all tensors to numpy arrays
                    for key in batch_dict.keys():
                        if isinstance(batch_dict[key], torch.Tensor):
                            batch_dict[key] = batch_dict[key].cpu().numpy()

                    # 2. strip the batch dimension, strip unused keys
                    batch_dict["points"] = batch_dict["points"][:, 1:]
                    for key in ["voxels", "voxel_coords", "voxel_num_points"]:
                        batch_dict.pop(key)

                    # 3. update batch_dict
                    # batch_dict["points"] = dst_points

                    # 4. run preprocessing
                    batch_dict = dataloader.dataset.point_feature_encoder.forward(
                        batch_dict
                    )
                    batch_dict = dataloader.dataset.data_processor.forward(batch_dict)

                    # 5. convert all numpy arrays to torch tensors
                    for key in batch_dict.keys():
                        if isinstance(batch_dict[key], np.ndarray) and np.issubdtype(
                            batch_dict[key].dtype, np.number
                        ):
                            batch_dict[key] = torch.tensor(batch_dict[key]).cuda()

                    # 6. add batch dimension back
                    points = batch_dict["points"].cpu().numpy()
                    points = np.concatenate(
                        (
                            np.zeros((points.shape[0], 1), dtype=np.float32),
                            points,
                        ),
                        axis=1,
                    )
                    batch_dict["points"] = torch.tensor(points).cuda()

                    voxel_coords = batch_dict["voxel_coords"].cpu().numpy()
                    voxel_coords = np.concatenate(
                        (
                            np.zeros((voxel_coords.shape[0], 1), dtype=np.int32),
                            voxel_coords,
                        ),
                        axis=1,
                    )
                    batch_dict["voxel_coords"] = torch.tensor(voxel_coords).cuda()

            all_num_points.append(len(batch_dict["points"]))

            # batch_dict
            # ['points', 'frame_id', 'metadata', 'gt_boxes', 'use_lead_xyz', 'voxels', 'voxel_coords', 'voxel_num_points', 'batch_size']
            print('dst["points"]          :', batch_dict["points"].shape)
            print('dst["frame_id"]        :', batch_dict["frame_id"])
            print('dst["metadata"]        :', batch_dict["metadata"])
            print('dst["gt_boxes"]        :', batch_dict["gt_boxes"].shape)
            print('dst["use_lead_xyz"]    :', batch_dict["use_lead_xyz"])
            print('dst["voxels"]          :', batch_dict["voxels"].shape)
            print('dst["voxel_coords"]    :', batch_dict["voxel_coords"].shape)
            print('dst["voxel_num_points"]:', batch_dict["voxel_num_points"].shape)
            print('dst["batch_size"]      :', batch_dict["batch_size"])

            pred_dicts, ret_dict = model(batch_dict)

        # Visualize
        if args.visualize:
            if args.translator == "nksr":
                # Plot after translation, with rays
                # V.draw_scenes(
                #     points=batch_dict["points"][:, 1:],
                #     gt_boxes=batch_dict["gt_boxes"][0],
                #     ref_boxes=pred_dicts[0]["pred_boxes"],
                #     ref_labels=pred_dicts[0]["pred_labels"],
                #     ref_scores=pred_dicts[0]["pred_scores"],
                #     src_points=src_points,
                #     vertices=data_dict["vertices"],
                #     triangles=data_dict["triangles"],
                #     rays_o=data_dict["rays_o"],
                #     rays_d=data_dict["rays_d"],
                #     rays_length=10,
                #     rays_subsample=0.01,
                # )

                # Plot after translation
                V.draw_scenes(
                    points=batch_dict["points"][:, 1:],
                    gt_boxes=batch_dict["gt_boxes"][0],
                    ref_boxes=pred_dicts[0]["pred_boxes"],
                    ref_labels=pred_dicts[0]["pred_labels"],
                    ref_scores=pred_dicts[0]["pred_scores"],
                    src_points=None,
                    vertices=data_dict["vertices"],
                    triangles=data_dict["triangles"],
                    rays_o=None,
                    rays_d=None,
                    rays_length=10,
                    rays_subsample=0.01,
                )

                # Plot before translation
                V.draw_scenes(
                    points=src_points,
                    gt_boxes=None,
                    ref_boxes=None,
                    ref_labels=None,
                    ref_scores=None,
                )

            elif args.translator == "pointerset":
                # Plot after translation
                V.draw_scenes(
                    points=batch_dict["points"][:, 1:],
                    gt_boxes=batch_dict["gt_boxes"][0],
                    ref_boxes=pred_dicts[0]["pred_boxes"],
                    ref_labels=pred_dicts[0]["pred_labels"],
                    ref_scores=pred_dicts[0]["pred_scores"],
                    src_points=None,
                    rays_o=None,
                    rays_d=None,
                    rays_length=10,
                    rays_subsample=0.01,
                )

            else:
                V.draw_scenes(
                    points=batch_dict["points"][:, 1:],
                    gt_boxes=batch_dict["gt_boxes"][0],
                    ref_boxes=pred_dicts[0]["pred_boxes"],
                    ref_labels=pred_dicts[0]["pred_labels"],
                    ref_scores=pred_dicts[0]["pred_scores"],
                )

        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict,
            pred_dicts,
            class_names,
            output_path=final_output_dir if save_to_file else None,
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

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

    result_str, result_dict = dataset.evaluation(
        det_annos,
        class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir,
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    # add avg predicted number of objects to tensorboard log
    ret_dict["eval_avg_pred_bboxes"] = total_pred_objects / max(1, len(det_annos))

    logger.info("Result is save to %s" % result_dir)
    logger.info("****************Evaluation done.*****************")

    # avg num points per frame
    avg_num_points = np.mean(all_num_points)
    logger.info(f"Average number of points per frame: {avg_num_points:.03f}")
    ret_dict["avg_num_points"] = avg_num_points

    return ret_dict


if __name__ == "__main__":
    pass
