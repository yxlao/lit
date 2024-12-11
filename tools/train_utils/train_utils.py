import glob
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import tqdm
from torch.nn.utils import clip_grad_norm_

from lit.recon_utils import bboxes_to_lineset

_script_dir = Path(__file__).resolve().parent
_lit_root = _script_dir.parent.parent
_tools_dir = _lit_root / "tools"
sys.path.append(str(_tools_dir))

from test import eval_single_ckpt

from eval_utils import eval_utils


def train_one_epoch(
    model,
    optimizer,
    train_loader,
    model_func,
    lr_scheduler,
    accumulated_iter,
    optim_cfg,
    rank,
    tbar,
    total_it_each_epoch,
    dataloader_iter,
    tb_log=None,
    leave_pbar=False,
):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(
            total=total_it_each_epoch,
            leave=leave_pbar,
            desc="train",
            dynamic_ncols=True,
        )

    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print("new iters")

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]["lr"]

        if tb_log is not None:
            tb_log.add_scalar("meta_data/learning_rate", cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        visualize = False
        if visualize:
            points = batch["points"][:, 1:4]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            gt_boxes = batch["gt_boxes"][0]
            gt_ls = bboxes_to_lineset(gt_boxes, frame_pose=np.eye(4))
            gt_ls.paint_uniform_color([0, 0, 1])

            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)

            o3d.visualization.draw_geometries([pcd, gt_ls, axes])

        loss, tb_dict, disp_dict = model_func(model, batch)

        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1
        disp_dict.update({"loss": loss.item(), "lr": cur_lr})

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar("train/loss", loss, accumulated_iter)
                tb_log.add_scalar("meta_data/learning_rate", cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar("train/" + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(
    model,
    optimizer,
    train_loader,
    target_loader,
    test_loader,
    model_func,
    lr_scheduler,
    optim_cfg,
    start_epoch,
    total_epochs,
    start_iter,
    rank,
    tb_log,
    ckpt_save_dir,
    ps_label_dir,
    source_sampler=None,
    target_sampler=None,
    lr_warmup_scheduler=None,
    ckpt_save_interval=1,
    max_ckpt_save_num=50,
    merge_all_iters_to_one_epoch=False,
    logger=None,
    ema_model=None,
    cfg=None,
    cmd_args_str=None,  # For logging
):
    accumulated_iter = start_iter
    with tqdm.trange(
        start_epoch,
        total_epochs,
        desc="epochs",
        dynamic_ncols=True,
        leave=(rank == 0),
    ) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, "merge_all_iters_to_one_epoch")
            train_loader.dataset.merge_all_iters_to_one_epoch(
                merge=True, epochs=total_epochs
            )
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if source_sampler is not None:
                source_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch(
                model,
                optimizer,
                train_loader,
                model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter,
                optim_cfg=optim_cfg,
                rank=rank,
                tbar=tbar,
                tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:
                # Save ckpt (unchanged)
                ckpt_list = glob.glob(str(ckpt_save_dir / "checkpoint_epoch_*.pth"))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(
                        0, len(ckpt_list) - max_ckpt_save_num + 1
                    ):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_path = ckpt_save_dir / ("checkpoint_epoch_%d" % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter),
                    filename=ckpt_path,
                )

                # For simplicity, test data loader and eval are always single GPU
                if test_loader is not None and rank == 0:
                    assert cfg is not None, "cfg should be provided for evaluation"

                    model.eval()

                    with tempfile.TemporaryDirectory() as temp_eval_dir:
                        temp_eval_dir = Path(temp_eval_dir)
                        ret_dict, result_str = eval_utils.eval_one_epoch(
                            cfg=cfg,  # cfg.LOCAL_RANK, etc.
                            model=model,
                            dataloader=test_loader,
                            epoch_id=trained_epoch,
                            logger=logger,
                            dist_test=False,
                            result_dir=temp_eval_dir,
                            save_to_file=False,  # We only need the result_str
                            args=None,  # Not used.
                        )

                    ckpt_dir = ckpt_path.parent
                    metric_dir = ckpt_dir.parent / "metric"
                    metric_path = metric_dir / f"{ckpt_path.stem}.txt"
                    metric_dir.mkdir(parents=True, exist_ok=True)
                    metric_path.parent.mkdir(parents=True, exist_ok=True)

                    # Place cmd_args_str on top
                    if cmd_args_str is not None:
                        result_str = cmd_args_str + "\n" + result_str

                    with open(metric_path, "w") as f:
                        f.write(result_str)

                    model.train()


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet

        version = "pcdet+" + pcdet.__version__
    except:
        version = "none"

    return {
        "epoch": epoch,
        "it": it,
        "model_state": model_state,
        "optimizer_state": optim_state,
        "version": version,
    }


def save_checkpoint(state, filename="checkpoint"):
    if False and "optimizer_state" in state:
        optimizer_state = state["optimizer_state"]
        state.pop("optimizer_state", None)
        optimizer_filename = "{}_optim.pth".format(filename)
        torch.save({"optimizer_state": optimizer_state}, optimizer_filename)

    filename = "{}.pth".format(filename)
    torch.save(state, filename)
