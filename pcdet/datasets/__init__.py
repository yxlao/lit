import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler

from pcdet.datasets.dataset import DatasetTemplate
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.datasets.lyft.lyft_dataset import LyftDataset
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.datasets.waymo.waymo_dataset import (
    MixedWaymoDataset,
    WaymoDataset,
    WaymoDatasetInfo,
)
from pcdet.datasets.wayscenes.wayscenes_dataset import WayScenesDataset
from pcdet.utils import common_utils

__all__ = {
    "DatasetTemplate": DatasetTemplate,
    "KittiDataset": KittiDataset,
    "WaymoDataset": WaymoDataset,
    "MixedWaymoDataset": MixedWaymoDataset,
    "WayScenesDataset": WayScenesDataset,
    "WaymoDatasetInfo": WaymoDatasetInfo,
    "NuScenesDataset": NuScenesDataset,
    "LyftDataset": LyftDataset,
}


class DistributedSampler(_DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(
    dataset_cfg,
    class_names,
    batch_size,
    dist,
    root_path=None,
    workers=4,
    logger=None,
    training=True,
    merge_all_iters_to_one_epoch=False,
    total_epochs=0,
    force_no_shuffle=False,
):
    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, "merge_all_iters_to_one_epoch")
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=workers,
        shuffle=(sampler is None) and training and (not force_no_shuffle),
        collate_fn=dataset.collate_batch,
        drop_last=False,
        sampler=sampler,
        timeout=0,
    )

    return dataset, dataloader, sampler


def build_wayscenes_dataloader(
    waymo_cfg,
    nuscenes_cfg,
    batch_size,
    dist,
    root_path=None,
    workers=4,
    logger=None,
    training=True,
    merge_all_iters_to_one_epoch=False,
    total_epochs=0,
    force_no_shuffle=False,
):
    """
    Builds a dataloader for the WayScenes dataset, which contains instances of
    both WaymoDataset and NuScenesDataset.

    Args:
        waymo_cfg: Configuration for Waymo dataset.
        nuscenes_cfg: Configuration for NuScenes dataset.
        batch_size: Batch size for the dataloader.
        dist: Boolean indicating if distributed training is used.
        root_path: Root directory path where datasets are stored.
        workers: Number of workers for the dataloader.
        logger: Logger for logging purposes.
        training: Boolean indicating if the dataset is used for training.
        merge_all_iters_to_one_epoch: If True, all iterations are merged into one epoch.
        total_epochs: Total number of epochs for training.
        force_no_shuffle: If True, disables shuffling of the dataset.

    Returns:
        dataset: The WayScenes dataset object.
        dataloader: The dataloader for the WayScenes dataset.
        sampler: The sampler used for the dataset; None if not in distributed mode.
    """
    from pcdet.datasets import WayScenesDataset

    dataset = WayScenesDataset(
        waymo_cfg=waymo_cfg,
        nuscenes_cfg=nuscenes_cfg,
        training=training,
        root_path=root_path,
        logger=logger,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(
            dataset, "merge_all_iters_to_one_epoch"
        ), "merge_all_iters_to_one_epoch not implemented for WayScenesDataset"
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=workers,
        shuffle=(sampler is None) and training and not force_no_shuffle,
        collate_fn=dataset.collate_batch,
        drop_last=False,
        sampler=sampler,
        timeout=0,
    )

    return dataset, dataloader, sampler
