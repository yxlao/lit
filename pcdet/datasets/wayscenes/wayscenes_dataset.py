import numpy as np
from torch.utils.data import Dataset

from pcdet.datasets.dataset import DatasetTemplate
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.datasets.waymo.waymo_dataset import WaymoDataset


class WayScenesDataset(DatasetTemplate):
    def __init__(
        self,
        waymo_cfg,
        nuscenes_cfg,
        training=True,
        root_path=None,
        logger=None,
    ):
        """
        Initializes WayScenes dataset which contains instances of WaymoDataset
        and NuScenesDataset.

        Args:
            waymo_cfg: Configuration for Waymo dataset.
            nuscenes_cfg: Configuration for NuScenes dataset.
            training: Boolean indicating if the dataset is used for training.
            root_path: Root directory path where datasets are stored.
            logger: Logger for logging purposes.
        """
        # Initialize Waymo dataset
        self.waymo_dataset = WaymoDataset(
            dataset_cfg=waymo_cfg,
            class_names=waymo_cfg.CLASS_NAMES,
            training=training,
            root_path=root_path,
            logger=logger,
        )

        # Initialize NuScenes dataset
        self.nuscenes_dataset = NuScenesDataset(
            dataset_cfg=nuscenes_cfg,
            class_names=nuscenes_cfg.CLASS_NAMES,
            training=training,
            root_path=root_path,
            logger=logger,
        )

        # Compute the dataset length based on WAYMO_TO_NUSCENES_RATIO
        self.waymo_sampled_length = int(
            len(self.nuscenes_dataset) * waymo_cfg.WAYMO_TO_NUSCENES_RATIO
        )
        self.total_length = self.waymo_sampled_length + len(self.nuscenes_dataset)
        print("[WayScenes Dataset]")
        print(f"- Total samples per epoch   : {self.total_length}")
        print(f"- Waymo samples per epoch   : {self.waymo_sampled_length}")
        print(f"- NuScenes samples per epoch: {len(self.nuscenes_dataset)}")

        # Manually export some shared attributes. ##############################
        # - Ideally, this shall not be necessary. A Dataset class shall only be
        #   responsible for providing data via __getitem__ and __len__ methods.
        # - However, in OpenPCDet, a Dataset class's attributes are also used
        #   to build the network architecture.
        # - Here, we first check if the shared attributes are the same for both
        #   datasets. If they are, we export them. If not, we pick one of the
        #   values manually.
        ########################################################################
        # class_names
        assert self.waymo_dataset.class_names == self.nuscenes_dataset.class_names
        self.class_names = self.waymo_dataset.class_names

        # point_feature_encoder
        # manually checked POINT_FEATURE_ENCODING in:
        # - tools/cfgs/dataset_configs/da_waymo_dataset.yaml
        # - tools/cfgs/dataset_configs/da_nuscenes_kitti_dataset.yaml
        self.point_feature_encoder = self.waymo_dataset.point_feature_encoder

        # grid_size
        assert np.allclose(
            self.waymo_dataset.grid_size,
            self.nuscenes_dataset.grid_size,
        )
        self.grid_size = self.waymo_dataset.grid_size

        # point_cloud_range
        assert np.allclose(
            self.waymo_dataset.point_cloud_range,
            self.nuscenes_dataset.point_cloud_range,
        )
        self.point_cloud_range = self.waymo_dataset.point_cloud_range

        # voxel_size
        assert np.allclose(
            self.waymo_dataset.voxel_size,
            self.nuscenes_dataset.voxel_size,
        )
        self.voxel_size = self.waymo_dataset.voxel_size

        # dataset_cfg
        # This is really bad, but it is required for SECOND
        self.dataset_cfg = self.waymo_dataset.dataset_cfg
        ########################################################################

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        if index < self.waymo_sampled_length:
            waymo_index = np.random.randint(0, len(self.waymo_dataset))
            # print(f"WayScenes[{index:05d}]: Waymo[{waymo_index:05d}]")
            return self.waymo_dataset[waymo_index]
        else:
            nuscenes_index = index - self.waymo_sampled_length
            # print(f"WayScenes[{index:05d}]: NuScenes[{nuscenes_index:05d}]")
            return self.nuscenes_dataset[nuscenes_index]
