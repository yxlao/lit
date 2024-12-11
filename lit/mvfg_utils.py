import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import DataLoader, Dataset

from pcdet.ops.pointnet2.pointnet2_stack.pointnet2_utils import furthest_point_sample


def fps_sampling(points: np.ndarray, num_fps_samples: int, allow_fewer_points=True):
    """
    Wrapper function for furthest point sampling on NumPy arrays.

    Args:
        points: (N, 3) NumPy array of points.
        num_fps_samples: int, number of points to sample.
        allow_fewer_points: bool, if True, allow fewer points to be sampled if
            the input points have fewer points than num_fps_samples, directly
            returning a clone of the points if there are not enough points to
            sample from. If False, first we augment the points by duplicating
            some of them, and this will guarantee (num_fps_samples, 3) is returned.

    Returns:
        sampled_points: (num_fps_samples, 3) NumPy array of the sampled points
            or the original points if allow_fewer_points is True and points
            have fewer points than num_fps_samples.
    """
    assert isinstance(points, np.ndarray), "points must be a NumPy array"
    assert points.ndim == 2 and points.shape[1] == 3, "points must be (N, 3)"

    N = points.shape[0]
    if N <= num_fps_samples:
        if allow_fewer_points:
            return points.copy()
        else:
            additional_indices = np.random.choice(N, num_fps_samples - N, replace=True)
            points = np.concatenate([points, points[additional_indices]], axis=0)

    points_tensor = torch.from_numpy(points).float().cuda()
    points_tensor = points_tensor.unsqueeze(0).contiguous()  # Add batch dimension

    sampled_indices_tensor = furthest_point_sample(
        points_tensor, num_fps_samples
    ).long()
    sampled_indices = sampled_indices_tensor[0].cpu().numpy()
    sampled_points = points[sampled_indices]

    return sampled_points


class MVFGDataset(Dataset):
    def __init__(self, mvfg_dir):
        self.mvfg_dir = Path(mvfg_dir)
        self.mvfg_files = []
        for scene_dir in self.mvfg_dir.iterdir():
            if scene_dir.is_dir():
                self.mvfg_files.extend(scene_dir.glob("*.pkl"))

    def __len__(self):
        return len(self.mvfg_files)

    def __getitem__(self, idx):
        mvfg_path = self.mvfg_files[idx]
        with open(mvfg_path, "rb") as f:
            fgmv = pickle.load(f)

        # Return as a dictionary
        return {
            "mv_enforced_fps_deepsdf_points": fgmv["mv_enforced_fps_deepsdf_points"],
            "fused_deepsdf_latent": fgmv["fused_deepsdf_latent"],
            "gt_latent": fgmv["gt_latent"],
        }


def collate_fn(batch):
    mv_enforced_fps_deepsdf_points = torch.tensor(
        np.array([item["mv_enforced_fps_deepsdf_points"] for item in batch]),
        dtype=torch.float32,
    )
    fused_deepsdf_latent = torch.stack(
        [
            torch.tensor(item["fused_deepsdf_latent"], dtype=torch.float32)
            for item in batch
        ]
    )
    gt_latent = torch.stack(
        [torch.tensor(item["gt_latent"], dtype=torch.float32) for item in batch]
    )

    return {
        "mv_enforced_fps_deepsdf_points": mv_enforced_fps_deepsdf_points,
        "fused_deepsdf_latent": fused_deepsdf_latent,
        "gt_latent": gt_latent,
    }


def get_mvfg_dataloader(mvfg_dir, batch_size=1, shuffle=True, num_workers=0):
    dataset = MVFGDataset(mvfg_dir)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return data_loader


class SharedPCNEncoderV1(nn.Module):
    # 694,272 trainable params
    def __init__(self):
        super(SharedPCNEncoderV1, self).__init__()
        self.mlp1 = nn.Conv1d(3, 128, 1)
        self.mlp2 = nn.Conv1d(128, 256, 1)
        self.mlp3 = nn.Conv1d(256, 512, 1)
        self.mlp4 = nn.Conv1d(512, 1024, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(1024)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.bn1(self.mlp1(x)))
        x = F.relu(self.bn2(self.mlp2(x)))
        x = F.relu(self.bn3(self.mlp3(x)))
        x = F.relu(self.bn4(self.mlp4(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.tanh(x)
        return x


class SharedPCNEncoderV2(nn.Module):
    # 822,784 trainable params
    def __init__(self):
        super(SharedPCNEncoderV2, self).__init__()
        self.mlp1 = nn.Conv1d(3, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.mlp2 = nn.Conv1d(128, 256, 1)
        self.mlp3 = nn.Conv1d(512, 512, 1)
        self.bn3 = nn.BatchNorm1d(512)
        self.mlp4 = nn.Conv1d(512, 1024, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        x: (B, 3, N)
        """
        # (B, 128, N)
        x = F.relu(self.bn1(self.mlp1(x)))
        # (B, 256, N)
        x = self.mlp2(x)
        # (B, 256, 1)
        x_max_pooled = torch.max(x, 2, keepdim=True)[0]
        # (B, 256, N)
        x_repeated = x_max_pooled.repeat(1, 1, x.size(2))
        # (B, 512, N)
        x = torch.cat((x, x_repeated), dim=1)
        # (B, 512, N)
        x = F.relu(self.bn3(self.mlp3(x)))
        # (B, 1024, N)
        x = self.mlp4(x)  # MLP4 without BN and ReLU
        # (B, 1024)
        x = torch.max(x, 2, keepdim=False)[0]
        # (B, 1024)
        x = self.tanh(x)
        return x


class MVDeepSDFModel(nn.Module):
    def __init__(self, ckpt_path=None):
        super(MVDeepSDFModel, self).__init__()
        self.shared_pc_encoder = SharedPCNEncoderV2()
        self.fc = nn.Linear(1280, 256)

        if ckpt_path:
            self._initialize_from_ckpt(ckpt_path)

    def forward(self, fused_deepsdf_latent, mv_points):
        """
        Args:
            fused_deepsdf_latent: (256,)
            points: (B, 3, N)
        """
        # Extract global features from mv_points
        global_features = self.shared_pc_encoder(mv_points)

        # Concatenate repeated deepsdf_latent with global features
        B = mv_points.size(0)
        fused_deepsdf_latent_repeated = fused_deepsdf_latent.repeat(B, 1)
        concatenated_features = torch.cat(
            (fused_deepsdf_latent_repeated, global_features), dim=1
        )

        # Average pooling across the batch dimension
        pooled_features = torch.mean(concatenated_features, dim=0, keepdim=True)

        # Get predicted latent code (pd_latent)
        pd_latent = self.fc(pooled_features)
        pd_latent = pd_latent.squeeze(0)  # (1, 256) -> (256,)

        return pd_latent

    def _initialize_from_ckpt(self, ckpt_path):
        """
        Initializes model weights from a given checkpoint path.
        """
        try:
            print(f"Loading checkpoint '{ckpt_path}'")
            ckpt = torch.load(ckpt_path, map_location="cpu")

            if "state_dict" in ckpt:
                model_state_dict = self.state_dict()
                loaded_state_dict = ckpt["state_dict"]

                for name, param in loaded_state_dict.items():
                    if name not in model_state_dict:
                        raise KeyError(
                            f"Param '{name}' found in checkpoint but not in model parameters."
                        )
                    if model_state_dict[name].shape != param.shape:
                        raise ValueError(
                            f"Shape mismatch for '{name}': model param "
                            f"{model_state_dict[name].shape}, "
                            f"checkpoint param {param.shape}."
                        )
                    if model_state_dict[name].dtype != param.dtype:
                        raise TypeError(
                            f"Type mismatch for '{name}': model param "
                            f"{model_state_dict[name].dtype}, "
                            f"checkpoint param {param.dtype}."
                        )

                self.load_state_dict(loaded_state_dict)
                print("Loaded model weights from checkpoint")

            else:
                raise KeyError("Checkpoint does not contain 'state_dict' key")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint from '{ckpt_path}'. Error: {e}"
            )
