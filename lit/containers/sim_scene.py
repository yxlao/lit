from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

from lit.containers.base_container import BaseContainer
from lit.containers.sim_frame import SimFrame


@dataclass
class SimScene(BaseContainer):
    """
    Storing data for simulated scene.
    """

    sim_frames: List[SimFrame] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        if self.sim_frames is None:
            self.sim_frames = []

    def to_dict(self):
        return {
            "sim_frames": [sim_frame.to_dict() for sim_frame in self.sim_frames],
        }

    @classmethod
    def from_dict(cls, dict_data: dict):
        return cls(
            sim_frames=[
                SimFrame.from_dict(sim_frame) for sim_frame in dict_data["sim_frames"]
            ],
        )

    def append_frame(self, sim_frame: SimFrame):
        if not isinstance(sim_frame, SimFrame):
            raise ValueError(f"sim_frame must be SimFrame, got {type(sim_frame)}")
        self.sim_frames.append(sim_frame)

    def save_sim_frames(self, sim_scene_dir: Path):
        """
        Save each frame's points in local coordinates in e.g. 0000.npz.
        We save as .npz rather than .pkl for compatibility with OpenPCDet.

        Example directory structure:

        sim_nuscenes/         # <- sim_dir
        ├── scene_name_0000/  # <- sim_scene_dir
        │   ├── 0000.npz
        │   ├── 0001.npz
        │   ├── ...
        ├── scene_name_0001/  # <- sim_scene_dir
        │   ├── 0000.npz
        │   ├── 0001.npz
        │   ├── ...
        ├── ...
        """
        sim_scene_dir.mkdir(parents=True, exist_ok=True)
        for sim_frame in self.sim_frames:
            npz_path = sim_scene_dir / f"{sim_frame.frame_index:04d}.npz"
            np.savez_compressed(npz_path, **sim_frame.to_dict())

    @classmethod
    def load_sim_frames(cls, sim_scene_dir: Path):
        """
        See the directory structure in SimScene.save_sim_frames().
        """
        sim_scene_dir = Path(sim_scene_dir)
        if not sim_scene_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {sim_scene_dir}")

        sim_frame_paths = sorted(sim_scene_dir.glob("*.npz"), key=lambda x: x.stem)
        if len(sim_frame_paths) == 0:
            raise FileNotFoundError(f"No .npz files found in {sim_scene_dir}")

        sim_frames = []
        for i, sim_frame_path in enumerate(sim_frame_paths):
            npz_data = np.load(sim_frame_path, allow_pickle=True)
            sim_frame = SimFrame.from_dict(npz_data)

            # Sanity check the frame index.
            expected_index = int(sim_frame_path.stem)
            if sim_frame.frame_index != expected_index:
                raise ValueError(
                    f"Frame index mismatch for {sim_frame_path}: "
                    f"expected {expected_index}, got {sim_frame.frame_index}"
                )
            sim_frames.append(sim_frame)

        # Sanity check the frame indices are sorted in ascending order
        frame_indices = [sim_frame.frame_index for sim_frame in sim_frames]
        if not all(
            frame_indices[i] < frame_indices[i + 1]
            for i in range(len(frame_indices) - 1)
        ):
            raise ValueError("Frame indices are not sorted in ascending order.")

        return cls(sim_frames=sim_frames)

    def save(self, path: Path, verbose=False):
        raise RuntimeError("Use SimScene.save_sim_frames() instead.")

    @classmethod
    def load(cls, path: Path):
        raise RuntimeError("Use SimScene.load_sim_frames() instead.")

    def __len__(self):
        return len(self.sim_frames)

    def __getitem__(self, idx):
        return self.sim_frames[idx]

    def get_frame_by_frame_index(self, frame_index: int):
        """
        Get a frame by frame_index, as self.sim_frames may not be sequential.
        """

        for frame in self.sim_frames:
            if frame.frame_index == frame_index:
                return frame
        raise ValueError(f"Cannot find frame with frame_index {frame_index}.")
