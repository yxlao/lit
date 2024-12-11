from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import List, Union

_script_dir = Path(__file__).parent.resolve().absolute()


@dataclass
class LitPaths:
    data_version: str = None
    data_domain: str = None
    scene_names: List[str] = field(default_factory=list)
    lit_data_root: Path = None
    scene_dir: Path = None
    fg_dir: Path = None
    bg_dir: Path = None
    sim_waymo_dir: Path = None
    sim_nuscenes_dir: Path = None
    sim_kitti_dir: Path = None

    # Private and static fields.
    _lit_root = _script_dir.parent
    _waymo_lit_data_root = _lit_root / "lit_data" / "waymo"
    _nuscenes_lit_data_root = _lit_root / "lit_data" / "nuscenes"
    _lit_split_dir = _lit_root / "lit_split"

    def __repr__(self):
        return (
            f"LitPaths(\n"
            f"    data_version={self.data_version}, \n"
            f"    data_domain={self.data_domain}, \n"
            f"    scene_names=[List of {len(self.scene_names)} scene names], \n"
            f"    lit_data_root={self.lit_data_root}, \n"
            f"    scene_dir={self.scene_dir}, \n"
            f"    fg_dir={self.fg_dir}, \n"
            f"    bg_dir={self.bg_dir}, \n"
            f"    sim_waymo_dir={self.sim_waymo_dir}, \n"
            f"    sim_nuscenes_dir={self.sim_nuscenes_dir}, \n"
            f"    sim_kitti_dir={self.sim_kitti_dir},\n"
            f")"
        )

    @staticmethod
    def _load_scene_names(lit_split_path: Path) -> List[str]:
        with open(lit_split_path, "r") as f:
            scene_names = f.read().splitlines()
        return scene_names

    @classmethod
    def from_relative_paths(
        cls,
        data_version: str,
        data_domain: str,
        scene_list_path_rel: Union[Path, str],
        scene_dir_rel: Union[Path, str],
        fg_dir_rel: Union[Path, str],
        bg_dir_rel: Union[Path, str],
        sim_waymo_dir_rel: Union[Path, str],
        sim_nuscenes_dir_rel: Union[Path, str],
        sim_kitti_dir_rel: Union[Path, str],
    ):
        if data_domain == "waymo":
            lit_data_root = LitPaths._waymo_lit_data_root
        elif data_domain == "nuscenes":
            lit_data_root = LitPaths._nuscenes_lit_data_root
        else:
            raise ValueError(f"Unknown data_domain: {data_domain}")

        # Load scene names.
        scene_list_path = LitPaths._lit_split_dir / scene_list_path_rel
        scene_names = LitPaths._load_scene_names(scene_list_path)

        # Construct LitPaths.
        lit_paths = cls(
            data_version=data_version,
            data_domain=data_domain,
            scene_names=scene_names,
            lit_data_root=lit_data_root,
            scene_dir=lit_data_root / scene_dir_rel,
            fg_dir=lit_data_root / fg_dir_rel,
            bg_dir=lit_data_root / bg_dir_rel,
            sim_waymo_dir=lit_data_root / sim_waymo_dir_rel,
            sim_nuscenes_dir=lit_data_root / sim_nuscenes_dir_rel,
            sim_kitti_dir=lit_data_root / sim_kitti_dir_rel,
        )

        return lit_paths


_lit_paths_versions = {
    # fmt: off

    # v0: full waymo/nuscenes scenes, with default reconstruction
    # - # Waymo scenes   : 1000
    # - # NuScenes scenes:  840
    "v0": {
        "waymo": LitPaths.from_relative_paths(
            data_version         = "v0",
            data_domain          = "waymo",
            scene_list_path_rel  = "waymo_scene_list_v0.txt",
            scene_dir_rel        = "scene",
            fg_dir_rel           = "fg_v0",
            bg_dir_rel           = "bg_v0",
            sim_waymo_dir_rel    = "sim_waymo_v0",
            sim_nuscenes_dir_rel = "sim_nuscenes_v0",
            sim_kitti_dir_rel    = "sim_kitti_v0",
        ),
        "nuscenes": LitPaths.from_relative_paths(
            data_version         = "v0",
            data_domain          = "nuscenes",
            scene_list_path_rel  = "nuscenes_scene_list_v0.txt",
            scene_dir_rel        = "scene",
            fg_dir_rel           = "fg_v0",
            bg_dir_rel           = "bg_v0",
            sim_waymo_dir_rel    = "sim_waymo_v0",
            sim_nuscenes_dir_rel = "sim_nuscenes_v0",
            sim_kitti_dir_rel    = "sim_kitti_v0",
        ),
    },

    # v1: a subset of scenes
    # - # Waymo scenes   :  350
    # - # NuScenes scenes:  350
    "v1": {
        "waymo": LitPaths.from_relative_paths(
            data_version         = "v1",
            data_domain          = "waymo",
            scene_list_path_rel  = "waymo_scene_list_v1.txt",
            scene_dir_rel        = "scene",
            fg_dir_rel           = "fg_v1",
            bg_dir_rel           = "bg_v1",
            sim_waymo_dir_rel    = "sim_waymo_v1",
            sim_nuscenes_dir_rel = "sim_nuscenes_v1",
            sim_kitti_dir_rel    = "sim_kitti_v1",
        ),
        "nuscenes": LitPaths.from_relative_paths(
            data_version         = "v1",
            data_domain          = "nuscenes",
            scene_list_path_rel  = "nuscenes_scene_list_v1.txt",
            scene_dir_rel        = "scene",
            fg_dir_rel           = "fg_v1",
            bg_dir_rel           = "bg_v1",
            sim_waymo_dir_rel    = "sim_waymo_v1",
            sim_nuscenes_dir_rel = "sim_nuscenes_v1",
            sim_kitti_dir_rel    = "sim_kitti_v1",
        ),
    },
    # fmt: on
}


def get_lit_paths(data_version: str, data_domain: str) -> SimpleNamespace:
    """
    Return the lit_paths for a given data_version and data_domain.
    """
    if (
        data_version not in _lit_paths_versions
        or data_domain not in _lit_paths_versions[data_version]
    ):
        raise ValueError(
            f"data_version={data_version}, "
            f"data_domain={data_domain} is not supported."
        )
    lit_paths = _lit_paths_versions[data_version][data_domain]
    print(f"Loaded lit_paths:\n{lit_paths}")

    return lit_paths
