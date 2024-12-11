import pickle
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import open3d as o3d


@dataclass
class LidarIntrinsics(ABC):
    """
    Abstract class for lidar intrinsics.

    Ref: https://www.mathworks.com/help/lidar/ref/lidarparameters.html
    """

    fov_up: float  # Positive value.
    fov_down: float  # Positive value.
    vertical_res: int
    horizontal_res: int
    max_range: float
    vertical_degrees: List[float] = None


@dataclass
class WaymoLidarIntrinsics(LidarIntrinsics):
    """
    Waymo lidar intrinsics.

    Ref: https://arxiv.org/abs/1912.04838
    """

    fov_up: float = 2.4  # Table 2 in paper.
    fov_down: float = 17.6  # Table 2 in paper.
    vertical_res: int = 64  # Waymo range image. 200 for small lidars.
    horizontal_res: int = 2650  # Waymo range image. 600 for small lidars.
    max_range: float = 75  # Table 2 in paper.
    vertical_degrees: List[float] = None


@dataclass
class NuScenesLidarIntrinsics(LidarIntrinsics):
    """
    NuScenes lidar intrinsics.

    https://velodynelidar.com/wp-content/uploads/2019/12/97-0038-Rev-N-97-0038-DATASHEETWEBHDL32E_Web.pdf
    - 32 Channels
    - Measurement Range: Up to 100 m (70 used in paper)
    - Range Accuracy: Up to ±2 cm (Typical)1
    - Single and Dual Returns (Strongest, Last)
    - Field of View (Vertical): +10.67° to -30.67° (41.33°)
    - Angular Resolution (Vertical): 1.33°
    - Field of View (Horizontal): 360°
    - Angular Resolution (Horizontal/Azimuth): 0.08° - 0.33°
    - Rotation Rate: 5 Hz - 20 Hz
      - 5  Hz: higher resolution
      - 20 Hz: higher frame rate <-> 0.33° <-> 360 / 0.33 = 1090
    - Integrated Web Server for Easy Monitoring and Configuration
    """

    # As vertical_degrees is used, fov_up and fov_down are used to filter
    # rays that are out-of-range.
    fov_up: float = 11.3139  # 10.65520 + (10.65520 - 9.33780) * 0.5
    fov_down: float = 31.36527  # 30.68386 + (30.68386 - 29.32104) * 0.5

    # These are the values from the spec.
    # fov_up: float = 10.67  # Table 2 in paper (10).
    # fov_down: float = 30.67  # Table 2 in paper (-30).

    vertical_res: int = 32  # Table 2 in paper.
    horizontal_res: int = 1090  # From spec.
    max_range: float = 70  # Table 2 in paper.
    vertical_degrees: List[float] = field(
        default_factory=lambda: [
            10.65520,
            9.33780,
            7.97498,
            6.65758,
            5.34018,
            3.97735,
            2.65996,
            1.34256,
            -0.02027,
            -1.33767,
            -2.65507,
            -4.01789,
            -5.33529,
            -6.65269,
            -8.01552,
            -9.33292,
            -10.65032,
            -12.01314,
            -13.33054,
            -14.64794,
            -16.01077,
            -17.32817,
            -18.64557,
            -20.00839,
            -21.32579,
            -22.64319,
            -24.00602,
            -25.32341,
            -26.64081,
            -28.00364,
            -29.32104,
            -30.68386,
        ]
    )

    def __post_init__(self):
        assert len(self.vertical_degrees) == self.vertical_res


@dataclass
class NuScenesVanillaLidarIntrinsics(LidarIntrinsics):
    """
    NuScenes lidar intrinsics without vertical_degrees. In this way, the lidar
    rays are evenly distributed in the vertical direction.
    """

    # https://velodynelidar.com/wp-content/uploads/2019/12/97-0038-Rev-N-97-0038-DATASHEETWEBHDL32E_Web.pdf
    fov_up: float = 10.67
    fov_down: float = 30.67

    # These are the values from the spec.
    # fov_up: float = 10.67  # Table 2 in paper (10).
    # fov_down: float = 30.67  # Table 2 in paper (-30).

    vertical_res: int = 32  # Table 2 in paper.
    horizontal_res: int = 1090  # From spec.
    max_range: float = 70  # Table 2 in paper.
    vertical_degrees: List[float] = None


@dataclass
class KITTILidarIntrinsics(LidarIntrinsics):
    """
    KITTI lidar intrinsics.

    Ref:
    - Vision meets Robotics: The KITTI Dataset
      https://www.cvlibs.net/publications/Geiger2013IJRR.pdf
    - Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite
      https://www.cvlibs.net/publications/Geiger2012CVPR.pdf
    - KITTI Setup
      https://www.cvlibs.net/datasets/kitti/setup.php
    - Lidar Distillation
      https://arxiv.org/abs/2203.14956
    """

    fov_up: float = 3.2  # Lidar Distillation.
    fov_down: float = 23.6  # Lidar Distillation.
    vertical_res: int = 64  # Vision meets Robotics: The KITTI Dataset.
    horizontal_res: int = 1863  # Lidar Distillation.
    max_range: float = 120  # Velodyne HDL-64E's theoretical range.
    vertical_degrees: List[float] = None


@dataclass
class Lidar:
    """
    A Lidar object is parametrized by its intrinsics and pose.
    A Lidar can intersect meshes and return the intersection points.
    """

    intrinsics: LidarIntrinsics = None
    pose: np.ndarray = None

    def __post_init__(self):
        pass

    def get_rays(self) -> np.ndarray:
        """
        Get the lidar rays in world coordinates.

        Returns:
            rays: (N, 6) float32.
                - rays[:, :3]: origin in world coordinates.
                - rays[:, 3:]: direction in world coordinates.
                - Rays are normalized (norm of direction = 1).
        """
        assert isinstance(self.intrinsics, LidarIntrinsics)
        assert isinstance(self.pose, np.ndarray)

        if (
            self.intrinsics.vertical_degrees is None
            or len(self.intrinsics.vertical_degrees) == 0
        ):
            rays_o, rays_d = Lidar._gen_lidar_rays(
                pose=self.pose,
                fov_up=self.intrinsics.fov_up,
                fov_down=self.intrinsics.fov_down,
                H=self.intrinsics.vertical_res,
                W=self.intrinsics.horizontal_res,
            )
        else:
            rays_o, rays_d = Lidar._gen_lidar_rays_with_vertical_degrees(
                pose=self.pose,
                vertical_degrees=self.intrinsics.vertical_degrees,
                W=self.intrinsics.horizontal_res,
            )
        rays = np.concatenate([rays_o, rays_d], axis=-1)
        return rays

    @staticmethod
    def _gen_lidar_rays(pose, fov_up, fov_down, H, W):
        """
        Get lidar rays for a single pose using NumPy, with separate upward and
        downward fields of view. The function generates rays in row-major order,
        meaning that rays are ordered as they would appear in an image, with
        rows being contiguous.

        Args:
            pose: [4, 4] array, camera-to-world transformation matrix.
            fov_up: float, the upward field of view in degrees.
            fov_down: float, the downward field of view in degrees.
            H: int, vertical resolution of the lidar sensor.
            W: int, horizontal resolution of the lidar sensor.

        Returns:
            A tuple of (rays_o, rays_d)
            rays_o: [N, 3] array, the origins of the lidar rays. The ordering of
                    the rays is row-major, i.e., rays_o[0] corresponds to the
                    top-left corner pixel of the range image, and rays_o[N-1] to
                    the bottom-right corner.
            rays_d: [N, 3] array, the directions of the lidar rays, corresponding
                    to the origins in rays_o.
        """
        # Creating a meshgrid for horizontal and vertical indices.
        j, i = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

        # Reshaping indices for ray calculation.
        i = i.reshape([H * W])
        j = j.reshape([H * W])

        # Calculating horizontal and vertical angles for each ray.
        total_fov = fov_up + fov_down
        beta = -(i - W / 2) / W * 2 * np.pi
        alpha = (fov_up - j / H * total_fov) / 180 * np.pi

        # Calculating direction vectors for each ray.
        directions = np.stack(
            [
                np.cos(alpha) * np.cos(beta),
                np.cos(alpha) * np.sin(beta),
                np.sin(alpha),
            ],
            axis=-1,
        )

        # Transforming direction vectors with the pose.
        rays_d = np.dot(directions, pose[:3, :3].T)  # (N, 3)
        rays_o = pose[:3, 3]  # [3]

        # Expanding for each ray.
        rays_o = np.expand_dims(rays_o, axis=0).repeat(len(directions), axis=0)

        return rays_o, rays_d

    @staticmethod
    def _gen_lidar_rays_with_vertical_degrees(pose, vertical_degrees, W):
        """
        Get lidar rays using specific vertical degrees for each ray, using NumPy.
        The function generates rays in row-major order, where rays are ordered as
        they would appear in an image, with rows being contiguous.

        Args:
            pose: [4, 4] array, camera-to-world transformation matrix.
            vertical_degrees: List[float], the vertical angles for each ray in degrees.
            W: int, the horizontal resolution of the lidar sensor.

        Returns:
            A tuple of (rays_o, rays_d)
            rays_o: [N, 3] array, the origins of the lidar rays. The ordering of the
                    rays is row-major, i.e., rays_o[0] corresponds to the top-left
                    corner pixel of the range image, and rays_o[N-1] to the
                    bottom-right corner.
            rays_d: [N, 3] array, the directions of the lidar rays, corresponding
                    to the origins in rays_o.
        """
        H = len(vertical_degrees)

        # Creating a meshgrid for horizontal and vertical indices with 'ij' indexing.
        j, i = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

        # Reshaping indices for ray calculation.
        i = i.reshape([H * W])
        j = j.reshape([H * W])

        # Calculating horizontal and vertical angles for each ray.
        beta = -(i - W / 2) / W * 2 * np.pi
        alpha = np.array(vertical_degrees) / 180 * np.pi

        # Mapping vertical index to its corresponding angle.
        alpha = alpha[j]

        # Calculating direction vectors for each ray.
        directions = np.stack(
            [
                np.cos(alpha) * np.cos(beta),
                np.cos(alpha) * np.sin(beta),
                np.sin(alpha),
            ],
            axis=-1,
        )

        # Transforming direction vectors with the pose.
        rays_d = np.dot(directions, pose[:3, :3].T)  # (N, 3)
        rays_o = pose[:3, 3]  # [3]

        # Expanding for each ray.
        rays_o = np.expand_dims(rays_o, axis=0).repeat(H * W, axis=0)

        return rays_o, rays_d


def main():
    script_dir = Path(__file__).parent.absolute().resolve()
    lit_root = script_dir.parent.parent
    test_data_dir = lit_root / "data" / "test_data"

    raycast_data_path = test_data_dir / "raycast_data.pkl"
    raycast_mesh_path = test_data_dir / "raycast_mesh.ply"

    with open(raycast_data_path, "rb") as f:
        raycast_data = pickle.load(f)
    raycast_mesh = o3d.io.read_triangle_mesh(str(raycast_mesh_path))
    raycast_mesh.compute_vertex_normals()

    # Plot
    points = raycast_data["points"]
    rays = raycast_data["rays"]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    o3d.visualization.draw_geometries([raycast_mesh, pcd])


if __name__ == "__main__":
    main()
