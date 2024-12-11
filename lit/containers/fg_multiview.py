from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

import camtools as ct
import numpy as np
import open3d as o3d
from pycg import vis


@dataclass
class FGMultiview:
    """
    Container for multi-view foreground data. An instance of this class contains
    multi-view scans of one synthetic vehicle in a real-world trajectory with
    reconstructed background mesh.
    """

    """
    Section 1: Basic GT info
    """
    # ShapeNet object_id, while the "synset_id" (category) is always "02958343".
    shapenet_id: str = None

    # Ground-truth mesh.
    # This is the ground-truth ShapeNet mesh.
    # This mesh is stored in canonical axes.
    gt_mesh: o3d.geometry.TriangleMesh = None

    # Ground-truth latent code.
    # (1, 256)
    # The latent code can be used to produce mesh in ShapeNet axes convention.
    # The neural network shall learn to reconstruct this latent code.
    gt_latent: np.array = None

    """
    Section 2: Points in canonical space (canonical axes and canonical scale)
    """
    # Canonical points of one foreground object.
    # - Scale: canonical scale (same as bbox)
    # - Axes : canonical axes (same as waymo)
    # - Shape: (N, 3)
    fused_canonical_points: np.ndarray = None

    # List of multi-view canonical points.
    # List of (x, 3) of length B, where B is the number of scans.
    # - Scale: canonical scale (same as bbox)
    # - Axes : canonical axes (same as waymo)
    mv_canonical_points: List[np.ndarray] = field(default_factory=list)

    """
    Section 3: Points/latents in DeepSDF space (ShapeNet axes and DeepSDF scale)
    """
    # Fused DeepSDF points.
    # - Scale: DeepSDF scale (normalized with deepsdf_normalization)
    # - Axes : ShapeNet axes (same as DeepSDF
    fused_deepsdf_points: np.ndarray = None

    # Fused DeepSDF latent code (256,), computed from fused_deepsdf_points.
    fused_deepsdf_latent: np.ndarray = None

    # Multi-view DeepSDF points.
    # - Scale: DeepSDF scale (normalized with deepsdf_normalization)
    # - Axes : ShapeNet axes (same as DeepSDF
    mv_deepsdf_points: List[np.ndarray] = field(default_factory=list)

    # FPS-sampled multi-view DeepSDF points.
    # - Scale : DeepSDF scale (normalized with deepsdf_normalization)
    # - Axes  : ShapeNet axes (same as DeepSDF)
    # - Sample: Sampled to at most 256 points
    mv_fps_deepsdf_points: List[np.ndarray] = field(default_factory=list)

    # Enforced FPS-sampled multi-view DeepSDF points.
    # This makes sure allow_fewer_points=False during sampling.
    mv_enforced_fps_deepsdf_points: List[np.ndarray] = field(default_factory=list)

    # Multi-view latent codes from mv_deepsdf_points (non-sampled).
    # B x (256,), where B is number of scans.
    mv_deepsdf_latents: List[np.ndarray] = field(default_factory=list)

    """
    Section 4: Additional states
    """
    # Normalization (offset, scale) for points.
    # The normalization is computed from the fused_canonical_points, and can
    # be applied to single-view or multi-view fused points.
    _deepsdf_normalization_offset_scale: tuple[float] = field(default_factory=tuple)

    def __post_init__(self):
        super().__post_init__()

    def visualize(self):
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)

        # Canonical axes objects.
        gt_mesh = copy.deepcopy(self.gt_mesh)
        gt_mesh.compute_vertex_normals()
        fused_canonical_pcd = o3d.geometry.PointCloud()
        fused_canonical_pcd.points = o3d.utility.Vector3dVector(
            self.fused_canonical_points
        )

        # DeepSDF axes objects.
        fused_deepsdf_pcd = o3d.geometry.PointCloud()
        fused_deepsdf_pcd.points = o3d.utility.Vector3dVector(self.fused_deepsdf_points)

        _ = vis.show_3d(
            [axes, gt_mesh],
            [axes, fused_canonical_pcd],
            [axes, fused_deepsdf_pcd],
            use_new_api=False,
        )

    @staticmethod
    def _compute_deepsdf_normalization(fused_points, buffer=1.03):
        """
        Normalize points for running DeepSDF models, such that:
        1. The center (avg of min max bounds, not centroid) of the object is at
           the origin.
        2. Max distance of a point from the origin is (1 * buffer).

        Normalization does not change axes convention. Both ShapeNet axes and
        Canonical axes points can be normalized with the same normalization.

        DeepSDF uses normalized points in ShapeNet axes convention.
        """
        if not isinstance(fused_points, np.ndarray):
            raise ValueError(f"points must be np.ndarray, got {type(fused_points)}")
        if fused_points.ndim != 2 or fused_points.shape[1] != 3:
            raise ValueError(f"points must be (N, 3), got {fused_points.shape}")
        if fused_points.size == 0:
            raise ValueError("Points array is empty.")

        min_vals = np.min(fused_points, axis=0)
        max_vals = np.max(fused_points, axis=0)
        center = (min_vals + max_vals) / 2.0
        offset = -center

        centered_points = fused_points - center
        max_distance = np.max(np.linalg.norm(centered_points, axis=1))
        max_distance *= buffer
        scale = 1.0 / max_distance

        return offset, scale

    def normalize_by_fused_normalization(self, points):
        if self._deepsdf_normalization_offset_scale is None:
            if self.fused_canonical_points is None:
                raise ValueError(
                    "self.fused_canonical_points is None, "
                    "cannot compute fused normalization"
                )
            self._deepsdf_normalization_offset_scale = (
                ForegroundMultiView._compute_deepsdf_normalization(
                    fused_points=self.fused_canonical_points
                )
            )
        offset, scale = self._deepsdf_normalization_offset_scale
        return (points + offset) * scale

    def denormalize_by_fused_normalization(self, points):
        if self._deepsdf_normalization_offset_scale is None:
            if self.fused_canonical_points is None:
                raise ValueError(
                    "self.fused_canonical_points is None, "
                    "cannot compute fused normalization"
                )
            self._deepsdf_normalization_offset_scale = (
                ForegroundMultiView._compute_deepsdf_normalization(
                    fused_points=self.fused_canonical_points
                )
            )
        offset, scale = self._deepsdf_normalization_offset_scale
        return points / scale - offset

    @staticmethod
    def rotate_axes_canonical_to_shapenet(points):
        rotate_c2s = np.array(
            [
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        return ct.transform.transform_points(points, rotate_c2s)

    @staticmethod
    def rotate_axes_shapenet_to_canonical(points):
        rotate_s2c = np.array(
            [
                [0, 0, -1, 0],
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        return ct.transform.transform_points(points, rotate_s2c)

    def one_step_shapenet_to_canonical(self, points_or_mesh):
        """
        Denormalizes and rotates points or mesh from ShapeNet axes to canonical
        axes in one step, without modifying the original points or mesh.

        Args:
            points_or_mesh: np.ndarray of points (N, 3) or
                open3d.geometry.TriangleMesh.

        Returns:
            Transformed np.ndarray of points (N, 3) or
                a new open3d.geometry.TriangleMesh in canonical axes.
        """
        if isinstance(points_or_mesh, np.ndarray):
            points = points_or_mesh
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"points must be (N, 3), got shape {points.shape}")

            points = self.rotate_axes_shapenet_to_canonical(points)
            points = self.denormalize_by_fused_normalization(points)

            return points

        elif isinstance(points_or_mesh, o3d.geometry.TriangleMesh):
            mesh = copy.deepcopy(points_or_mesh)
            vertices = np.asarray(mesh.vertices)
            vertices = self.rotate_axes_shapenet_to_canonical(vertices)
            vertices = self.denormalize_by_fused_normalization(vertices)
            mesh.vertices = o3d.utility.Vector3dVector(vertices)

            if mesh.has_vertex_normals():
                mesh.compute_vertex_normals()

            return mesh

        else:
            raise TypeError(
                "Input must be either an np.ndarray of points or an TriangleMesh."
            )

    def one_step_canonical_to_shapenet(self, points_or_mesh):
        """
        Normalizes and rotates points or mesh from canonical axes to ShapeNet
        axes in one step, without modifying the original points or mesh.

        Args:
            points_or_mesh: np.ndarray of points (N, 3) or
                open3d.geometry.TriangleMesh.

        Returns:
            Transformed np.ndarray of points (N, 3) or
                a new open3d.geometry.TriangleMesh in ShapeNet axes.
        """
        if isinstance(points_or_mesh, np.ndarray):
            points = points_or_mesh
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"points must be (N, 3), got shape {points.shape}")

            normalized_points = self.normalize_by_fused_normalization(points)
            points = self.rotate_axes_canonical_to_shapenet(normalized_points)

            return points

        elif isinstance(points_or_mesh, o3d.geometry.TriangleMesh):
            mesh = copy.deepcopy(points_or_mesh)
            vertices = np.asarray(mesh.vertices)
            normalized_vertices = self.normalize_by_fused_normalization(vertices)
            vertices = self.rotate_axes_canonical_to_shapenet(normalized_vertices)
            mesh.vertices = o3d.utility.Vector3dVector(vertices)

            if mesh.has_vertex_normals():
                mesh.compute_vertex_normals()

            return mesh

        else:
            raise TypeError(
                "Input must be either an np.ndarray of points or an TriangleMesh."
            )

    def save(self, path: Path):
        path = Path(path)
        if path.suffix != ".pkl":
            raise ValueError(f"path must end with .pkl, got {path}")

        # Convert the dataclass to a dictionary
        data = asdict(self)

        # Process the ground truth mesh for saving
        if self.gt_mesh is not None:
            data["gt_mesh"] = {
                "vertices": np.asarray(self.gt_mesh.vertices, dtype=np.float32),
                "triangles": np.asarray(self.gt_mesh.triangles, dtype=np.int32),
            }

        # Convert numpy arrays to float32 for points and latent codes
        for key in [
            "fused_canonical_points",
            "mv_canonical_points",
            "fused_deepsdf_points",
            "fused_deepsdf_latent",
            "mv_deepsdf_points",
            "mv_fps_deepsdf_points",
            "mv_enforced_fps_deepsdf_points",
            "mv_deepsdf_latents",
        ]:
            if key in data:
                if isinstance(data[key], list):
                    data[key] = [np.array(item, dtype=np.float32) for item in data[key]]
                else:  # For single arrays
                    data[key] = np.array(data[key], dtype=np.float32)

        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path):
        path = Path(path)
        if path.suffix != ".pkl":
            raise ValueError(f"path must end with .pkl, got {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)

        # Reconstruct the ground truth mesh
        if "gt_mesh" in data:
            mesh_data = data.pop("gt_mesh")
            gt_mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(mesh_data["vertices"]),
                triangles=o3d.utility.Vector3iVector(mesh_data["triangles"]),
            )
            data["gt_mesh"] = gt_mesh

        return cls(**data)
