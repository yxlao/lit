import copy
from pathlib import Path
from types import SimpleNamespace
from typing import List

import camtools as ct
import nksr
import numpy as np
import open3d as o3d
import torch
from pycg import vis

from lit.containers.fg_object import FGObject
from lit.extern.deepsdf.complete import DeepSDFEngine
from lit.mvfg_utils import MVDeepSDFModel, fps_sampling
from lit.recon_utils import bbox_to_corners, bbox_to_lineset, largest_cluster_mesh


def _get_deepsdf_root():
    import lit.extern.deepsdf

    if lit.extern.deepsdf.__path__:
        deepsdf_root = Path(lit.extern.deepsdf.__path__[0])
        return deepsdf_root
    else:
        raise RuntimeError("No directory path found for the lit.extern.deepsdf package")


class NKSRReconstructor:
    """
    Reconstruct mesh given point cloud or point cloud with normals.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nksr_reconstructor = nksr.Reconstructor(self.device)

    def reconstruct(self, points, normals=None):
        """
        Reconstruct mesh given point cloud or point cloud with normals.

        Args:
            points: (N, 3), point cloud.
            normals: (N, 3), normals of the point cloud.

        Returns:
            mesh: open3d.geometry.TriangleMesh, reconstructed mesh.
        """
        # Make a copy.
        points = np.copy(points)

        # Normalize points to unit cube.
        centroid = np.mean(points, axis=0)
        points -= centroid
        scale = np.max(np.linalg.norm(points, axis=1))
        points /= scale

        if normals is None:
            # Estimate normals with Open3D.
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.estimate_normals()
            normals = np.asarray(pcd.normals)

        # Convert to torch.
        points = torch.from_numpy(points).float().to(self.device)
        normals = torch.from_numpy(normals).float().to(self.device)

        # Reconstruct.
        field = self.nksr_reconstructor.reconstruct(
            xyz=points,
            normal=normals,
            voxel_size=0.05,
        )
        nskr_mesh = field.extract_dual_mesh(mise_iter=2)
        vertices = nskr_mesh.v.cpu().numpy()
        triangles = nskr_mesh.f.cpu().numpy()

        # Denormalize.
        vertices *= scale
        vertices += centroid

        # Convert to open3d.
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()

        return mesh


class BallPivotingReconstructor:
    """
    Reconstruct mesh given point cloud or point cloud with ball pivoting.
    """

    def __init__(self, radii=[0.05, 0.1, 0.2]):
        self.radii = radii

    def reconstruct(self, points):
        # Make a copy.
        points = np.copy(points)

        # Normalize points to unit cube.
        centroid = np.mean(points, axis=0)
        points -= centroid
        scale = np.max(np.linalg.norm(points, axis=1))
        points /= scale

        # Convert to open3d.
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals()

        # Reconstruct.
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector(self.radii),
        )

        # Denormalize.
        vertices = np.asarray(mesh.vertices)
        vertices *= scale
        vertices += centroid
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.compute_vertex_normals()

        return mesh


class AlphaShapeReconstructor:
    """
    Reconstruct mesh given point cloud or point cloud with alpha shape.
    """

    def __init__(self, alpha=0.18):
        self.alpha = alpha

    def reconstruct(self, points):
        # Make a copy.
        points = np.copy(points)

        # Normalize points to unit cube.
        centroid = np.mean(points, axis=0)
        points -= centroid
        scale = np.max(np.linalg.norm(points, axis=1))
        points /= scale

        # Convert to open3d.
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals()

        # Reconstruct.
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd,
            self.alpha,
        )

        # Denormalize.
        vertices = np.asarray(mesh.vertices)
        vertices *= scale
        vertices += centroid
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.compute_vertex_normals()

        return mesh


class PointCloudCompleter:
    """
    Point Cloud Completer class with hard-coded config and model loading for
    KITTI point cloud completion.

    - config: cfgs/PCN_models/PoinTr.yaml
    - model: ckpts/KITTI.pth
    """

    def __init__(self) -> None:
        # Hard-coded config and model paths.
        self.config_path = _pointr_root / "cfgs/PCN_models/PoinTr.yaml"
        self.ckpt_path = _pointr_root / "ckpts/KITTI.pth"
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        if not self.ckpt_path.is_file():
            raise FileNotFoundError(f"Model file not found at {self.ckpt_path}")

        # Load config and model.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = cfg_from_yaml_file(self.config_path)
        self.model = builder.model_builder(self.config.model)
        builder.load_model(self.model, self.ckpt_path)
        self.model.to(self.device)
        self.model.eval()

        # Prepare transformations.
        self.transforms = Compose(
            [
                {
                    "callback": "NormalizeObjectPose",
                    "parameters": {
                        "input_keys": {
                            "ptcloud": "points",
                            "bbox": "bbox_corners",
                        }
                    },
                    "objects": [
                        "points",
                        "bbox_corners",
                    ],
                },
                {
                    "callback": "RandomSamplePoints",
                    "parameters": {"n_points": 2048},
                    "objects": ["points"],
                },
                {
                    "callback": "ToTensor",
                    "objects": [
                        "points",
                        "bbox_corners",
                    ],
                },
            ]
        )
        self.inverse_transforms = Compose(
            [
                {
                    "callback": "InverseNormalizeObjectPose",
                    "parameters": {
                        "input_keys": {
                            "ptcloud": "points",
                            "bbox": "bbox_corners",
                        }
                    },
                    "objects": [
                        "points",
                        "bbox_corners",
                    ],
                },
                {
                    "callback": "ToTensor",
                    "objects": [
                        "points",
                        "bbox_corners",
                    ],
                },
            ]
        )

    def complete(self, src_points, src_bbox_corners):
        """
        Complete a point cloud.

        Normalization is critical. See:
        - datasets/KITTIDataset.py
        - datasets/data_transforms.py

        Args:
            src_points: (N, 3), point cloud to complete.
            bbox_corners: (8, 3), corners of a bounding box defined in PCN format.

        Returns:
            dst_points: (N, 3), completed point cloud.
        """
        # Normalize.
        src_data = {
            "points": src_points.copy(),
            "bbox_corners": src_bbox_corners.copy(),
        }
        src_data = self.transforms(src_data)
        src_points = src_data["points"]

        # Inference
        ret = self.model(src_points.unsqueeze(0).to(self.device))
        dst_points = ret[-1].squeeze(0).detach().cpu().numpy()

        # Inverse normalize.
        dst_data = {
            "points": dst_points.copy(),
            "bbox_corners": src_bbox_corners.copy(),
        }
        dst_data = self.inverse_transforms(dst_data)
        dst_points = dst_data["points"]

        # Remove statistical outliers.
        dst_pcd = o3d.geometry.PointCloud()
        dst_pcd.points = o3d.utility.Vector3dVector(dst_points)
        dst_pcd, _ = dst_pcd.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0,
        )
        dst_points = np.asarray(dst_pcd.points)

        return dst_points


class FGReconstructor:
    """
    Foreground reconstructors.
    """

    def __init__(
        self,
        use_deepsdf: bool = True,
        use_mvdeepsdf: bool = False,
        deepsdf_params: dict = None,
    ) -> None:
        self.device = torch.device("cuda:0")
        # self.completer = PointCloudCompleter()
        self.nksr_reconstructor = NKSRReconstructor()
        self.ball_reconstructor = BallPivotingReconstructor()
        self.alph_reconstructor = AlphaShapeReconstructor(alpha=0.20)
        # self.sap_reconstructor = SAPReconstructor(max_bbox_extend=0.95)

        # DeepSDF engine.
        dp = SimpleNamespace()
        dp.iters = 200
        dp.lr = 1e-4
        dp.num_samples = 8000
        dp.voxel_resolution = 128
        dp.max_batch = 32**3
        dp.l2reg = True
        dp.clamp_dist = 0.1
        if deepsdf_params is not None:
            for k, v in deepsdf_params.items():
                setattr(dp, k, v)
        print(f"FGReconstructor: deepsdf_params={dp}")

        deepsdf_root = _get_deepsdf_root()
        self.deepsdf_engine = DeepSDFEngine(
            specs_path=deepsdf_root / "packaged" / "specs.json",
            ckpt_path=deepsdf_root / "packaged" / "ckpt.pth",
            mean_latent_path=deepsdf_root / "packaged" / "mean_latent.pth",
            iters=dp.iters,
            lr=dp.lr,
            num_samples=dp.num_samples,
            voxel_resolution=dp.voxel_resolution,
            max_batch=dp.max_batch,
            l2reg=dp.l2reg,
            clamp_dist=dp.clamp_dist,
        )

        # v1: use_deepsdf = False.
        # v2: use_deepsdf = True.
        # v3: use_deepsdf = True.
        self.use_deepsdf = use_deepsdf
        print(f"FGReconstructor: use_deepsdf={self.use_deepsdf}")

        if use_mvdeepsdf and not use_deepsdf:
            raise ValueError("use_mvdeepsdf requires use_deepsdf to be True.")
        self.use_mvdeepsdf = use_mvdeepsdf
        if self.use_mvdeepsdf:
            self.mvdeepsdf_model = MVDeepSDFModel(
                ckpt_path=_lit_root / "tools/mvdeepsdf_log/default/ckpts/0050.pth"
            )
            self.mvdeepsdf_model.eval()
            self.mvdeepsdf_model.to(self.device)

    @staticmethod
    def resize_mesh_to_fit_bbox(mesh, axis_aligned_centered_corners):
        """
        Resize mesh to fit tightly within bbox_corners.

        Args:
            mesh: open3d.geometry.TriangleMesh
            axis_aligned_centered_corners: (8, 3), corners of bounding boxes.
                This is used to normalize the points for point cloud completion.
                The corners are assumed to be:
                    - axis-aligned
                    - centered around the origin

        Returns:
            open3d.geometry.TriangleMesh
        """
        vertices = np.asarray(mesh.vertices)

        # Calculate the size of the bounding box and the mesh.
        bbox_min_bound = axis_aligned_centered_corners.min(axis=0)
        bbox_max_bound = axis_aligned_centered_corners.max(axis=0)
        bbox_extents = bbox_max_bound - bbox_min_bound
        mesh_min_bound = vertices.min(axis=0)
        mesh_max_bound = vertices.max(axis=0)
        mesh_extents = mesh_max_bound - mesh_min_bound

        # Move the mesh to the origin, scale it, and move it to the correct position.
        scale_factors = bbox_extents / mesh_extents
        vertices = (vertices - mesh_min_bound) * scale_factors + bbox_min_bound
        np.testing.assert_allclose(vertices.min(axis=0), bbox_min_bound)
        np.testing.assert_allclose(vertices.max(axis=0), bbox_max_bound)

        # Create a new mesh.
        scaled_mesh = o3d.geometry.TriangleMesh()
        scaled_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        scaled_mesh.triangles = mesh.triangles

        return scaled_mesh

    @staticmethod
    def _rotate_canonical_to_shapenet(points: np.ndarray):
        """
        Rotate points from our (Waymo) convention to ShapeNet convention.
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Input points array must have shape (N, 3)")

        R = np.array(
            [
                [0, -1, 0],
                [0, 0, 1],
                [-1, 0, 0],
            ]
        )
        transform = np.eye(4)
        transform[:3, :3] = R
        return ct.transform.transform_points(points, transform)

    @staticmethod
    def _rotate_shapenet_to_canonical(points: np.ndarray):
        """
        Rotate points from ShapeNet convention back to our (Waymo) convention.
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Input points array must have shape (N, 3)")

        R = np.array(
            [
                [0, 0, -1],
                [-1, 0, 0],
                [0, 1, 0],
            ]
        )
        transform = np.eye(4)
        transform[:3, :3] = R
        return ct.transform.transform_points(points, transform)

    def recon_deepsdf(
        self,
        mv_canonical_points: List[np.ndarray],
        fused_canonical_points: np.ndarray,
        axis_aligned_centered_corners: np.ndarray,
    ):
        """
        Reconstruct fused foreground points, with DeepSDF-based method.
        Visualize with an additional heading line.

        Args:
            points: (N, 3), foreground points.
            axis_aligned_centered_corners: (8, 3), corners of bounding boxes.
                This is used to normalize the points for point cloud completion.
                The corners are assumed to be:
                    - axis-aligned
                    - centered around the origin
        """
        if fused_canonical_points.ndim != 2 or fused_canonical_points.shape[1] != 3:
            raise ValueError("Input points array must have shape (N, 3)")
        if axis_aligned_centered_corners.shape != (8, 3):
            raise ValueError("Input corners array must have shape (8, 3)")

        # Remove statistical outliers.
        fused_canonical_pcd = o3d.geometry.PointCloud()
        fused_canonical_pcd.points = o3d.utility.Vector3dVector(fused_canonical_points)
        fused_canonical_pcd, _ = fused_canonical_pcd.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=1.0,
        )
        if len(fused_canonical_pcd.points) > 0:
            fused_canonical_points = np.asarray(fused_canonical_pcd.points)

        # Rotate waymo->shapenet.
        mesh = self.deepsdf_engine.canonical_points_to_mesh(
            fused_canonical_points, do_normalize=True
        )
        mesh.compute_vertex_normals()

        if self.use_mvdeepsdf:
            # Prepare inputs for self.mvdeepsdf_model.
            fgmv = ForegroundMultiView(
                mv_canonical_points=mv_canonical_points,
                fused_canonical_points=fused_canonical_points,
            )
            mv_deepsdf_points = []
            mv_enforced_fps_deepsdf_points = []
            for canonical_points in fgmv.mv_canonical_points:
                deepsdf_points = fgmv.one_step_canonical_to_shapenet(canonical_points)
                mv_deepsdf_points.append(deepsdf_points)
                enforced_fps_deepsdf_points = fps_sampling(
                    points=deepsdf_points,
                    num_fps_samples=256,
                    allow_fewer_points=False,
                )
                mv_enforced_fps_deepsdf_points.append(enforced_fps_deepsdf_points)
            mv_enforced_fps_deepsdf_points = np.array(
                mv_enforced_fps_deepsdf_points, dtype=np.float32
            )
            # (B, 256, 3) -> (B, 3, 256)
            mv_points = mv_enforced_fps_deepsdf_points.transpose(0, 2, 1)
            fused_deepsdf_points = np.concatenate(mv_deepsdf_points, axis=0)
            fused_deepsdf_latent = self.deepsdf_engine._optimize_latent(
                fused_deepsdf_points,
                verbose=False,
            )
            fused_deepsdf_latent = fused_deepsdf_latent.detach().cpu().numpy().flatten()
            mv_points = torch.from_numpy(mv_points).float().to(self.device)
            fused_deepsdf_latent = (
                torch.from_numpy(fused_deepsdf_latent).float().to(self.device)
            )

            # Run MV-DeepSDF inference.
            pd_latent = self.mvdeepsdf_model(
                fused_deepsdf_latent=fused_deepsdf_latent,
                mv_points=mv_points,
            )
            pd_latent = pd_latent.detach().cpu().numpy().flatten()

            # Recon mesh with pd_latent
            mesh_mv = self.deepsdf_engine.np_latent_to_mesh(pd_latent)
            mesh_mv = fgmv.one_step_shapenet_to_canonical(mesh_mv)

            # Visualize
            visualize_mv_recon = True
            if visualize_mv_recon:
                mesh.compute_vertex_normals()
                mesh_mv.compute_vertex_normals()
                axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
                _ = vis.show_3d(
                    [mesh, axes],
                    [mesh_mv, axes],
                )

            # Replace mesh with mv mesh.
            mesh = mesh_mv

        # Rescale to fit bbox.
        mesh = FGReconstructor.resize_mesh_to_fit_bbox(
            mesh,
            axis_aligned_centered_corners=axis_aligned_centered_corners,
        )

        visualize_deepsdf = False
        if visualize_deepsdf:
            # Centered pointcloud.
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(fused_canonical_points)

            # Centered corners lineset.
            lines = np.array(
                [
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 0],
                    [4, 5],
                    [5, 6],
                    [6, 7],
                    [7, 4],
                    [0, 4],
                    [1, 5],
                    [2, 6],
                    [3, 7],
                ]
            )
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(axis_aligned_centered_corners)
            ls.lines = o3d.utility.Vector2iVector(lines)
            ls.colors = o3d.utility.Vector3dVector([[0, 0, 0]] * len(lines))

            # Heading direction lineset.
            top_center = np.mean(axis_aligned_centered_corners[4:8], axis=0)
            top_front_center = (
                axis_aligned_centered_corners[4] + axis_aligned_centered_corners[5]
            ) / 2
            heading_ls = o3d.geometry.LineSet()
            heading_ls.points = o3d.utility.Vector3dVector(
                [top_center, top_front_center]
            )
            heading_ls.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
            heading_ls.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
            ls += heading_ls

            mesh.compute_vertex_normals()
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
            o3d.visualization.draw_geometries(
                [pcd, ls, axes, mesh], window_name="Recon Inputs"
            )

        return mesh

    def recon(
        self,
        points: np.ndarray,
        axis_aligned_centered_corners: np.ndarray,
    ):
        """
        Reconstruct foreground points.

        Args:
            points: (N, 3), foreground points.
            axis_aligned_centered_corners: (8, 3), corners of bounding boxes.
                This is used to normalize the points for point cloud completion.
                The corners are assumed to be:
                    - axis-aligned
                    - centered around the origin

        Return:
            o3d.geometry.TriangleMesh
        """
        # Complete with PoinTr.
        points_completed = self.completer.complete(
            src_points=points,
            src_bbox_corners=axis_aligned_centered_corners,
        )

        # Reconstruct with AlphaShape and sample points from surface.
        alpha_mesh = self.alph_reconstructor.reconstruct(points_completed)
        alpha_mesh = largest_cluster_mesh(alpha_mesh)
        alpha_mesh_pcd = alpha_mesh.sample_points_poisson_disk(5000)

        # Reconstruct with ShapeAsPoints.
        alpha_mesh_points = np.asarray(alpha_mesh_pcd.points)
        sap_mesh = self.sap_reconstructor.reconstruct(alpha_mesh_points)

        # Scale to fit bbox.
        mesh = FGReconstructor.resize_mesh_to_fit_bbox(
            sap_mesh,
            axis_aligned_centered_corners=axis_aligned_centered_corners,
        )

        visualize_fg_recon_steps = False
        if visualize_fg_recon_steps:
            # Clone.
            alpha_mesh_clone = copy.deepcopy(alpha_mesh)
            alpha_mesh_pcd_clone = copy.deepcopy(alpha_mesh_pcd)
            sap_mesh_clone = copy.deepcopy(sap_mesh)
            mesh_clone = copy.deepcopy(mesh)

            # Shift up.
            shift_up = np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 4], [0, 0, 0, 1]]
            )
            alpha_mesh_clone.transform(shift_up).transform(shift_up).transform(shift_up)
            alpha_mesh_pcd_clone.transform(shift_up).transform(shift_up)
            sap_mesh_clone.transform(shift_up)

            # BBox as lineset.
            lines = np.array(
                [
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 0],
                    [4, 5],
                    [5, 6],
                    [6, 7],
                    [7, 4],
                    [0, 4],
                    [1, 5],
                    [2, 6],
                    [3, 7],
                ]
            )
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(axis_aligned_centered_corners)
            ls.lines = o3d.utility.Vector2iVector(lines)

            # Visualize.
            alpha_mesh_clone.compute_vertex_normals()
            sap_mesh_clone.compute_vertex_normals()
            mesh_clone.compute_vertex_normals()
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            o3d.visualization.draw_geometries(
                [
                    axes,
                    alpha_mesh_clone,
                    alpha_mesh_pcd_clone,
                    sap_mesh_clone,
                    mesh_clone,
                    ls,
                ]
            )

        return mesh

    def recon_fg_object(self, fg_object: FGObject):
        """
        Reconstruct an FGObject.

        Args:
            fg_object: The fg_object to reconstruct.

        Returns:
            o3d.geometry.TriangleMesh
        """
        # Collect centered points for one foreground object.
        mv_canonical_points = []  # List of (X, 3)
        for fg_box in fg_object.fg_boxes:
            pseudo_pose = fg_box.compute_local_pseudo_pose()
            pseudo_T = ct.convert.pose_to_T(pseudo_pose)
            canonical_points = ct.transform.transform_points(
                fg_box.local_points, pseudo_T
            )
            mv_canonical_points.append(canonical_points)
        # (N, 3)
        fused_canonical_points = np.concatenate(mv_canonical_points, axis=0)

        # Collect centered corners.
        # - `fused_centered_corners` is a list that contains the corners of the
        #   bounding boxes of each FGBox in the FGObject.
        # - These corners are transformed to be centered around the origin using
        #   the pseudo pose of each FGBox.
        # - The purpose of `fused_centered_corners` is to compute the average
        #   corners of all the bounding boxes in the FGObject.
        # - One reason for using "corners" rather than bboxes directly is that
        #   bboxes cannot be freely transformed. Bboxes has limited degrees of
        #   freedom. But, if the bboxes are all very similar, then it might be
        #   feasible to use averaged bboxes directly (todo in the future).
        fused_centered_corners = []
        for fg_box in fg_object.fg_boxes:
            pseudo_pose = fg_box.compute_local_pseudo_pose()
            pseudo_T = ct.convert.pose_to_T(pseudo_pose)
            local_corners = bbox_to_corners(fg_box.local_bbox)
            centered_corners = ct.transform.transform_points(local_corners, pseudo_T)
            fused_centered_corners.append(centered_corners)

        # (M, 8, 3)
        fused_centered_corners = np.array(fused_centered_corners)
        avg_centered_corners = np.mean(fused_centered_corners, axis=0)

        # Reconstruct v1 or v4.
        if self.use_deepsdf:
            mesh = self.recon_deepsdf(
                mv_canonical_points=mv_canonical_points,
                fused_canonical_points=fused_canonical_points,
                axis_aligned_centered_corners=avg_centered_corners,
            )
        else:
            mesh = self.recon(
                points=fused_canonical_points,
                axis_aligned_centered_corners=avg_centered_corners,
            )

        # Visualize centered ls.
        visualize_ls = False
        if visualize_ls:
            # Centered pointcloud.
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(fused_canonical_points)

            # Centered lineset.
            center_ls = o3d.geometry.LineSet()
            num_ls = 0
            for fg_box in fg_object.fg_boxes:
                ls_bbox = bbox_to_lineset(fg_box.local_bbox)
                ls_bbox.transform(np.linalg.inv(fg_box.compute_local_pseudo_pose()))
                center_ls += ls_bbox
                num_ls += 1

            print(f"Visualizing {num_ls} linesets.")
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
            o3d.visualization.draw_geometries([pcd, center_ls, axes])

        # Visualize original ls (in corresponding frames).
        visualize_original_ls = False
        if visualize_original_ls:
            sample_every_n = 4
            num_samples = 5
            selected_fg_boxes = fg_object.fg_boxes[::sample_every_n][:num_samples]

            original_ls = o3d.geometry.LineSet()
            num_ls = 0
            for fg_box in selected_fg_boxes:
                ls_bbox = bbox_to_lineset(fg_box.local_bbox)
                ls_bbox.transform(fg_box.frame_pose)
                original_ls += ls_bbox
                num_ls += 1

            # fg_box.local_points transformed to world coord with frame_pose
            original_pcd = o3d.geometry.PointCloud()
            for fg_box in selected_fg_boxes:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(fg_box.local_points)
                pcd.transform(fg_box.frame_pose)
                original_pcd += pcd

            print(f"Visualizing {num_ls} linesets.")
            o3d.visualization.draw_geometries([original_ls, original_pcd])

        # Visualize mesh.
        visualize_mesh = False
        if visualize_mesh:
            world_ls = o3d.geometry.LineSet()
            world_mesh = o3d.geometry.TriangleMesh()
            for fg_box in fg_object.fg_boxes:
                pseudo_pose = fg_box.compute_local_pseudo_pose()
                # Ls. Local (frame) coord -> world coord.
                frame_ls = bbox_to_lineset(fg_box.local_bbox)
                frame_ls.transform(fg_box.frame_pose)
                world_ls += frame_ls

                # Mesh. Centered coord -> local (frame) coord -> world cord.
                frame_mesh = copy.deepcopy(mesh)
                frame_mesh.transform(fg_box.frame_pose @ pseudo_pose)
                frame_mesh.compute_vertex_normals()
                world_mesh += frame_mesh

            o3d.visualization.draw_geometries([world_mesh, world_ls])

        return mesh
