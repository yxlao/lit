import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import camtools as ct
import numpy as np
import open3d as o3d
import skimage.measure
import torch
import trimesh
from pycg import vis
from torch import nn, optim

import lit.extern.deepsdf.deep_sdf as deep_sdf

_script_dir = Path(__file__).resolve().absolute().parent
_deepsdf_root = _script_dir


def load_sv2_point_cloud(synset_id: str, object_id: str, data_root: Path) -> np.ndarray:
    surface_samples_path = (
        data_root / "SurfaceSamples" / "ShapeNetV2" / synset_id / f"{object_id}.ply"
    )
    mesh = o3d.io.read_triangle_mesh(str(surface_samples_path))
    points = np.asarray(mesh.vertices)
    return points


def _load_mean_latent(train_latent_dir):
    train_latent_dir = Path(train_latent_dir)
    with torch.no_grad():
        train_latent_paths = sorted(list(train_latent_dir.glob("*.pth")))
        train_latents = [
            torch.load(str(p)).reshape((1, 256)).cpu().numpy()
            for p in train_latent_paths
        ]
        train_latents = np.concatenate(train_latents, axis=0)
        mean_latent = np.mean(train_latents, axis=0, keepdims=True)
        mean_latent = torch.tensor(mean_latent, dtype=torch.float32).cuda()
    return mean_latent


class DeepSDFEngine:
    def __init__(
        self,
        specs_path,
        ckpt_path,
        mean_latent_path,
        iters=500,
        lr=1e-4,
        num_samples=2048,
        voxel_resolution=256,
        max_batch=32**3,
        l2reg=True,
        clamp_dist=0.1,
        verbose=False,
    ):
        self.specs_path = Path(specs_path)
        self.ckpt_path = Path(ckpt_path)
        self.mean_latent_path = Path(mean_latent_path)
        self.iters = iters
        self.lr = lr
        self.num_samples = num_samples
        self.voxel_resolution = voxel_resolution
        self.max_batch = max_batch
        self.l2reg = l2reg
        self.clamp_dist = clamp_dist
        self.verbose = verbose

        self.decoder = self._load_network()
        self.mean_latent = torch.load(self.mean_latent_path)

    @staticmethod
    def _compute_normalization(world_points, buffer=1.03):
        """
        Normalize points, such that:
        1. The center (avg of min max bounds, not centroid) of the object is at
           the origin.
        2. Max distance of a point from the origin is (1 * buffer).

        Normalization does not change axes convention. Both ShapeNet axes and
        Canonical axes points can be normalized.
        """
        if len(world_points) == 0:
            raise ValueError("Points array is empty.")

        min_vals = np.min(world_points, axis=0)
        max_vals = np.max(world_points, axis=0)
        center = (min_vals + max_vals) / 2.0
        offset = -center

        centered_points = world_points - center
        max_distance = np.max(np.linalg.norm(centered_points, axis=1))
        max_distance *= buffer
        scale = 1.0 / max_distance

        # Handles nan scale (e.g. when len(world_points) == 1)
        if not np.isfinite(scale):
            scale = 1.0

        return offset, scale

    @staticmethod
    def _normalize_points(world_points, offset, scale):
        return (world_points + offset) * scale

    @staticmethod
    def _denormalize_points(deepsdf_points, offset, scale):
        return deepsdf_points / scale - offset

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

    def _load_network(self):
        specs = json.load(open(self.specs_path))
        arch = __import__(
            "lit.extern.deepsdf.networks." + specs["NetworkArch"], fromlist=["Decoder"]
        )
        latent_size = specs["CodeLength"]
        decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()
        decoder = nn.DataParallel(decoder)

        checkpoint = torch.load(self.ckpt_path)
        decoder.load_state_dict(checkpoint["model_state_dict"])
        decoder = decoder.module
        decoder.eval()

        return decoder

    def _optimize_latent(self, points):
        """
        Optimizes the latent code to fit the given points. The points are assumed
        to be normalized and in ShapeNet axes.

        Return:
            (1, 256) latent code.
        """
        points = torch.tensor(points, dtype=torch.float32, device="cuda")
        latent = torch.clone(self.mean_latent).cuda().requires_grad_(True)

        optimizer = optim.Adam([latent], lr=self.lr)
        loss_l1 = nn.L1Loss()

        actual_num_samples = min(len(points), self.num_samples)

        last_log_time = time.time()
        for iteration in range(self.iters):
            optimizer.zero_grad()

            indices = torch.randperm(len(points))[:actual_num_samples]
            sampled_points = points[indices]
            latent_inputs = latent.expand(actual_num_samples, -1)

            inputs = torch.cat([latent_inputs, sampled_points], dim=1)
            pred_sdf = self.decoder(inputs)
            pred_sdf = torch.clamp(pred_sdf, -self.clamp_dist, self.clamp_dist)
            sdf_target = torch.zeros_like(pred_sdf)
            loss = loss_l1(pred_sdf, sdf_target)
            if self.l2reg:
                loss += 1e-4 * torch.mean(latent.pow(2))
            loss.backward()
            optimizer.step()

            if self.verbose and iteration % 100 == 0:
                current_time = time.time()
                elapsed_time = current_time - last_log_time
                last_log_time = current_time
                print(
                    f"[Iter {iteration}] "
                    f"Loss: {loss.item():.04f}, "
                    f"Elapsed: {elapsed_time:.04f} sec"
                )

        return latent

    def _optimize_mv_latents(self, mv_points):
        """
        Optimizes multiple latent codes to fit given sets of multi-view points
        independently by minimizing the L1 loss between the predicted SDF values
        and the target SDF values. The optimization is performed in a batched
        manner using the forward_batched function of the decoder.

        Each point cloud in `mv_points` is sampled or duplicated to have
        exactly `self.num_samples` points within each iteration to ensure diversity
        in the sampled points over iterations.

        Args:
            mv_points (List[np.ndarray]): A list of numpy arrays, each of
                shape (N_i, 3), where N_i is the number of points in the i-th
                point cloud, and each point is a 3D coordinate (x, y, z). The
                points are assumed to be normalized and aligned with the
                ShapeNet coordinate axes.

        Returns:
            torch.Tensor: The optimized latent vectors of shape (B, L), where L
                is the latent size, referred to as mv_latents.
        """
        B = len(mv_points)
        device = "cuda"

        # Convert mv_points to a list of tensors on GPU
        mv_points_tensors = [
            torch.tensor(points, dtype=torch.float32, device=device)
            for points in mv_points
        ]

        # Replicate mean_latent for each item in the batch
        mv_latents = self.mean_latent.repeat(B, 1).requires_grad_(True)

        optimizer = optim.Adam([mv_latents], lr=self.lr)
        loss_l1 = nn.L1Loss(reduction="none")

        last_log_time = time.time()
        for iteration in range(self.iters):
            optimizer.zero_grad()

            sampled_points_tensors = []
            for points in mv_points_tensors:
                if len(points) < self.num_samples:
                    # Sample with replacement if there are not enough points
                    indices = torch.randint(
                        len(points), (self.num_samples,), device=device
                    )
                else:
                    # Sample without replacement if there are enough points
                    indices = torch.randperm(len(points), device=device)[
                        : self.num_samples
                    ]
                sampled_points = points[indices]
                sampled_points_tensors.append(sampled_points)

            # Stack all sampled points into a single tensor
            mv_points_tensor = torch.stack(sampled_points_tensors, dim=0)

            latent_inputs = mv_latents.unsqueeze(1).expand(-1, self.num_samples, -1)
            inputs = torch.cat([latent_inputs, mv_points_tensor], dim=2).view(
                -1, self.mean_latent.size(1) + 3
            )

            pred_sdf = self.decoder(inputs).view(B, self.num_samples, -1)
            pred_sdf = torch.clamp(pred_sdf, -self.clamp_dist, self.clamp_dist)

            sdf_target = torch.zeros_like(pred_sdf)
            loss = loss_l1(pred_sdf, sdf_target).mean(dim=2)
            total_loss = loss.mean()
            individual_losses = loss.mean(dim=1)

            total_loss.backward()
            optimizer.step()

            if self.verbose and iteration % 100 == 0:
                current_time = time.time()
                elapsed_time = current_time - last_log_time
                last_log_time = current_time
                individual_loss_str = ", ".join([f"{l:.4f}" for l in individual_losses])
                print(
                    f"[Iter {iteration}] Avg Loss: {total_loss:.4f}, Individual Losses: {individual_loss_str}, Elapsed: {elapsed_time:.2f} sec"
                )

        return mv_latents

    def points_to_mesh(
        self,
        points: np.ndarray,
        do_normalize: bool,
        fallback_to_mean_latent: bool = True,
    ) -> o3d.geometry.TriangleMesh:
        """
        Converts ShapeNet axes points to a ShapeNet axes mesh.

        Args:
            points: (N, 3) array of points in ShapeNet axes.
            do_normalize: bool, if True, normalize the points before processing.

        Returns:
            mesh: Open3D mesh object in ShapeNet axes.
        """
        # Keep original points for visualization
        original_points = points.copy()

        # Normalize
        if do_normalize:
            offset, scale = self._compute_normalization(points)
            points = self._normalize_points(points, offset, scale)

        # Mesh to latent
        if len(points) < 10:
            print(f"[WARNING] Falling back to mean latent with {len(points)} points.")
            latent = self.mean_latent

            # Conclusion: because of the small number of points, the
            # optimization doesn't really optimize the latent code much, this
            # makes the pred_mesh look like the mean_mesh. We just simply use
            # the mean_latent to avoid numerical instability.
            vis_mean_latent = False
            if vis_mean_latent:
                pred_latent = self._optimize_latent(points)
                pred_mesh = self.latent_to_mesh(pred_latent)
                pred_mesh.compute_vertex_normals()
                pred_mesh.paint_uniform_color([1, 0, 0])
                mean_latent = self.mean_latent
                mean_mesh = self.latent_to_mesh(mean_latent)
                mean_mesh.compute_vertex_normals()
                mean_mesh.paint_uniform_color([0, 0, 1])
                axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
                vis.show_3d(
                    [pred_mesh, axes],
                    [mean_mesh, axes],
                )
        else:
            latent = self._optimize_latent(points)

        # Latent to mesh
        try:
            mesh = self.latent_to_mesh(latent)
        except:
            if fallback_to_mean_latent:
                print("[WARNING] Falling back to mean latent as latent_to_mesh fails")
                latent = self.mean_latent
                mesh = self.latent_to_mesh(latent)
            else:
                raise

        # Denormalize
        if do_normalize:
            # Denormalize mesh if normalization was performed
            mesh.vertices = o3d.utility.Vector3dVector(
                self._denormalize_points(np.asarray(mesh.vertices), offset, scale)
            )

        visualize = False
        if visualize:
            # Original PCD is blue
            original_pcd = o3d.geometry.PointCloud()
            original_pcd.points = o3d.utility.Vector3dVector(original_points)
            original_pcd.paint_uniform_color([0, 0, 1])

            # Maybe-normalized PCD is red
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color([1, 0, 0])

            # Mesh is always in the original scale
            mesh.compute_vertex_normals()
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            geometries = [original_pcd, pcd, mesh, axes]

            o3d.visualization.draw_geometries(geometries)

        return mesh

    def mv_points_to_mesh(self, mv_points, do_normalize=True):
        """
        Processes multiple sets of points and generates the corresponding meshes.

        Args:
            mv_points_list (List[np.ndarray]): A list of arrays where each array
                represents a set of points in a point cloud.
            do_normalize (bool): If True, normalizes each point cloud before processing.

        Returns:
            List[o3d.geometry.TriangleMesh]: A list of Open3D mesh objects.
        """
        normalization_params = []
        if do_normalize:
            normalized_points = []
            for points in mv_points:
                offset, scale = self._compute_normalization(points)
                normalized_points.append(self._normalize_points(points, offset, scale))
                normalization_params.append((offset, scale))
            mv_points = normalized_points

        mv_latents = self._optimize_mv_latents(mv_points)
        meshes = [self.latent_to_mesh(latent.unsqueeze(0)) for latent in mv_latents]

        if do_normalize:
            for mesh, (offset, scale) in zip(meshes, normalization_params):
                mesh.vertices = o3d.utility.Vector3dVector(
                    self._denormalize_points(np.asarray(mesh.vertices), offset, scale)
                )

        return meshes

    def canonical_points_to_mesh(
        self,
        points: np.ndarray,
        do_normalize: bool,
    ) -> o3d.geometry.TriangleMesh:
        """
        Canonical axes points to canonical axes mesh.

        Args:
            points: (N, 3) array of points in canonical axes.
            do_normalize: bool, if True, normalize the points before processing.

        Returns:
            mesh: Open3D mesh object in canonical axes.
        """
        # Rotate canonical -> shapenet
        points = DeepSDFEngine._rotate_canonical_to_shapenet(points)

        # Recon
        mesh = self.points_to_mesh(points, do_normalize=do_normalize)

        # Rotate shapenet -> canonical
        mesh.vertices = o3d.utility.Vector3dVector(
            DeepSDFEngine._rotate_shapenet_to_canonical(np.asarray(mesh.vertices))
        )
        return mesh

    def np_latent_to_mesh(self, latent):
        """
        latent: (256,) numpy float32 array.
        """
        latent_vec = torch.tensor(latent, dtype=torch.float32).cuda().reshape(1, 256)
        return self.latent_to_mesh(latent_vec)

    def latent_to_mesh(self, latent_vec):
        """
        ShapeNet axes latent code to ShapeNet axes mesh.

        latent_vec: (1, 256)? torch tensor in CUDA.
        """
        self.decoder.eval()
        voxel_origin = np.array([-1, -1, -1], dtype=np.float32)
        voxel_size = 2.0 / (self.voxel_resolution - 1)
        grid = np.meshgrid(
            np.linspace(voxel_origin[0], voxel_origin[0] + 2, self.voxel_resolution),
            np.linspace(voxel_origin[1], voxel_origin[1] + 2, self.voxel_resolution),
            np.linspace(voxel_origin[2], voxel_origin[2] + 2, self.voxel_resolution),
            indexing="ij",
        )
        grid = np.stack(grid, axis=-1).reshape(-1, 3)
        grid_torch = torch.from_numpy(grid).float().cuda()

        sdf_values = torch.zeros(grid_torch.shape[0], 1).cuda()
        num_samples = grid_torch.shape[0]
        head = 0
        while head < num_samples:
            sample_subset = grid_torch[
                head : min(head + self.max_batch, num_samples), :
            ]
            latent_inputs = latent_vec.expand(sample_subset.size(0), -1)
            inputs = torch.cat([latent_inputs, sample_subset], dim=1)
            sdf_values[head : min(head + self.max_batch, num_samples)] = self.decoder(
                inputs
            ).detach()
            head += self.max_batch

        sdf_shape = (
            self.voxel_resolution,
            self.voxel_resolution,
            self.voxel_resolution,
        )
        sdf_values_np = sdf_values.cpu().numpy().reshape(sdf_shape)
        verts, faces, normals, _ = skimage.measure.marching_cubes(
            sdf_values_np, level=0.0, spacing=[voxel_size] * 3
        )

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts + voxel_origin)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        return mesh


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct a shape from a partial point cloud using DeepSDF."
    )
    parser.add_argument(
        "--specs_path",
        "-s",
        default="examples/cars/specs.json",
        help="Path to the experiment specifications.",
    )
    parser.add_argument(
        "--ckpt_path",
        "-c",
        default="examples/cars/ModelParameters/2000.pth",
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--train_latent_dir",
        "-t",
        default="examples/cars/Reconstructions/2000/Codes/ShapeNetV2/02958343",
        help="Directory containing the training set latent code .pth files.",
    )
    parser.add_argument(
        "--resolution",
        "-r",
        dest="voxel_resolution",
        default=256,
        type=int,
        help="Voxel resolution (vr) in a cube of size 2, voxel_size = 2.0 / (vr - 1)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=500,
        help="Number of iterations for latent code optimization.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for optimizer."
    )
    deep_sdf.add_common_args(parser)
    args = parser.parse_args()
    deep_sdf.configure_logging(args)

    # Package specs, ckpt, mean_latent to _deepsdf_root/packaged
    packaged_dir = _deepsdf_root / "packaged"
    specs_path = packaged_dir / "specs.json"
    ckpt_path = packaged_dir / "ckpt.pth"
    mean_latent_path = packaged_dir / "mean_latent.pth"

    shutil.copy(args.specs_path, specs_path)
    shutil.copy(args.ckpt_path, ckpt_path)
    mean_latent = _load_mean_latent(args.train_latent_dir)
    torch.save(mean_latent, mean_latent_path)

    print(f"[packaged] {specs_path}")
    print(f"[packaged] {ckpt_path}")
    print(f"[packaged] {mean_latent_path}")

    # Initialize reconstructor from packaged files
    reconstructor = DeepSDFEngine(
        specs_path=specs_path,
        ckpt_path=ckpt_path,
        mean_latent_path=mean_latent_path,
        iters=args.iters,
        lr=args.lr,
        num_samples=2048,
        voxel_resolution=args.voxel_resolution,
        max_batch=32**3,
        l2reg=True,
        clamp_dist=0.1,
        verbose=True,
    )

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

    print("\n# Processing points_00")
    points_00 = load_sv2_point_cloud(
        synset_id="02958343",
        object_id="ebc59fa7d366b486122181f48ecf7852",
        data_root=Path("data"),
    )
    mesh_00 = reconstructor.points_to_mesh(points_00, do_normalize=True)
    mesh_00.compute_vertex_normals()
    pcd_00 = o3d.geometry.PointCloud()
    pcd_00.points = o3d.utility.Vector3dVector(points_00)

    print("\n# Processing points_01")
    points_01 = load_sv2_point_cloud(
        synset_id="02958343",
        object_id="fd7741b7927726bda37f3fc191551700",
        data_root=Path("data"),
    )
    mesh_01 = reconstructor.points_to_mesh(points_01, do_normalize=True)
    mesh_01.compute_vertex_normals()
    pcd_01 = o3d.geometry.PointCloud()
    pcd_01.points = o3d.utility.Vector3dVector(points_01)

    print("\n# Processing points_02")
    points_02 = load_sv2_point_cloud(
        synset_id="02958343",
        object_id="fe3dc721f5026196d61b6a34f3fd808c",
        data_root=Path("data"),
    )
    mesh_02 = reconstructor.points_to_mesh(points_02, do_normalize=True)
    mesh_02.compute_vertex_normals()
    pcd_02 = o3d.geometry.PointCloud()
    pcd_02.points = o3d.utility.Vector3dVector(points_02)

    print("\n# Processing 3 mv_points")
    mv_points = [points_00, points_01, points_02]
    meshes = reconstructor.mv_points_to_mesh(mv_points, do_normalize=True)
    for mesh in meshes:
        mesh.compute_vertex_normals()
    vis.show_3d(
        [mesh_00, pcd_00, axes],
        [meshes[0], pcd_00, axes],
        [mesh_01, pcd_01, axes],
        [meshes[1], pcd_01, axes],
        [mesh_02, pcd_02, axes],
        [meshes[2], pcd_02, axes],
    )

    print("\n# Processing 10 mv_points")
    mv_points = [points_00] * 50
    meshes = reconstructor.mv_points_to_mesh(mv_points, do_normalize=True)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    main()
