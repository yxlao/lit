import nksr
import open3d as o3d
import torch


class BGReconstructor:
    """
    Background reconstructor with NKSR.
    """

    def __init__(self, voxel_size=None, chunked=True) -> None:
        """
        Args:
            voxel_size: Voxel size for reconstruction. None means 0.1.
            chunked: Whether to use chunked reconstruction.
        """
        self.device = torch.device("cuda:0")
        self.chunk_tmp_device = torch.device("cpu")
        self.reconstructor = nksr.Reconstructor(self.device)

        # Important parameters for NKSR.
        if chunked and voxel_size is not None:
            raise ValueError(
                "Cannot use chunked reconstruction with custom voxel size."
            )
        self.voxel_size = voxel_size
        self.chunk_size = 51.2 if chunked else -1

    def recon(self, points, lidar_centers):
        """
        Reconstruct background points.

        Args:
            points: (N, 3) points.
            lidar_centers: (N, 3) per-point lidar centers.
        """
        points = torch.from_numpy(points).float().to(self.device)
        lidar_centers = torch.from_numpy(lidar_centers).float().to(self.device)

        field = self.reconstructor.reconstruct(
            xyz=points,
            sensor=lidar_centers,
            detail_level=None,
            voxel_size=self.voxel_size,  # If chunk is used, voxel_size is ignored.
            # Minor configs for better efficiency (not necessary)
            approx_kernel_grad=True,
            solver_tol=1e-4,
            fused_mode=True,
            # Chunked reconstruction (if OOM)
            chunk_size=self.chunk_size,
            preprocess_fn=nksr.get_estimate_normal_preprocess_fn(64, 85.0),
        )
        nksr_mesh = field.extract_dual_mesh(mise_iter=1)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(nksr_mesh.v.cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(nksr_mesh.f.cpu().numpy())
        mesh.compute_vertex_normals()

        return mesh
