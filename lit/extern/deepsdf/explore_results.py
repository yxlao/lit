import argparse
import copy
import json
import shutil
from pathlib import Path

import camtools as ct
import numpy as np
import open3d as o3d
import skimage.measure
import torch
import trimesh
from pycg import vis
from torch import nn, optim
from tqdm import tqdm

import lit.extern.deepsdf.deep_sdf

_script_dir = Path(__file__).resolve().absolute().parent
_deepsdf_root = _script_dir


import sys

import ipdb

sys.excepthook = lambda et, ev, tb: (
    None
    if issubclass(et, (KeyboardInterrupt, SystemExit))
    else (print(f"Unhandled exception: {ev}"), ipdb.post_mortem(tb))
)


class DeepSDFMesher:
    """
    Minimal DeepSDF Latent->Mesh converter.
    """

    def __init__(
        self,
        specs_path,
        ckpt_path,
        voxel_resolution=128,
        max_batch=32**3,
    ):
        self.specs_path = Path(specs_path)
        self.ckpt_path = Path(ckpt_path)
        self.voxel_resolution = voxel_resolution
        self.max_batch = max_batch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.decoder = self._load_network()

    def _load_network(self):
        specs = json.load(open(self.specs_path))
        arch = __import__(
            "lit.extern.deepsdf.networks." + specs["NetworkArch"], fromlist=["Decoder"]
        )
        latent_size = specs["CodeLength"]
        decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).to(self.device)
        decoder = nn.DataParallel(decoder)

        checkpoint = torch.load(self.ckpt_path)
        decoder.load_state_dict(checkpoint["model_state_dict"])
        decoder = decoder.module
        decoder.eval()
        return decoder

    def latent_to_mesh(self, latent):
        self.decoder.eval()
        latent = latent.to(self.device)

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
            latent_inputs = latent.expand(sample_subset.size(0), -1)
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
        mesh.compute_vertex_normals()

        return mesh


def explore(epoch):
    # Latents saved by train_deep_sdf.py
    train_latents_path = Path(f"examples/cars/LatentCodes/{epoch}.pth")
    train_latents_dict = torch.load(train_latents_path)

    # Shape is (1608, 256)
    train_latents = train_latents_dict["latent_codes"]["weight"]

    # Load the split
    split_path = Path("examples/splits/sv2_cars_train.json")
    with open(split_path, "r") as f:
        split_data = json.load(f)
    object_ids = split_data["ShapeNetV2"]["02958343"]

    # Reconstructed latent codes of the training set
    recon_latent_dir = Path(
        f"examples/cars/Reconstructions/{epoch}/Codes/ShapeNetV2/02958343"
    )

    mesher = DeepSDFMesher(
        specs_path="examples/cars/specs.json",
        ckpt_path=f"examples/cars/ModelParameters/{epoch}.pth",
    )

    # Compare train latent and recon latent of training set
    for object_index, object_id in enumerate(object_ids):
        # Load recon latent.
        # Given pre-trained network parameters, we recover the latent codes for
        # each training samples. For each training sample, the recon_latent
        # shall be close to train_latent, as they are all coming from the
        # training set.
        recon_latent_path = recon_latent_dir / f"{object_id}.pth"
        recon_latent = torch.load(recon_latent_path).requires_grad_(False)
        recon_latent = recon_latent.cpu().flatten()
        recon_mesh = mesher.latent_to_mesh(recon_latent)
        recon_mesh.compute_vertex_normals()

        # Load train latent.
        # There are latent codes for all training samples, where the codes are
        # optimized together with the network parameters,
        train_latent = train_latents[object_index]
        train_mesh = mesher.latent_to_mesh(train_latent)
        train_mesh.compute_vertex_normals()

        # Visualize
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        scenes = vis.show_3d(
            [recon_mesh, axes],
            [train_mesh, axes],
            use_new_api=False,
        )
    pass


def all_train_latents_to_mesh(epoch):
    """
    Convert all train latent codes to mesh.

    Load latent codes from: examples/cars/LatentCodes/2000.pth
    Save meshes to        : examples/cars/LatentMeshes/2000/object_id.ply
    """
    train_latents_path = Path(f"examples/cars/LatentCodes/{epoch}.pth")
    train_latents_dict = torch.load(train_latents_path)

    # Assuming {"latent_codes:": {"weight": tensor}}
    train_latents = train_latents_dict["latent_codes"]["weight"]

    split_path = Path("examples/splits/sv2_cars_train.json")
    with open(split_path, "r") as f:
        split_data = json.load(f)
    object_ids = split_data["ShapeNetV2"]["02958343"]

    # The length of train_latents should match the length of object_ids
    assert len(train_latents) == len(object_ids)

    mesher = DeepSDFMesher(
        specs_path="examples/cars/specs.json",
        ckpt_path=f"examples/cars/ModelParameters/{epoch}.pth",
    )

    output_dir = Path(f"examples/cars/LatentMeshes/{epoch}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for object_index, object_id in tqdm(
        enumerate(object_ids),
        total=len(object_ids),
        desc="Generating Meshes",
    ):
        train_latent = train_latents[object_index].detach()
        mesh = mesher.latent_to_mesh(train_latent)
        mesh.compute_vertex_normals()

        mesh_path = output_dir / f"{object_id}.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)

        print(f"Saved mesh for object_id {object_id} at {mesh_path}")


def render_geometries_with_default_camera(geometries, height, width, visible=False):
    """
    Render a mesh using Open3D legacy visualizer. This requires a display.

    Args:
        mesh: Open3d TriangleMesh.
        height: int image height.
        width: int image width.
        visible: bool whether to show the window. Your machine must have a monitor.

    Returns:
        image: (H, W, 3) float32 np.ndarray image.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=visible)
    for geometry in geometries:
        vis.add_geometry(geometry)
    for geometry in geometries:
        vis.update_geometry(geometry)
    vis.poll_events()
    vis.update_renderer()
    buffer = vis.capture_screen_float_buffer()
    vis.destroy_window()

    return np.array(buffer)


def render_meshes_to_images(
    epoch,
    image_height=480,
    image_width=640,
    visible=False,
):
    """
    Render saved meshes to images.

    Args:
        epoch: int, epoch number of the meshes.
        output_dir: Path or str, directory to save the rendered images.
        image_height: int, the height of the rendered images.
        image_width: int, the width of the rendered images.
        visible: bool, if True the Open3D window will be shown (requires a display).
    """
    mesh_dir = Path(f"examples/cars/LatentMeshes/{epoch}")
    output_image_dir = Path(f"examples/cars/LatentImages/{epoch}")
    output_image_dir.mkdir(parents=True, exist_ok=True)

    for mesh_path in tqdm(
        list(mesh_dir.glob("*.ply")), desc="Rendering meshes to images"
    ):
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.compute_vertex_normals()
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        im_render = render_geometries_with_default_camera(
            geometries=[mesh, axes],
            height=image_height,
            width=image_width,
            visible=visible,
        )
        im_path = output_image_dir / f"{mesh_path.stem}.png"
        ct.io.imwrite(im_path, im_render)
        print(f"Saved rendered image to {im_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Explore or convert train and recon latent codes to mesh."
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="Explore train vs recon latent codes",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert train latent codes to mesh",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the mesh to images",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=2000,
        help="Epoch number for the files to use (default: 2000)",
    )

    args = parser.parse_args()

    # Check that only one action is specified
    if sum([args.explore, args.convert, args.render]) != 1:
        raise ValueError("Please specify exactly one action.")

    if args.explore:
        explore(args.epoch)
    elif args.convert:
        all_train_latents_to_mesh(args.epoch)
    elif args.render:
        render_meshes_to_images(args.epoch)


if __name__ == "__main__":
    main()
