from pathlib import Path
import json
import argparse
from explore_results import render_geometries_with_default_camera
import open3d as o3d
from tqdm import tqdm
import camtools as ct
import numpy as np

_data_root = Path("data")
_synset_id = "02958343"

# Results from pre-processing
_norm_params_dir = _data_root / "NormalizationParameters" / "ShapeNetV2" / _synset_id
_sdf_samples_dir = _data_root / "SdfSamples" / "ShapeNetV2" / _synset_id
_surface_samples_dir = _data_root / "SurfaceSamples" / "ShapeNetV2" / _synset_id

# Rendered point clouds
_legit_point_cloud_dir = Path("examples") / "cars" / "LegitPointClouds"
_select_point_cloud_dir = Path("examples") / "cars" / "SelectPointClouds"

# Output lists
_legit_json_path = Path(f"examples/splits/sv2_cars_legit.json")
_train_json_path = Path(f"examples/splits/sv2_cars_train.json")
_test_json_path = Path(f"examples/splits/sv2_cars_test.json")


def get_object_ids_from_dir(data_dir):
    """
    Extract object IDs from filenames in the given directory path.
    Object IDs are derived from filenames by removing the file extension.
    """
    object_ids = set()
    for path in data_dir.iterdir():
        if path.is_file():
            object_ids.add(path.stem)
    return object_ids


def write_object_ids(synset_id, object_ids, output_file_path):
    """
    Write the given object IDs to a JSON file with the specified structure
    and print the number of items written.
    """
    data = {"ShapeNetV2": {synset_id: object_ids}}

    # Ensure the output directory exists
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write data to the JSON file
    with open(output_file_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"{len(object_ids)} items written to {output_file_path}")


def gen_legit():
    # Extract object IDs from each directory
    norm_object_ids = get_object_ids_from_dir(_norm_params_dir)
    sdf_object_ids = get_object_ids_from_dir(_sdf_samples_dir)
    surface_object_ids = get_object_ids_from_dir(_surface_samples_dir)

    # Find the intersection of object IDs across all directories
    object_ids = sorted(
        norm_object_ids.intersection(sdf_object_ids, surface_object_ids)
    )

    # Write to json
    write_object_ids(
        synset_id=_synset_id,
        object_ids=object_ids,
        output_file_path=_legit_json_path,
    )


def render_legit():
    # Read json to get object ids
    with open(_legit_json_path, "r") as f:
        data = json.load(f)
    object_ids = data["ShapeNetV2"][_synset_id]

    # Render point clouds
    for object_id in tqdm(object_ids, desc="Rendering point clouds"):
        pcd_path = _surface_samples_dir / f"{object_id}.ply"
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        # Rotate pcd along y-axis by 90 degrees.
        transform = np.array(
            [
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 1],
            ]
        )
        pcd.transform(transform)
        im_render = render_geometries_with_default_camera(
            geometries=[pcd], height=480, width=640, visible=False
        )
        im_render = ct.image.crop_white_boarders(im_render)
        _legit_point_cloud_dir.mkdir(parents=True, exist_ok=True)
        pcd_path = _legit_point_cloud_dir / f"{object_id}.png"
        ct.io.imwrite(pcd_path, im_render)


def gen_train_test_split():
    # Split object IDs for train and test
    split_index = int(len(object_ids) * 0.9)
    train_object_ids = object_ids[:split_index]
    test_object_ids = object_ids[split_index:]

    write_object_ids(
        synset_id=synset_id,
        object_ids=train_object_ids,
        output_file_path=_train_json_path,
    )
    write_object_ids(
        synset_id=synset_id,
        object_ids=test_object_ids,
        output_file_path=_test_json_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Explore or convert train and recon latent codes to mesh."
    )
    parser.add_argument(
        "--gen_legit",
        action="store_true",
        help="Generate legit (can be pre-processed) sample list to sv2_cars_legit.json",
    )
    parser.add_argument(
        "--render_legit",
        action="store_true",
        help="Render all point clouds in sv2_cars_legit.json to examples/cars/LegitPointClouds",
    )
    parser.add_argument(
        "--gen_train_test_split",
        action="store_true",
        help="Generate train (0.95) and test (0.05) split from examples/cars/SelectPointClouds",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=2000,
        help="Epoch number for the files to use (default: 2000)",
    )
    args = parser.parse_args()

    if sum([args.gen_legit, args.render_legit, args.gen_train_test_split]) != 1:
        raise ValueError("Exactly one action must be specified")

    if args.gen_legit:
        gen_legit()
    elif args.render_legit:
        render_legit()
    elif args.gen_train_test_split:
        gen_train_test_split()


if __name__ == "__main__":
    main()
