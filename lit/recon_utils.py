import time

import camtools as ct
import igraph as ig
import numpy as np
import open3d as o3d
import torch

from lit.ext import lit_ext
from pcdet.utils.box_utils import box_to_corners_3d


def scale_bbox(
    bbox: np.ndarray,
    src_to_dst_scales: np.ndarray,
):
    """
    Args:
        bbox: (8,) or (7,), in OpenPCDet format.
              x, y, z, dx, dy, dz, heading, (label)
        src_to_dst_scales: (3,) scales.

    Notes:
        x: unchanged.
        y: unchanged.
        z: lower by dz_gap.
        dx: scaled by src_to_dst_scales[0].
        dy: scaled by src_to_dst_scales[1].
        dz: scaled by src_to_dst_scales[2].
        heading: unchanged.
        label: unchanged.

    Return:
        bbox: (8,) or (7,), in OpenPCDet format.
              x, y, z, dx, dy, dz, heading, (label)
    """
    assert isinstance(bbox, np.ndarray)
    assert bbox.ndim == 1
    assert bbox.shape[0] in [7, 8]

    src_to_dst_scales = np.asarray(src_to_dst_scales)
    assert src_to_dst_scales.ndim == 1
    assert src_to_dst_scales.shape[0] == 3

    scaled_bbox = bbox.copy()

    # Lower z
    dz = bbox[5]
    scaled_dz = dz * src_to_dst_scales[2]
    dz_gap = (dz - scaled_dz) / 2
    scaled_bbox[2] -= dz_gap

    # Scale dx, dy, dz.
    scaled_bbox[3:6] *= src_to_dst_scales

    return scaled_bbox


def scale_bboxes(
    bboxes: np.ndarray,
    src_to_dst_scales: np.ndarray,
):
    """
    Args:
        bboxes: (N, 8) or (N, 7), in OpenPCDet format.
                x, y, z, dx, dy, dz, heading, (label)
        src_to_dst_scales: (3,) scales.

    Notes:
        x: unchanged.
        y: unchanged.
        z: lower by dz_gap.
        dx: scaled by src_to_dst_scales[0].
        dy: scaled by src_to_dst_scales[1].
        dz: scaled by src_to_dst_scales[2].
        heading: unchanged.
        label: unchanged.

    Return:
        scaled_bboxes: (N, 8) or (N, 7), in OpenPCDet format.
                       x, y, z, dx, dy, dz, heading, (label)
    """
    assert isinstance(bboxes, np.ndarray)
    assert bboxes.ndim == 2
    assert bboxes.shape[1] in [7, 8]

    src_to_dst_scales = np.asarray(src_to_dst_scales)
    assert src_to_dst_scales.ndim == 1
    assert src_to_dst_scales.shape[0] == 3

    scaled_bboxes = bboxes.copy()

    # Lower z
    dz = bboxes[:, 5]
    scaled_dz = dz * src_to_dst_scales[2]
    dz_gap = (dz - scaled_dz) / 2
    scaled_bboxes[:, 2] -= dz_gap

    # Scale dx, dy, dz.
    scaled_bboxes[:, 3:6] *= src_to_dst_scales

    return scaled_bboxes


def scale_bboxes_by_domain(
    bboxes: np.ndarray,
    src_domain: str,
    dst_domain: str,
):
    """
    Args:
        bboxes: (N, 8) or (N, 7), in OpenPCDet format.
                x, y, z, dx, dy, dz, heading, (label)
        src_domain: "waymo", "nuscenes", or "kitti".
        dst_domain: "waymo", "nuscenes", or "kitti".

    Return:
        scaled_bboxes: (N, 8) or (N, 7), in OpenPCDet format.
                       x, y, z, dx, dy, dz, heading, (label)
    """
    assert (src_domain, dst_domain) in [
        ("waymo", "nuscenes"),
        ("waymo", "kitti"),
        ("nuscenes", "kitti"),
    ]

    # Read scales.
    scales_dict = {
        "waymo_to_kitti_bbox_scale": [
            0.8086602687835693,
            0.7795897722244263,
            0.854170024394989,
        ],
        "waymo_to_nuscenes_bbox_scale": [
            0.9662209153175354,
            0.9140116572380066,
            0.9629638195037842,
        ],
        "nuscenes_to_kitti_bbox_scale": [
            0.836931049823761,
            0.8529319763183594,
            0.8870219588279724,
        ],
    }
    src_to_dst_scales = scales_dict[f"{src_domain}_to_{dst_domain}_bbox_scale"]

    return scale_bboxes(
        bboxes=bboxes,
        src_to_dst_scales=src_to_dst_scales,
    )


def scale_points_with_bbox(
    points: np.ndarray,
    bbox: np.ndarray,
    src_to_dst_scales: np.ndarray,
):
    """
    Scaled points located inside a bbox, such that:
    - The extend of the bbox is scaled by src_to_dst_scales.
    - The x and y centers of the bbox remain unchanged.
    - The bbox still touches the ground. That is, the src bbox and dst bbox
      shares the same z_min.

    Args:
        points: (N, 3).
        bbox: (8,) or (7,), in OpenPCDet format.
              x, y, z, dx, dy, dz, heading, (label)
        src_to_dst_scales: (3,) scales.

    Return:
        scaled_points, scaled_bbox
    """
    assert isinstance(points, np.ndarray)
    assert points.ndim == 2
    assert points.shape[1] == 3

    assert isinstance(bbox, np.ndarray)
    assert bbox.ndim == 1
    assert bbox.shape[0] in [7, 8]

    src_to_dst_scales = np.asarray(src_to_dst_scales)
    assert src_to_dst_scales.ndim == 1
    assert src_to_dst_scales.shape[0] == 3

    # Center points to bbox center.
    bbox_center = bbox[:3]
    scaled_points = points.copy()
    scaled_points = scaled_points - bbox_center

    # Scale points by src_to_dst_scales.
    scaled_points = scaled_points * src_to_dst_scales

    # Put points back to the original bbox center.
    scaled_points = scaled_points + bbox_center

    # Lower points by dz_gap.
    dz = bbox[5]
    scaled_dz = dz * src_to_dst_scales[2]
    dz_gap = (dz - scaled_dz) / 2
    scaled_points[:, 2] -= dz_gap
    scaled_points = scaled_points.astype(points.dtype)

    # Compute scaled bbox.
    scaled_bbox = bbox.copy()
    scaled_bbox[2] -= dz_gap  # Lower z.
    scaled_bbox[3:6] *= src_to_dst_scales  # Scale dx, dy, dz.
    scaled_bbox = scaled_bbox.astype(bbox.dtype)

    return scaled_points, scaled_bbox


def scale_points_with_bbox_by_domain(
    points: np.ndarray,
    bbox: np.ndarray,
    src_domain: str,
    dst_domain: str,
):
    """
    Args:
        points: (N, 3).
        bbox: (8,) or (7,), in OpenPCDet format.
              x, y, z, dx, dy, dz, heading, (label)
        src_domain: "waymo", "nuscenes", or "kitti".
        dst_domain: "waymo", "nuscenes", or "kitti".

    Return:
        scaled_points, scaled_bbox
    """
    assert (src_domain, dst_domain) in [
        ("waymo", "nuscenes"),
        ("waymo", "kitti"),
        ("nuscenes", "kitti"),
    ]

    # Read scales.
    scales_dict = {
        "waymo_to_kitti_bbox_scale": [
            0.8086602687835693,
            0.7795897722244263,
            0.854170024394989,
        ],
        "waymo_to_nuscenes_bbox_scale": [
            0.9662209153175354,
            0.9140116572380066,
            0.9629638195037842,
        ],
        "nuscenes_to_kitti_bbox_scale": [
            0.836931049823761,
            0.8529319763183594,
            0.8870219588279724,
        ],
    }
    src_to_dst_scales = scales_dict[f"{src_domain}_to_{dst_domain}_bbox_scale"]

    return scale_points_with_bbox(
        points=points,
        bbox=bbox,
        src_to_dst_scales=src_to_dst_scales,
    )


def mesh_to_wire_frame(mesh: o3d.geometry.TriangleMesh):
    """
    Convert mesh to wire frame lineset.
    """
    assert isinstance(mesh, o3d.geometry.TriangleMesh)

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    edges_ab = triangles[:, [0, 1]]
    edges_bc = triangles[:, [1, 2]]
    edges_ca = triangles[:, [2, 0]]
    lines = np.concatenate([edges_ab, edges_bc, edges_ca], axis=0)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(vertices)
    ls.lines = o3d.utility.Vector2iVector(lines)

    return ls


def largest_cluster_mesh(mesh: o3d.geometry.TriangleMesh):
    """
    Args:
        mesh: open3d.geometry.TriangleMesh.

    Returns:
        mesh: open3d.geometry.TriangleMesh, of the largest cluster.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Get the vertex indices of the biggest cluster.
    g = ig.Graph()
    g.add_vertices(len(vertices))
    edges_a_b = triangles[:, [0, 1]]
    edges_b_c = triangles[:, [1, 2]]
    edges_c_a = triangles[:, [2, 0]]
    edges = np.concatenate([edges_a_b, edges_b_c, edges_c_a], axis=0)
    g.add_edges(edges)
    clusters = g.clusters()
    biggest_cluster_id = np.argmax(clusters.sizes())
    vert_ids = clusters[biggest_cluster_id]

    # Create mesh and select the biggest cluster.
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh = mesh.select_by_index(vert_ids)

    return mesh


# Context manager for temporarily changing the verbosity level to ERROR for Open3D.
# This is a bug in Open3D.
# [Open3D WARNING] invalid color in PaintUniformColor, clipping to [0, 1]
class SuppressOpen3DWarning:
    def __enter__(self):
        self.old_verbosity_level = o3d.utility.get_verbosity_level()
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    def __exit__(self, exc_type, exc_val, exc_tb):
        o3d.utility.set_verbosity_level(self.old_verbosity_level)


def _bbox_to_lineset_with_open3d(bbox, frame_pose=None):
    """
    Deprecated. This may generate differently ordered points compared to
    bbox_to_lineset().

    Args:
        bbox: (8,) or (7,), in OpenPCDet format.
        frame_pose: (4, 4) pose.
            The bbox will be transformed to world coordinate.

    Returns:
        An Open3D lineset.

    Notes:
        bbox: [x, y, z, dx, dy, dz, heading, class]
        - x      : center x.
        - y      : center y.
        - z      : center z.
        - dx     : full length in x direction before rotation.
        - dy     : full length in y direction before rotation.
        - dz     : full length in z direction before rotation.
        - heading: rotation angle around z axis in radian, positive is
                   counter-clockwise.

        It is not possible to transform bbox to another bbox with arbitrary
        pose, as bbox can only be rotated around z axis. Therefore, bbox is
        always used to represent a bbox in the local coordinate of the frame.
    """
    assert isinstance(bbox, np.ndarray)
    assert bbox.ndim == 1
    assert bbox.shape[0] in [7, 8]

    center = bbox[0:3]
    lwh = bbox[3:6]
    axis_angles = np.array([0, 0, bbox[6] + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

    with SuppressOpen3DWarning():
        ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # Transform points.
    if frame_pose is not None:
        points = np.asarray(ls.points)
        points = ct.transform.transform_points(points, frame_pose)
        ls.points = o3d.utility.Vector3dVector(points)

    # Assign colors.
    ls.colors = o3d.utility.Vector3dVector(np.zeros_like(ls.points))

    return ls


def obb_to_lineset(obb: o3d.geometry.OrientedBoundingBox, frame_pose=None):
    with SuppressOpen3DWarning():
        ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)

    # Transform points.
    if frame_pose is not None:
        points = np.asarray(ls.points)
        points = ct.transform.transform_points(points, frame_pose)
        ls.points = o3d.utility.Vector3dVector(points)

    return ls


def bbox_to_lineset(bbox, frame_pose=None):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1

    Args:
        bbox: (8,) or (7,), in OpenPCDet format.
        frame_pose: (4, 4) pose.
            The bbox will be transformed to world coordinate.

    Returns:
        An Open3D lineset.

    Notes:
        bbox: [x, y, z, dx, dy, dz, heading, class]
        - x      : center x.
        - y      : center y.
        - z      : center z.
        - dx     : full length in x direction before rotation.
        - dy     : full length in y direction before rotation.
        - dz     : full length in z direction before rotation.
        - heading: rotation angle around z axis in radian, positive is
                   counter-clockwise.

        It is not possible to transform bbox to another bbox with arbitrary
        pose, as bbox can only be rotated around z axis. Therefore, bbox is
        always used to represent a bbox in the local coordinate of the frame.
    """
    corners = box_to_corners_3d(bbox)
    lines = np.array(
        [
            # Bottom plane
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            # Top plane
            [4, 5],
            [5, 6],
            [6, 7],
            # Vertical lines
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
    )

    # Calculate the center of the top plane
    top_center = np.mean(corners[4:8], axis=0)

    # Determine the front of the bbox (midpoint between corners 4 and 5)
    top_front_midpoint = (corners[4] + corners[5]) / 2

    # Create the main bbox LineSet
    main_ls = o3d.geometry.LineSet()
    main_ls.points = o3d.utility.Vector3dVector(corners)
    main_ls.lines = o3d.utility.Vector2iVector(lines)
    main_ls.colors = o3d.utility.Vector3dVector(np.zeros((lines.shape[0], 3)))

    # Create a separate LineSet for the heading line
    heading_ls = o3d.geometry.LineSet()
    heading_ls.points = o3d.utility.Vector3dVector([top_center, top_front_midpoint])
    heading_ls.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
    heading_ls.colors = o3d.utility.Vector3dVector([[1, 0, 0]])

    # Combine the two LineSets
    combined_ls = main_ls + heading_ls

    # Apply transformation to the combined LineSet if frame_pose is provided
    if frame_pose is not None:
        combined_ls.transform(frame_pose)

    return combined_ls


def bbox_to_corners(bbox, frame_pose=None):
    """
    Args:
        bbox: (8,) or (7,), in OpenPCDet format.
        frame_pose: (4, 4) pose.
            The bbox will be transformed to world coordinate.

    Returns:
        Array of shape (8, 3), corner vertices of the bbox.
    """
    corners = box_to_corners_3d(bbox)
    if frame_pose is not None:
        corners = ct.transform.transform_points(corners, frame_pose)
    return corners


def bboxes_to_lineset(bboxes, frame_pose):
    """
    Args:
        boxes: (N, 8) boxes.
        frame_pose: (4, 4) pose.
            All boxes are in the same frame of the pose.
            The bbox will be transformed to world coordinate.

    Returns:
        A fused Open3D line set.

    Ref:
        tools/visual_utils/open3d_vis_utils.py::translate_boxes_to_open3d_instance()
    """
    assert isinstance(bboxes, np.ndarray)
    assert isinstance(frame_pose, np.ndarray)
    assert bboxes.ndim == 2
    assert bboxes.shape[1] in [7, 8]
    assert frame_pose.shape == (4, 4)

    frame_lineset = o3d.geometry.LineSet()
    for bbox in bboxes:
        frame_lineset += bbox_to_lineset(bbox, frame_pose=frame_pose)
    return frame_lineset


def bbox_to_open3d_obb(bbox):
    """
    Convert a bbox to Open3D OrientedBoundingBox.
    """
    assert isinstance(bbox, np.ndarray)
    assert bbox.ndim == 1
    assert bbox.shape[0] in [7, 8]

    center = bbox[:3]
    lwh = bbox[3:6]
    axis_angles = np.array([0, 0, bbox[6] + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    obb = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

    return obb


def get_indices_inside_bbox(points, bbox):
    """
    Return point indices within a bbox.
    TODO: consider class info in bbox.

    Args:
        points: (N, 3).
        bbox: (8,) or (7,), in OpenPCDet format.

    Returns:
        (M, 3) points in the bbox.
    """
    assert isinstance(points, np.ndarray)
    assert isinstance(bbox, np.ndarray)
    assert points.ndim == 2
    assert points.shape[1] == 3
    assert bbox.ndim == 1
    assert bbox.shape[0] in [7, 8]

    obb = bbox_to_open3d_obb(bbox)
    points_o3d = o3d.utility.Vector3dVector(points)
    inside_indices = np.array(obb.get_point_indices_within_bounding_box(points_o3d))
    return inside_indices


def get_indices_inside_bboxes(points, bboxes):
    """
    Return a list of point indices within multiple bboxes.

    Args:
        points: (N, 3).
        bboxes: (M, 8) or (M, 7), in OpenPCDet format.

    Returns:
        List[1D array]. Each array contains point indices in a bbox.

    Notes:
        1. This will return a list of 1D arrays, where
           get_indices_outside_bboxes() returns a single 1D array.
        2. If we call indices_inside_bbox() multiple times,
           o3d.utility.Vector3dVector() will be called multiple times, which is
           slow. Therefore, use indices_inside_bboxes() instead.
    """
    assert isinstance(points, np.ndarray)
    assert isinstance(bboxes, np.ndarray)
    assert points.ndim == 2
    assert points.shape[1] == 3
    assert bboxes.ndim == 2
    assert bboxes.shape[1] in [7, 8]

    points_o3d = o3d.utility.Vector3dVector(points)
    indices_inside_bboxes = []
    for bbox in bboxes:
        obb = bbox_to_open3d_obb(bbox)
        indices_inside_bboxes.append(
            np.array(obb.get_point_indices_within_bounding_box(points_o3d))
        )
    return indices_inside_bboxes


def get_indices_outside_bboxes(points, bboxes):
    """
    Return (combined, i.e. shape (N,)) point indices within multiple bboxes.

    Args:
        points: (N, 3).
        bboxes: (M, 8) or (M, 7), in OpenPCDet format.
    """
    assert isinstance(points, np.ndarray)
    assert isinstance(bboxes, np.ndarray)
    assert points.ndim == 2
    assert points.shape[1] == 3
    assert bboxes.ndim == 2
    assert bboxes.shape[1] in [7, 8]

    points_o3d = o3d.utility.Vector3dVector(points)

    inside_indices = set()
    for bbox in bboxes:
        obb = bbox_to_open3d_obb(bbox)
        inside_indices.update(
            np.array(obb.get_point_indices_within_bounding_box(points_o3d))
        )
    inside_indices = np.array(sorted(list(inside_indices)))

    all_indices = np.arange(len(points))
    outside_indices = all_indices[
        ~np.in1d(all_indices, inside_indices, assume_unique=True)
    ]
    return outside_indices


def bbox_corners_to_lineset(bbox_corners):
    """
    Args:
        bbox_corners: (8, 3), corners of a bbox.

    Returns:
        ls: open3d.geometry.LineSet, lines of the bbox.

       7 -------- 6                            7 -------- 6
      /|         /|                           /|         /|
     4-------- 5 .                           4-------- 5 .
     | |        | |                          | |        | |
     . 3 -------  2                          . 3 -------  2
     |/         |/                           |/         |/
     0 -------- 1                           0 -------- 1

       7 -------- 6                            7 -------- 6
      /|         /|                           /|         /|
     4-------- 5 .                           4-------- 5 .
     | |        | |                          | |        | |
     . 3 -------  2                          . 3 -------  2
     |/         |/                           |/         |/
     0 -------- 1                           0 -------- 1

       7 -------- 6                            7 -------- 6
      /|         /|                           /|         /|
     4-------- 5 .                           4-------- 5 .
     | |        | |                          | |        | |
     . 3 -------  2                          . 3 -------  2
     |/         |/                           |/         |/
     0 -------- 1                           0 -------- 1

                            Z  ^
                               | / X
                               |/
                        Y <----+

    - The origin is the lidar center.
    - The origin is above the ground.
    - Typically:
        - Edge 0-3 is the longest   (the length of the car).
        - Edge 0-1 is in the middle (the width of the car).
        - Edge 0-4 is the shortest  (the height of the car).
    """
    lines = [
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
    colors = [[1, 0, 0] for i in range(len(lines))]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(bbox_corners)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def extract_points_from_box(points, bbox):
    """
    Extract points from a bbox.
    TODO: consider class info in a bbox.

    Args:
        points: (N, 3).
        bbox: (8,) or (7,), in OpenPCDet format.

    Returns:
        (M, 3) points in the bbox.
    """
    assert isinstance(points, np.ndarray)
    assert isinstance(bbox, np.ndarray)
    assert points.ndim == 2
    assert points.shape[1] == 3
    assert bbox.ndim == 1
    assert bbox.shape[0] in [7, 8]

    def box_to_open3d_obb(bbox):
        """
        Convert a bbox to Open3D OrientedBoundingBox.
        """
        assert isinstance(bbox, np.ndarray)
        assert bbox.ndim == 1
        assert bbox.shape[0] in [7, 8]

        center = bbox[:3]
        lwh = bbox[3:6]
        axis_angles = np.array([0, 0, bbox[6] + 1e-10])
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
        obb = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

        return obb

    obb = box_to_open3d_obb(bbox)
    points_v3d = o3d.utility.Vector3dVector(points)
    ins_indices = np.array(obb.get_point_indices_within_bounding_box(points_v3d))
    # out_indices = np.setdiff1d(np.arange(points.shape[0]), ins_indices)

    return points[ins_indices]


def remove_statistical_outlier(
    points,
    lidar_centers=None,
    nb_neighbors=80,
    std_ratio=2.0,
    verbose=True,
):
    """
    Set nb_neighbors = 0 to disable.
    """
    if nb_neighbors == 0 or nb_neighbors is None:
        if lidar_centers is None:
            return points
        else:
            return points, lidar_centers

    if lidar_centers is not None and len(points) != len(lidar_centers):
        raise ValueError(
            f"len(points) != len(lidar_centers): {len(points)} != {len(lidar_centers)}"
        )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    num_src_points = len(pcd.points)
    pcd, select_indices = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    num_dst_points = len(pcd.points)

    if verbose:
        print(f"remove_statistical_outlier: {num_src_points} -> {num_dst_points}")

    points = points[select_indices]

    if lidar_centers is not None:
        lidar_centers = lidar_centers[select_indices]
        return points, lidar_centers
    else:
        return points


def voxel_downsample(
    points,
    lidar_centers,
    voxel_size=0.25,
    verbose=False,
):
    if len(points) != len(lidar_centers):
        raise ValueError(
            f"len(points) != len(lidar_centers): {len(points)} != {len(lidar_centers)}"
        )

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(points)
    dst_pcd = src_pcd.voxel_down_sample(voxel_size=voxel_size)
    dst_points = np.asarray(dst_pcd.points)

    k = 20
    dst_lidar_centers = []
    src_kdtree = o3d.geometry.KDTreeFlann(src_pcd)
    for dst_point in dst_points:
        [_, indices, _] = src_kdtree.search_knn_vector_3d(dst_point, k)
        k_src_lidar_centers = lidar_centers[indices]

        # Majority vote of lidar centers by row.
        unique_rows, counts = np.unique(
            k_src_lidar_centers,
            axis=0,
            return_counts=True,
        )
        dst_lidar_center = unique_rows[np.argmax(counts)]
        dst_lidar_centers.append(dst_lidar_center)
    dst_lidar_centers = np.asarray(dst_lidar_centers)

    if verbose:
        print(f"voxel_downsample: {len(points)} -> {len(dst_points)}")

    return dst_points, dst_lidar_centers


def rotate_points_y_front_to_x_front(points, inverse=False):
    """
    Rotate points 90 degrees, counter-clockwise, around z axes. This effectively
    rotates points from y-pointing-to-road_front to x-pointing-to-road_front.

    Args:
        points: [N, 3]
        inverse: If True, rotate from x-front to y-front.

    Returns:
        points: [N, 3] of rotated points.
    """
    points = np.asarray(points)
    assert points.ndim == 2
    assert points.shape[1] == 3

    if inverse:
        rot_matrix = np.array(
            [
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ]
        )
    else:
        rot_matrix = np.array(
            [
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1],
            ]
        )

    t_matrix = np.array([0, 0, 0])
    points = points @ rot_matrix.T + t_matrix

    return points


def rotate_bbox_y_front_to_x_front(bbox):
    """
    Args:
        bbox: [x, y, z, dx, dy, dz, heading, class]

    Returns:
        bbox: [x, y, z, dx, dy, dz, heading, class] of rotated bbox.
    """
    bbox = np.asarray(bbox)
    assert bbox.ndim == 1

    if bbox.shape[0] == 7:
        x, y, z, dx, dy, dz, heading = bbox
    elif bbox.shape[0] == 8:
        x, y, z, dx, dy, dz, heading, label = bbox
    else:
        raise ValueError(f"Unknown bbox shape: {bbox.shape}")

    new_x, new_y, new_z = rotate_points_y_front_to_x_front(np.array([[x, y, z]]))[0]
    new_dx = dy
    new_dy = dx
    new_dz = dz
    new_heading = heading

    if bbox.shape[0] == 7:
        new_bbox = np.array([new_x, new_y, new_z, new_dx, new_dy, new_dz, new_heading])
    elif bbox.shape[0] == 8:
        new_bbox = np.array(
            [new_x, new_y, new_z, new_dx, new_dy, new_dz, new_heading, label]
        )
    else:
        raise ValueError(f"Unknown bbox shape: {bbox.shape}")

    return new_bbox


def rotate_bboxes_y_front_to_x_front(bboxes, inverse=False):
    """
    Similar to rotate_bbox_y_front_to_x_front, but for multiple bboxes at the
    same time in a vectorized manner.

    Args:
        bboxes: [N, 7]
        inverse: If True, rotate from x-front to y-front.

    Returns:
        bboxes: [N, 7] of rotated bboxes.
    """
    bboxes = np.asarray(bboxes)
    assert bboxes.ndim == 2
    assert bboxes.shape[1] == 7

    # Rotate.
    new_xyzs = bboxes[:, :3]  # Unchanged.
    new_xyzs = rotate_points_y_front_to_x_front(new_xyzs, inverse=inverse)
    new_dx = bboxes[:, 4]  # Changed to dy.
    new_dy = bboxes[:, 3]  # Changed to dx.
    new_dz = bboxes[:, 5]  # Unchanged.
    new_headings = bboxes[:, 6]  # Unchanged.

    # Pack.
    new_bboxes = np.hstack(
        [
            new_xyzs,
            new_dx[:, None],
            new_dy[:, None],
            new_dz[:, None],
            new_headings[:, None],
        ]
    )

    return new_bboxes


def incident_angles_to_colors(incident_angles):
    """
    Args:
        incident_angles: (N, ) array from 0 to pi.

    Return:
        colors: (N, 3) array in floats.
    """

    # Normalize incident_angles to range [0, 1]
    normalized_angles = incident_angles / np.pi

    # Create a color map: we'll use a simple red-to-blue gradient.
    # Red (1, 0, 0) for angle = pi, and Blue (0, 0, 1) for angle = 0
    colors = np.zeros((len(incident_angles), 3))  # Initialize color array
    colors[:, 0] = normalized_angles  # Red channel
    colors[:, 2] = 1 - normalized_angles  # Blue channel

    return colors


def split_mesh_by_cc(mesh: o3d.geometry.TriangleMesh):
    """
    Split mesh by connected components.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertices = torch.tensor(vertices, dtype=torch.float32)
    triangles = torch.tensor(triangles, dtype=torch.int64)

    start_time = time.time()
    results = lit_ext.split_mesh_by_cc(vertices, triangles)
    print(f"split_mesh_by_cc: {time.time() - start_time:.2f} s")

    meshes = []
    for vertices, triangles in results:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices.numpy())
        mesh.triangles = o3d.utility.Vector3iVector(triangles.numpy())
        meshes.append(mesh)

    return meshes
