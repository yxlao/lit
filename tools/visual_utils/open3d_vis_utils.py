"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""

import matplotlib
import numpy as np
import open3d
import torch

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[: max_color_num + 1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(
    points,
    gt_boxes=None,
    ref_boxes=None,
    ref_labels=None,
    ref_scores=None,
    point_colors=None,
    draw_origin=True,
    white_background=True,
    src_points=None,
    vertices=None,
    triangles=None,
    rays_o=None,
    rays_d=None,
    rays_length=10,
    rays_subsample=0.01,
):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    if not white_background:
        vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(
            size=5.0, origin=[0, 0, 0]
        )
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        if white_background:
            pts.colors = open3d.utility.Vector3dVector(np.zeros((points.shape[0], 3)))
        else:
            pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    # Draw src_points in red
    if src_points is not None:
        src_pts = open3d.geometry.PointCloud()
        src_pts.points = open3d.utility.Vector3dVector(src_points[:, :3])
        src_pts.paint_uniform_color([1, 0, 0])
        vis.add_geometry(src_pts)

    # Draw mesh
    if vertices is not None and triangles is not None:
        mesh = open3d.geometry.TriangleMesh(
            vertices=open3d.utility.Vector3dVector(vertices),
            triangles=open3d.utility.Vector3iVector(triangles),
        )
        mesh.compute_vertex_normals()
        vis.add_geometry(mesh)

    # Draw rays as lineset
    if rays_o is not None and rays_d is not None:
        # Subsample rays
        num_rays = int(rays_o.shape[0] * rays_subsample)
        frame_indices = np.random.choice(rays_o.shape[0], num_rays, replace=False)
        rays_o = rays_o[frame_indices]
        rays_d = rays_d[frame_indices]

        rays_t = rays_o + rays_d * rays_length
        ls = open3d.geometry.LineSet()
        points = np.concatenate([rays_o, rays_t], axis=0)
        lines = np.array([[i, i + rays_o.shape[0]] for i in range(rays_o.shape[0])])
        ls.points = open3d.utility.Vector3dVector(points)
        ls.lines = open3d.utility.Vector2iVector(lines)
        ls.colors = open3d.utility.Vector3dVector(
            np.tile([0, 0, 1], (rays_o.shape[0], 1))
        )
        vis.add_geometry(ls)

    vis.run()
    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
       4-------- 6
     /|         /|
    5 -------- 3 .
    | |        | |
    . 7 -------- 1
    |/         |/
    2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis
