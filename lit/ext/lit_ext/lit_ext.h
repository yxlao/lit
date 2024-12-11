#pragma once

#include <torch/extension.h>

/// @brief Split a mesh into multiple meshes based on connected components.
///
/// @param vertices Tensor of vertices. Shape: (num_vertices, 3).
/// @param triangles Tensor of triangle indices. Each row represents a triangle,
/// with each element being an index into the vertices tensor. Shape:
/// (num_triangles, 3).
/// @return A vector of pairs, where each pair contains the vertices and
/// triangles tensors of a connected component.
std::vector<std::pair<torch::Tensor, torch::Tensor>> split_mesh_by_cc(
        const torch::Tensor &vertices, const torch::Tensor &triangles);
