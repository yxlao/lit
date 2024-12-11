#include <torch/extension.h>

#include <iostream>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

std::pair<torch::Tensor, torch::Tensor> select_by_index(
        const torch::Tensor &vertices,
        const torch::Tensor &triangles,
        const std::vector<int64_t> &vertex_indices) {
    std::unordered_map<int64_t, int64_t> new_vertex_map;
    std::vector<int64_t> new_vertex_indices;

    for (size_t i = 0; i < vertex_indices.size(); ++i) {
        int64_t old_index = vertex_indices[i];
        if (new_vertex_map.find(old_index) == new_vertex_map.end()) {
            int64_t new_index = new_vertex_indices.size();
            new_vertex_indices.push_back(old_index);
            new_vertex_map[old_index] = new_index;
        }
    }

    torch::Tensor new_vertices = vertices.index_select(
            0, torch::tensor(new_vertex_indices, torch::kInt64));

    std::vector<int64_t> new_triangle_indices;
    auto triangles_accessor = triangles.accessor<int64_t, 2>();
    for (int64_t i = 0; i < triangles.size(0); ++i) {
        int64_t v0 = triangles_accessor[i][0];
        int64_t v1 = triangles_accessor[i][1];
        int64_t v2 = triangles_accessor[i][2];

        if (new_vertex_map.find(v0) != new_vertex_map.end() &&
            new_vertex_map.find(v1) != new_vertex_map.end() &&
            new_vertex_map.find(v2) != new_vertex_map.end()) {
            new_triangle_indices.push_back(i);
        }
    }

    torch::Tensor new_triangles =
            torch::empty({static_cast<int64_t>(new_triangle_indices.size()), 3},
                         triangles.options());
    auto new_triangles_accessor = new_triangles.accessor<int64_t, 2>();
    for (size_t i = 0; i < new_triangle_indices.size(); ++i) {
        int64_t idx = new_triangle_indices[i];
        new_triangles_accessor[i][0] =
                new_vertex_map[triangles_accessor[idx][0]];
        new_triangles_accessor[i][1] =
                new_vertex_map[triangles_accessor[idx][1]];
        new_triangles_accessor[i][2] =
                new_vertex_map[triangles_accessor[idx][2]];
    }

    return std::make_pair(new_vertices, new_triangles);
}

std::vector<std::pair<torch::Tensor, torch::Tensor>> split_mesh_by_cc(
        const torch::Tensor &vertices, const torch::Tensor &triangles) {
    AT_ASSERT(vertices.dtype() == torch::kFloat32, "Vertices must be float32");
    AT_ASSERT(triangles.dtype() == torch::kInt64, "Triangles must be int64");

    AT_ASSERT(vertices.dim() == 2 && vertices.size(1) == 3,
              "Vertices must have a shape of (N, 3)");
    AT_ASSERT(triangles.dim() == 2 && triangles.size(1) == 3,
              "Triangles must have a shape of (M, 3)");

    // Ensure tensors are on CPU
    auto cpu_vertices = vertices.to(torch::kCPU);
    auto cpu_triangles = triangles.to(torch::kCPU);

    auto vertices_accessor = cpu_vertices.accessor<float, 2>();
    auto triangles_accessor = cpu_triangles.accessor<int64_t, 2>();

    int64_t n_vertices = cpu_vertices.size(0);
    int64_t n_triangles = cpu_triangles.size(0);

    // Build adjacency list
    std::vector<std::unordered_set<int64_t>> adjacency(n_vertices);
    for (int64_t i = 0; i < n_triangles; ++i) {
        adjacency[triangles_accessor[i][0]].insert(triangles_accessor[i][1]);
        adjacency[triangles_accessor[i][1]].insert(triangles_accessor[i][2]);
        adjacency[triangles_accessor[i][2]].insert(triangles_accessor[i][0]);
    }

    // Find connected components using BFS
    std::vector<bool> visited(n_vertices, false);
    std::vector<std::vector<int64_t>> components;
    for (int64_t i = 0; i < n_vertices; ++i) {
        if (!visited[i]) {
            std::vector<int64_t> component;
            std::queue<int64_t> queue;
            queue.push(i);
            visited[i] = true;

            while (!queue.empty()) {
                int64_t v = queue.front();
                queue.pop();
                component.push_back(v);

                for (auto adj_v : adjacency[v]) {
                    if (!visited[adj_v]) {
                        queue.push(adj_v);
                        visited[adj_v] = true;
                    }
                }
            }
            components.push_back(component);
        }
    }

    // Split meshes
    std::vector<std::pair<torch::Tensor, torch::Tensor>> split_meshes;
    for (const auto &comp : components) {
        auto split_mesh = select_by_index(cpu_vertices, cpu_triangles, comp);
        split_meshes.push_back(split_mesh);
    }

    return split_meshes;
}
