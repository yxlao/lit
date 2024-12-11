#include <torch/extension.h>

#include "lit_ext.h"

using namespace pybind11::literals;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("split_mesh_by_cc", &split_mesh_by_cc,
          "Split mesh into meshes by connected components.", "vertices"_a,
          "triangles"_a);
}
