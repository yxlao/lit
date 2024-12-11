import os
import subprocess
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name, module, sources):
    print(f"Making CUDA extension: {name}...")
    cuda_ext = CUDAExtension(
        name="%s.%s" % (module, name),
        sources=[os.path.join(*module.split("."), src) for src in sources],
    )
    return cuda_ext


def main():
    version = "0.3.0"

    setup(
        name="lit",
        version=version,
        description="LiT: LiDAR Translator",
        install_requires=[
            "numpy",
            "torch>=1.1",
            "spconv",
            "numba",
            "tensorboardX",
            "easydict",
            "pyyaml",
        ],
        license="MIT",
        packages=["pcdet", "lit"],
        cmdclass={"build_ext": BuildExtension},
        ext_modules=[
            make_cuda_ext(
                name="iou3d_nms_cuda",
                module="pcdet.ops.iou3d_nms",
                sources=[
                    "src/iou3d_cpu.cpp",
                    "src/iou3d_nms_api.cpp",
                    "src/iou3d_nms.cpp",
                    "src/iou3d_nms_kernel.cu",
                ],
            ),
            make_cuda_ext(
                name="roiaware_pool3d_cuda",
                module="pcdet.ops.roiaware_pool3d",
                sources=[
                    "src/roiaware_pool3d.cpp",
                    "src/roiaware_pool3d_kernel.cu",
                ],
            ),
            make_cuda_ext(
                name="roipoint_pool3d_cuda",
                module="pcdet.ops.roipoint_pool3d",
                sources=[
                    "src/roipoint_pool3d.cpp",
                    "src/roipoint_pool3d_kernel.cu",
                ],
            ),
            make_cuda_ext(
                name="pointnet2_stack_cuda",
                module="pcdet.ops.pointnet2.pointnet2_stack",
                sources=[
                    "src/pointnet2_api.cpp",
                    "src/ball_query.cpp",
                    "src/ball_query_gpu.cu",
                    "src/group_points.cpp",
                    "src/group_points_gpu.cu",
                    "src/sampling.cpp",
                    "src/sampling_gpu.cu",
                    "src/interpolate.cpp",
                    "src/interpolate_gpu.cu",
                ],
            ),
            make_cuda_ext(
                name="pointnet2_batch_cuda",
                module="pcdet.ops.pointnet2.pointnet2_batch",
                sources=[
                    "src/pointnet2_api.cpp",
                    "src/ball_query.cpp",
                    "src/ball_query_gpu.cu",
                    "src/group_points.cpp",
                    "src/group_points_gpu.cu",
                    "src/interpolate.cpp",
                    "src/interpolate_gpu.cu",
                    "src/sampling.cpp",
                    "src/sampling_gpu.cu",
                ],
            ),
        ],
    )


if __name__ == "__main__":
    main()
