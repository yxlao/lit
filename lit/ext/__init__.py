# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from pathlib import Path

from torch.utils.cpp_extension import load


def p(rel_path):
    abs_path = Path(__file__).parent / rel_path
    return str(abs_path)


lit_ext = load(
    name="lit_ext",
    sources=[
        p("lit_ext/bind.cpp"),
        p("lit_ext/lit_ext.cpp"),
    ],
    extra_cflags=["-O2"],
    extra_cuda_cflags=["-O2", "-Xcompiler -fno-gnu-unique"],
    verbose=True,
)
