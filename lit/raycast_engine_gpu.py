"""
GPU Raycast engine for mesh-ray intersection.
"""

import ctypes
import os
import pickle
from pathlib import Path

import camtools as ct
import cupy as cp
import numpy as np
import open3d as o3d
import optix
from pynvrtc.compiler import Program

from lit.lidar import Lidar
from lit.path_utils import LitPaths
from lit.raycast_engine import RaycastEngine
from lit.raycast_engine_cpu import RaycastEngineCPU


class RaycastEngineGPU(RaycastEngine):
    """
    Wrapper to store "global" variables and provide a better API.
    """

    # All possible OptiX include paths. The first one that exists will be used.
    otk_pyoptix_root = LitPaths._lit_root / "lit" / "extern" / "otk-pyoptix"
    _optix_include_paths = [
        otk_pyoptix_root / "sdk/NVIDIA-OptiX-SDK-7.4.0-linux64-x86_64/include",
        Path.home() / "bin/NVIDIA-OptiX-SDK-7.4.0-linux64-x86_64/include",
    ]

    # System-wide CUDA include path.
    _cuda_include_path = Path("/usr/local/cuda/include")

    # Additional include paths for stddef.h. This is needed for Optix 7.0.
    _stddef_include_path = None

    # Path to OptiX kernel file, including our custom kernels.
    _kernel_file_path = Path(__file__).parent / "raycast_engine_gpu.cu"

    class Logger:
        def __init__(self):
            self.num_messages = 0

        def __call__(self, level, tag, mssg):
            print("[{:>2}][{:>12}]: {}".format(level, tag, mssg))
            self.num_messages += 1

    def __init__(self, verbose=False) -> None:
        # Create OptiX context.
        self.verbose = verbose
        self.logger = RaycastEngineGPU.Logger()
        self.ctx = self._create_ctx(
            self.logger, verbosity_level=4 if self.verbose else 0
        )

        # Compile OptiX pipeline.
        self.pipeline_options = RaycastEngineGPU._set_pipeline_options()
        kernel_ptx = RaycastEngineGPU._compile_cuda(
            str(RaycastEngineGPU._kernel_file_path)
        )
        self.module = self._create_module(
            self.ctx,
            self.pipeline_options,
            kernel_ptx,
        )
        self.prog_groups = self._create_program_groups(
            self.ctx,
            self.module,
        )
        self.pipeline = self._create_pipeline(
            self.ctx,
            self.prog_groups,
            self.pipeline_options,
        )
        # Filled by self._create_sbt(), which is called by set_geometry.
        self.d_raygen_sbt = None
        self.d_miss_sbt = None
        self.d_hitgroup_sbt = None
        self.sbt = None

        # These properties are set by set_geometry().
        self.gas_handle = None
        self.d_gas_output_buffer = None
        self.cp_vertices = None
        self.cp_triangles = None

    def _create_ctx(self, logger, verbosity_level):
        # OptiX param can be set with optional keyword constructor arguments.
        ctx_options = optix.DeviceContextOptions(
            logCallbackFunction=logger, logCallbackLevel=verbosity_level
        )

        # They can also be set and queried as properties on the struct.
        if optix.version()[1] >= 2:
            ctx_options.validationMode = optix.DEVICE_CONTEXT_VALIDATION_MODE_ALL

        cu_ctx = 0
        return optix.deviceContextCreate(cu_ctx, ctx_options)

    def _create_accel(self, ctx, np_vertices, np_triangles):
        """
        Args:
            ctx: Optix context.
            np_vertices: (N, 3) array of vertices.
            np_triangles: (M, 3) array of triangle indices.
        """
        ct.sanity.assert_shape_nx3(np_vertices, name="np_vertices")
        ct.sanity.assert_shape_nx3(np_triangles, name="np_triangles")

        accel_options = optix.AccelBuildOptions(
            buildFlags=int(optix.BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS),
            operation=optix.BUILD_OPERATION_BUILD,
        )

        np_vertices = np_vertices.ravel()
        np_triangles = np_triangles.ravel()

        self.cp_vertices = cp.array(np_vertices, dtype="f4")
        self.cp_triangles = cp.array(np_triangles, dtype="u4")

        triangle_input_flags = [optix.GEOMETRY_FLAG_NONE]  # One flag is sufficient
        triangle_input = optix.BuildInputTriangleArray()
        triangle_input.vertexFormat = optix.VERTEX_FORMAT_FLOAT3
        triangle_input.numVertices = len(self.cp_vertices) // 3
        triangle_input.vertexBuffers = [self.cp_vertices.data.ptr]
        triangle_input.indexFormat = optix.INDICES_FORMAT_UNSIGNED_INT3
        triangle_input.numIndexTriplets = len(self.cp_triangles) // 3
        triangle_input.indexBuffer = self.cp_triangles.data.ptr
        triangle_input.flags = triangle_input_flags
        triangle_input.numSbtRecords = 1

        gas_buffer_sizes = ctx.accelComputeMemoryUsage(
            [accel_options], [triangle_input]
        )

        d_temp_buffer_gas = cp.cuda.alloc(gas_buffer_sizes.tempSizeInBytes)
        d_gas_output_buffer = cp.cuda.alloc(gas_buffer_sizes.outputSizeInBytes)

        gas_handle = ctx.accelBuild(
            0,  # CUDA stream
            [accel_options],
            [triangle_input],
            d_temp_buffer_gas.ptr,
            gas_buffer_sizes.tempSizeInBytes,
            d_gas_output_buffer.ptr,
            gas_buffer_sizes.outputSizeInBytes,
            [],  # emitted properties
        )

        return (gas_handle, d_gas_output_buffer)

    def _create_module(self, ctx, pipeline_options, triangle_ptx):
        module_options = optix.ModuleCompileOptions(
            maxRegisterCount=optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            optLevel=optix.COMPILE_OPTIMIZATION_DEFAULT,
            debugLevel=optix.COMPILE_DEBUG_LEVEL_DEFAULT,
        )
        module, log = ctx.moduleCreateFromPTX(
            module_options, pipeline_options, triangle_ptx
        )
        return module

    def _create_program_groups(self, ctx, module):
        raygen_prog_group_desc = optix.ProgramGroupDesc()
        raygen_prog_group_desc.raygenModule = module
        raygen_prog_group_desc.raygenEntryFunctionName = "__raygen__rg"
        raygen_prog_group, log = ctx.programGroupCreate([raygen_prog_group_desc])

        miss_prog_group_desc = optix.ProgramGroupDesc()
        miss_prog_group_desc.missModule = module
        miss_prog_group_desc.missEntryFunctionName = "__miss__ms"
        miss_prog_group, log = ctx.programGroupCreate([miss_prog_group_desc])

        hitgroup_prog_group_desc = optix.ProgramGroupDesc()
        hitgroup_prog_group_desc.hitgroupModuleCH = module
        hitgroup_prog_group_desc.hitgroupEntryFunctionNameCH = "__closesthit__ch"
        hitgroup_prog_group, log = ctx.programGroupCreate([hitgroup_prog_group_desc])

        return [raygen_prog_group[0], miss_prog_group[0], hitgroup_prog_group[0]]

    def _create_pipeline(self, ctx, program_groups, pipeline_compile_options):
        max_trace_depth = 1
        pipeline_link_options = optix.PipelineLinkOptions()
        pipeline_link_options.maxTraceDepth = max_trace_depth
        pipeline_link_options.debugLevel = optix.COMPILE_DEBUG_LEVEL_FULL

        log = ""
        pipeline = ctx.pipelineCreate(
            pipeline_compile_options, pipeline_link_options, program_groups, log
        )

        stack_sizes = optix.StackSizes()
        for prog_group in program_groups:
            optix.util.accumulateStackSizes(prog_group, stack_sizes)

        (
            dc_stack_size_from_trav,
            dc_stack_size_from_state,
            cc_stack_size,
        ) = optix.util.computeStackSizes(
            stack_sizes,
            max_trace_depth,
            0,  # maxCCDepth
            0,  # maxDCDepth
        )
        pipeline.setStackSize(
            dc_stack_size_from_trav,
            dc_stack_size_from_state,
            cc_stack_size,
            1,  # maxTraversableDepth
        )

        return pipeline

    def _create_sbt(self, prog_groups):
        """
        TODO: this relies on self.cp_vertices and self.triangles, which are set by
        self._create_accel(). This is not ideal.
        """

        raygen_prog_group, miss_prog_group, hitgroup_prog_group = prog_groups

        header_format = "{}B".format(optix.SBT_RECORD_HEADER_SIZE)

        # Raygen record
        formats = [header_format]
        itemsize = RaycastEngineGPU._get_aligned_itemsize(
            formats, optix.SBT_RECORD_ALIGNMENT
        )
        dtype = np.dtype(
            {
                "names": ["header"],
                "formats": formats,
                "itemsize": itemsize,
                "align": True,
            }
        )
        h_raygen_sbt = np.array([0], dtype=dtype)
        optix.sbtRecordPackHeader(raygen_prog_group, h_raygen_sbt)
        d_raygen_sbt = RaycastEngineGPU._array_to_device_memory(h_raygen_sbt)

        # Miss record
        formats = [header_format, "f4", "f4", "f4"]
        itemsize = RaycastEngineGPU._get_aligned_itemsize(
            formats, optix.SBT_RECORD_ALIGNMENT
        )
        dtype = np.dtype(
            {
                "names": ["header", "r", "g", "b"],
                "formats": formats,
                "itemsize": itemsize,
                "align": True,
            }
        )
        h_miss_sbt = np.array([(0, 0.3, 0.1, 0.2)], dtype=dtype)  # MissData
        optix.sbtRecordPackHeader(miss_prog_group, h_miss_sbt)
        d_miss_sbt = RaycastEngineGPU._array_to_device_memory(h_miss_sbt)

        # Hitgroup record.
        formats = [header_format, ctypes.c_void_p, ctypes.c_void_p]
        itemsize = RaycastEngineGPU._get_aligned_itemsize(
            formats, optix.SBT_RECORD_ALIGNMENT
        )
        dtype = np.dtype(
            {
                "names": ["header", "vertex_buffer", "triangle_buffer"],
                "formats": formats,
                "itemsize": itemsize,
                "align": True,
            }
        )

        assert isinstance(self.cp_vertices, cp.ndarray)
        vertex_buffer_ptr_value = ctypes.c_void_p(self.cp_vertices.data.ptr).value

        assert isinstance(self.cp_triangles, cp.ndarray)
        triangle_buffer_ptr_value = ctypes.c_void_p(self.cp_triangles.data.ptr).value

        h_hitgroup_sbt = np.array(
            [(0, vertex_buffer_ptr_value, triangle_buffer_ptr_value)],
            dtype=dtype,
        )
        optix.sbtRecordPackHeader(hitgroup_prog_group, h_hitgroup_sbt)
        d_hitgroup_sbt = RaycastEngineGPU._array_to_device_memory(h_hitgroup_sbt)

        sbt = optix.ShaderBindingTable(
            raygenRecord=d_raygen_sbt.ptr,
            missRecordBase=d_miss_sbt.ptr,
            missRecordStrideInBytes=h_miss_sbt.dtype.itemsize,
            missRecordCount=1,
            hitgroupRecordBase=d_hitgroup_sbt.ptr,
            hitgroupRecordStrideInBytes=h_hitgroup_sbt.dtype.itemsize,
            hitgroupRecordCount=1,
        )

        return d_raygen_sbt, d_miss_sbt, d_hitgroup_sbt, sbt

    @staticmethod
    def _set_pipeline_options():
        if optix.version()[1] >= 2:
            return optix.PipelineCompileOptions(
                usesMotionBlur=False,
                traversableGraphFlags=int(
                    optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS
                ),
                numPayloadValues=4,  # Check __raygen__rg() to see num payload values.
                numAttributeValues=3,
                exceptionFlags=int(optix.EXCEPTION_FLAG_NONE),
                pipelineLaunchParamsVariableName="params",
                usesPrimitiveTypeFlags=optix.PRIMITIVE_TYPE_FLAGS_TRIANGLE,
            )
        else:
            return optix.PipelineCompileOptions(
                usesMotionBlur=False,
                traversableGraphFlags=int(
                    optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS
                ),
                numPayloadValues=4,  # Check __raygen__rg() to see num payload values.
                numAttributeValues=3,
                exceptionFlags=int(optix.EXCEPTION_FLAG_NONE),
                pipelineLaunchParamsVariableName="params",
            )

    @staticmethod
    def _get_aligned_itemsize(formats, alignment):
        def round_up(val, mult_of):
            if val % mult_of == 0:
                return val
            else:
                return val + mult_of - val % mult_of

        names = []
        for i in range(len(formats)):
            names.append("x" + str(i))

        temp_dtype = np.dtype({"names": names, "formats": formats, "align": True})
        return round_up(temp_dtype.itemsize, alignment)

    @staticmethod
    def _optix_version_gte(version):
        if optix.version()[0] > version[0]:
            return True
        if optix.version()[0] == version[0] and optix.version()[1] >= version[1]:
            return True
        return False

    @staticmethod
    def _array_to_device_memory(numpy_array, stream=cp.cuda.Stream()):
        byte_size = numpy_array.size * numpy_array.dtype.itemsize

        h_ptr = ctypes.c_void_p(numpy_array.ctypes.data)
        d_mem = cp.cuda.memory.alloc(byte_size)
        d_mem.copy_from_async(h_ptr, byte_size, stream)
        return d_mem

    @staticmethod
    def _compile_cuda(cuda_file_path):
        cuda_file_path = str(cuda_file_path)
        with open(cuda_file_path, "rb") as f:
            src = f.read()
        nvrtc_dll = os.environ.get("NVRTC_DLL")
        if nvrtc_dll is None:
            nvrtc_dll = ""
        else:
            print(f"NVRTC_DLL = {nvrtc_dll}")
        prog = Program(src.decode(), cuda_file_path, lib_name=nvrtc_dll)

        # Check the fist existing OptiX include path.
        optix_include_path = None
        for path in RaycastEngineGPU._optix_include_paths:
            if path.exists():
                optix_include_path = str(path)
                break
        if optix_include_path is None:
            raise RuntimeError(f"OptiX include path does not exist: {path}")

        # Check CUDA include path.
        if not RaycastEngineGPU._cuda_include_path.exists():
            raise RuntimeError(
                f"CUDA include path does not exist: {RaycastEngineGPU._cuda_include_path}"
            )
        cuda_include_path = str(RaycastEngineGPU._cuda_include_path)

        compile_options = [
            "-use_fast_math",
            "-lineinfo",
            "-default-device",
            "-std=c++11",
            "-rdc",
            "true",
            f"-I{cuda_include_path}",
            f"-I{optix_include_path}",
        ]

        # Optix 7.0 compiles need path to system <stddef.h>. The value of
        # optix.stddef_path is compiled in constant.
        if optix.version()[1] == 0:
            compile_options.append(f"-I{RaycastEngineGPU._stddef_include_path}")

        ptx = prog.compile(compile_options)
        return ptx

    def set_geometry(self, vertices: np.ndarray, triangles: np.ndarray) -> None:
        """
        Set the geometry of the scene. This is useful to raycast the same
        scene for multiple lidar poses.
        """
        self.gas_handle, self.d_gas_output_buffer = self._create_accel(
            self.ctx,
            np_vertices=vertices,
            np_triangles=triangles,
        )

        (
            self.d_raygen_sbt,
            self.d_miss_sbt,
            self.d_hitgroup_sbt,
            self.sbt,
        ) = self._create_sbt(self.prog_groups)

    def rays_intersect_mesh(
        self,
        rays: np.ndarray,
        mesh: o3d.geometry.TriangleMesh,
    ):
        """
        Intersect the mesh with the given rays.

        Args:
            rays: (N, 6) float32 numpy array
            mesh: o3d.geometry.TriangleMesh

        Returns:
            points: (N, 3) float32 numpy array
        """
        # Sanity checks.
        if not isinstance(rays, np.ndarray):
            raise TypeError("rays must be a numpy array.")
        if rays.ndim != 2 or rays.shape[1] != 6:
            raise ValueError("rays must be a (N, 6) array.")

        raise NotImplementedError()

    def lidar_intersect_mesh(
        self,
        lidar: Lidar,
        mesh: o3d.geometry.TriangleMesh,
    ):
        """
        Intersect the mesh with the lidar rays.

        Args:
            lidar: Lidar
            mesh: o3d.geometry.TriangleMesh

        Returns:
            points: (N, 3) float32 numpy array
            incident_angles: (N,) float32 numpy array
        """
        self.set_geometry(
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            triangles=np.asarray(mesh.triangles, dtype=np.int32),
        )

        # Copy rays to device.
        # This can be done more efficiently by calculating rays on GPU.
        rays = lidar.get_rays()
        rays_o = rays[:, :3]
        rays_d = rays[:, 3:]
        d_rays_o = cp.array(rays_o, dtype="f4")
        d_rays_d = cp.array(rays_d, dtype="f4")

        # Allocate output buffer.
        # This can be done more efficiently by reusing the same buffer.
        lidar_width = lidar.intrinsics.horizontal_res
        lidar_height = lidar.intrinsics.vertical_res
        d_hits = cp.empty((lidar_width, lidar_height, 3), dtype="f4")
        d_incident_angles = cp.empty((lidar_width, lidar_height), dtype="f4")

        # Prepare params.
        params = [
            ("u4", "width", lidar_width),  # uint32_t int
            ("u4", "height", lidar_height),  # uint32_t int
            ("u8", "hits", d_hits.data.ptr),  # uint64_t pointer
            ("u8", "incident_angles", d_incident_angles.data.ptr),  # uint64_t pointer
            ("u8", "rays_o", d_rays_o.data.ptr),  # uint64_t pointer
            ("u8", "rays_d", d_rays_d.data.ptr),  # uint64_t pointer
            ("u8", "gas_handle", self.gas_handle),  # uint64_t pointer
        ]
        formats = [x[0] for x in params]
        names = [x[1] for x in params]
        values = [x[2] for x in params]
        itemsize = RaycastEngineGPU._get_aligned_itemsize(formats, 8)
        params_dtype = np.dtype(
            {
                "names": names,
                "formats": formats,
                "itemsize": itemsize,
                "align": True,
            }
        )
        h_params = np.array([tuple(values)], dtype=params_dtype)
        d_params = RaycastEngineGPU._array_to_device_memory(h_params)

        # Launch!
        stream = cp.cuda.Stream()
        optix.launch(
            self.pipeline,
            stream.ptr,
            d_params.ptr,
            h_params.dtype.itemsize,
            self.sbt,
            lidar_width,
            lidar_height,
            1,  # depth
        )
        stream.synchronize()
        h_hits = cp.asnumpy(d_hits)
        h_incident_angles = cp.asnumpy(d_incident_angles)

        # Filter out missed points.
        h_hits = h_hits.reshape((lidar_height * lidar_width, 3))
        h_incident_angles = h_incident_angles.reshape((lidar_height * lidar_width,))
        im_render_mask = np.isfinite(h_hits[:, 0])
        points = h_hits[im_render_mask]
        incident_angles = h_incident_angles[im_render_mask]

        # Filter out out-of-range points.
        max_range = lidar.intrinsics.max_range
        lidar_center = ct.convert.pose_to_C(lidar.pose)
        point_dists = np.linalg.norm(points - lidar_center, axis=1)
        points = points[point_dists < max_range]
        incident_angles = incident_angles[point_dists < max_range]

        assert len(points) == len(incident_angles)

        return points, incident_angles


def main():
    # Load test data.
    script_dir = Path(__file__).parent.absolute().resolve()
    lit_root = script_dir.parent.parent
    data_dir = lit_root / "data"
    raycast_data_path = data_dir / "test_data" / "raycast_data.pkl"
    raycast_mesh_path = data_dir / "test_data" / "raycast_mesh.ply"

    # Read lidar data.
    with open(raycast_data_path, "rb") as f:
        raycast_data = pickle.load(f)
    lidar = Lidar(
        intrinsics=raycast_data["lidar_intrinsics"],
        pose=raycast_data["pose"],
    )

    # Read mesh.
    mesh = o3d.io.read_triangle_mesh(str(raycast_mesh_path))
    mesh.compute_vertex_normals()

    # Ray cast GPU.
    raycast_engine_gpu = RaycastEngineGPU()
    points_gpu = raycast_engine_gpu.lidar_intersect_mesh(lidar=lidar, mesh=mesh)

    # Ray cast CPU for comparison.
    raycast_engine_cpu = RaycastEngineCPU()
    points_cpu = raycast_engine_cpu.lidar_intersect_mesh(lidar=lidar, mesh=mesh)
    is_all_close = np.allclose(points_cpu, points_gpu, rtol=1e-03, atol=1e-03)

    print(f"len(points_cpu) = {len(points_cpu)}, len(points_gpu) = {len(points_gpu)}")
    print(f"np.allclose(points_cpu, points_gpu) = {is_all_close}")

    # Visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_gpu)
    o3d.visualization.draw_geometries([pcd, mesh])


if __name__ == "__main__":
    main()
