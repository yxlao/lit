import ctypes  # C interop helpers
import os
import pickle
import time
from pathlib import Path

import camtools as ct
import cupy as cp  # CUDA bindings
import numpy as np  # Packing of structures in C-compatible format
import open3d as o3d
import optix
import path_util
from PIL import Image, ImageOps  # Image IO
from pynvrtc.compiler import Program


class Logger:
    def __init__(self):
        self.num_mssgs = 0

    def __call__(self, level, tag, mssg):
        print("[{:>2}][{:>12}]: {}".format(level, tag, mssg))
        self.num_mssgs += 1


class OptixEngine:
    """
    Wrapper to store "global" variables and provide a better API.
    """

    def __init__(self) -> None:
        self.logger = Logger()
        self.ctx = self.create_ctx()
        self.pipeline_options = OptixEngine.set_pipeline_options()

        # Compile static code.
        triangle_cu = os.path.join(os.path.dirname(__file__), "triangle.cu")
        triangle_ptx = OptixEngine.compile_cuda(triangle_cu)
        self.module = self.create_module(self.ctx, self.pipeline_options, triangle_ptx)

        # State for geometry.
        self.is_geometry_set = False

        # Sensor info.
        self.width = None
        self.height = None

    def set_sensor(self, width, height):
        self.width = width
        self.height = height

    def set_geometry(self, vertices: np.ndarray, triangles: np.ndarray) -> None:
        self.gas_handle, self.d_gas_output_buffer = self.create_accel(
            self.ctx,
            np_vertices=vertices,
            np_triangles=triangles,
        )

        self.prog_groups = self.create_program_groups(self.ctx, self.module)
        self.pipeline = self.create_pipeline(
            self.ctx,
            self.prog_groups,
            self.pipeline_options,
        )
        (
            self.d_raygen_sbt,
            self.d_miss_sbt,
            self.d_hitgroup_sbt,
            self.sbt,
        ) = self.create_sbt(self.prog_groups)

        self.is_geometry_set = True

    def launch(self, rays_o, rays_d):
        """
        rays_o: (N, 3) array of ray origins.
        rays_d: (N, 3) array of ray directions.
        """

        if not self.is_geometry_set:
            raise RuntimeError("Mesh is not set. Call set_geometry() first.")
        print("Launching ... ")

        h_im = np.zeros((self.width, self.height, 3), np.float32)
        d_im = cp.array(h_im)
        d_rays_o = cp.array(rays_o, dtype="f4")
        d_rays_d = cp.array(rays_d, dtype="f4")

        params = [
            ("u4", "width", self.width),  # uint32_t int
            ("u4", "height", self.height),  # uint32_t int
            ("u8", "image", d_im.data.ptr),  # uint64_t pointer
            ("u8", "rays_o", d_rays_o.data.ptr),  # uint64_t pointer
            ("u8", "rays_d", d_rays_d.data.ptr),  # uint64_t pointer
            ("u8", "trav_handle", self.gas_handle),  # uint64_t pointer
        ]

        formats = [x[0] for x in params]
        names = [x[1] for x in params]
        values = [x[2] for x in params]
        itemsize = OptixEngine.get_aligned_itemsize(formats, 8)
        params_dtype = np.dtype(
            {"names": names, "formats": formats, "itemsize": itemsize, "align": True}
        )
        h_params = np.array([tuple(values)], dtype=params_dtype)
        d_params = OptixEngine.array_to_device_memory(h_params)

        stream = cp.cuda.Stream()
        optix.launch(
            self.pipeline,
            stream.ptr,
            d_params.ptr,
            h_params.dtype.itemsize,
            self.sbt,
            self.width,
            self.height,
            1,  # depth
        )

        stream.synchronize()

        h_im = cp.asnumpy(d_im)
        return h_im

    def create_ctx(self):
        print("Creating optix device context ...")

        # OptiX param struct fields can be set with optional
        # keyword constructor arguments.
        ctx_options = optix.DeviceContextOptions(
            logCallbackFunction=self.logger, logCallbackLevel=4
        )

        # They can also be set and queried as properties on the struct
        if optix.version()[1] >= 2:
            ctx_options.validationMode = optix.DEVICE_CONTEXT_VALIDATION_MODE_ALL

        cu_ctx = 0
        return optix.deviceContextCreate(cu_ctx, ctx_options)

    def create_accel(self, ctx, np_vertices, np_triangles):
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

        np_vertices = np.ascontiguousarray(np_vertices.flatten())
        np_triangles = np.ascontiguousarray(np_triangles.flatten())

        vertices = cp.array(np_vertices, dtype="f4")
        indices = cp.array(np_triangles, dtype="u4")

        triangle_input_flags = [optix.GEOMETRY_FLAG_NONE]  # One flag is sufficient
        triangle_input = optix.BuildInputTriangleArray()
        triangle_input.vertexFormat = optix.VERTEX_FORMAT_FLOAT3
        triangle_input.numVertices = len(vertices) // 3
        triangle_input.vertexBuffers = [vertices.data.ptr]
        triangle_input.indexFormat = optix.INDICES_FORMAT_UNSIGNED_INT3
        triangle_input.numIndexTriplets = len(indices) // 3
        triangle_input.indexBuffer = indices.data.ptr
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

    def create_module(self, ctx, pipeline_options, triangle_ptx):
        print("Creating optix module ...")

        module_options = optix.ModuleCompileOptions(
            maxRegisterCount=optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            optLevel=optix.COMPILE_OPTIMIZATION_DEFAULT,
            debugLevel=optix.COMPILE_DEBUG_LEVEL_DEFAULT,
        )

        module, log = ctx.moduleCreateFromPTX(
            module_options, pipeline_options, triangle_ptx
        )
        print("\tModule create log: <<<{}>>>".format(log))
        return module

    def create_program_groups(self, ctx, module):
        print("Creating program groups ... ")

        raygen_prog_group_desc = optix.ProgramGroupDesc()
        raygen_prog_group_desc.raygenModule = module
        raygen_prog_group_desc.raygenEntryFunctionName = "__raygen__rg"
        raygen_prog_group, log = ctx.programGroupCreate([raygen_prog_group_desc])
        print("\tProgramGroup raygen create log: <<<{}>>>".format(log))

        miss_prog_group_desc = optix.ProgramGroupDesc()
        miss_prog_group_desc.missModule = module
        miss_prog_group_desc.missEntryFunctionName = "__miss__ms"
        miss_prog_group, log = ctx.programGroupCreate([miss_prog_group_desc])
        print("\tProgramGroup miss create log: <<<{}>>>".format(log))

        hitgroup_prog_group_desc = optix.ProgramGroupDesc()
        hitgroup_prog_group_desc.hitgroupModuleCH = module
        hitgroup_prog_group_desc.hitgroupEntryFunctionNameCH = "__closesthit__ch"
        hitgroup_prog_group, log = ctx.programGroupCreate([hitgroup_prog_group_desc])
        print("\tProgramGroup hitgroup create log: <<<{}>>>".format(log))

        return [raygen_prog_group[0], miss_prog_group[0], hitgroup_prog_group[0]]

    def create_pipeline(self, ctx, program_groups, pipeline_compile_options):
        print("Creating pipeline ... ")

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
            stack_sizes, max_trace_depth, 0, 0  # maxCCDepth  # maxDCDepth
        )

        pipeline.setStackSize(
            dc_stack_size_from_trav,
            dc_stack_size_from_state,
            cc_stack_size,
            1,  # maxTraversableDepth
        )

        return pipeline

    def create_sbt(self, prog_groups):
        print("Creating sbt ... ")

        (raygen_prog_group, miss_prog_group, hitgroup_prog_group) = prog_groups

        header_format = "{}B".format(optix.SBT_RECORD_HEADER_SIZE)

        # Raygen record
        formats = [header_format]
        itemsize = OptixEngine.get_aligned_itemsize(formats, optix.SBT_RECORD_ALIGNMENT)
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
        d_raygen_sbt = OptixEngine.array_to_device_memory(h_raygen_sbt)

        # Miss record
        formats = [header_format, "f4", "f4", "f4"]
        itemsize = OptixEngine.get_aligned_itemsize(formats, optix.SBT_RECORD_ALIGNMENT)
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
        d_miss_sbt = OptixEngine.array_to_device_memory(h_miss_sbt)

        # Hitgroup record
        formats = [header_format]
        itemsize = OptixEngine.get_aligned_itemsize(formats, optix.SBT_RECORD_ALIGNMENT)
        dtype = np.dtype(
            {
                "names": ["header"],
                "formats": formats,
                "itemsize": itemsize,
                "align": True,
            }
        )
        h_hitgroup_sbt = np.array([(0)], dtype=dtype)
        optix.sbtRecordPackHeader(hitgroup_prog_group, h_hitgroup_sbt)
        d_hitgroup_sbt = OptixEngine.array_to_device_memory(h_hitgroup_sbt)

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
    def set_pipeline_options():
        if optix.version()[1] >= 2:
            return optix.PipelineCompileOptions(
                usesMotionBlur=False,
                traversableGraphFlags=int(
                    optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS
                ),
                numPayloadValues=3,
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
                numPayloadValues=3,
                numAttributeValues=3,
                exceptionFlags=int(optix.EXCEPTION_FLAG_NONE),
                pipelineLaunchParamsVariableName="params",
            )

    @staticmethod
    def get_aligned_itemsize(formats, alignment):
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
    def optix_version_gte(version):
        if optix.version()[0] > version[0]:
            return True
        if optix.version()[0] == version[0] and optix.version()[1] >= version[1]:
            return True
        return False

    @staticmethod
    def array_to_device_memory(numpy_array, stream=cp.cuda.Stream()):
        byte_size = numpy_array.size * numpy_array.dtype.itemsize

        h_ptr = ctypes.c_void_p(numpy_array.ctypes.data)
        d_mem = cp.cuda.memory.alloc(byte_size)
        d_mem.copy_from_async(h_ptr, byte_size, stream)
        return d_mem

    @staticmethod
    def compile_cuda(cuda_file):
        with open(cuda_file, "rb") as f:
            src = f.read()
        nvrtc_dll = os.environ.get("NVRTC_DLL")
        if nvrtc_dll is None:
            nvrtc_dll = ""
        print("NVRTC_DLL = {}".format(nvrtc_dll))
        prog = Program(src.decode(), cuda_file, lib_name=nvrtc_dll)
        compile_options = [
            "-use_fast_math",
            "-lineinfo",
            "-default-device",
            "-std=c++11",
            "-rdc",
            "true",
            #'-IC:\\Program Files\\NVIDIA GPU Computing Toolkit\CUDA\\v11.1\include'
            f"-I{path_util.cuda_tk_path}",
            f"-I{path_util.include_path}",
        ]
        # Optix 7.0 compiles need path to system stddef.h
        # the value of optix.stddef_path is compiled in constant. When building
        # the module, the value can be specified via an environment variable, e.g.
        #   export PYOPTIX_STDDEF_DIR="/usr/include/linux"
        if optix.version()[1] == 0:
            compile_options.append(f"-I{path_util.stddef_path}")

        ptx = prog.compile(compile_options)
        return ptx


def main():
    vertices = np.array(
        [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
            [0.0, -0.5, 0.0],
        ],
        dtype=np.float32,
    )
    triangles = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
        ],
        dtype=np.int32,
    )

    cube = o3d.geometry.TriangleMesh.create_box(width=0.25, height=0.25, depth=0.25)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
    mesh = cube + sphere

    vertices = np.array(mesh.vertices, dtype=np.float32)
    triangles = np.array(mesh.triangles, dtype=np.int32)

    # Read real-world mesh. ################################################
    script_dir = Path(__file__).parent.absolute().resolve()
    pyoptix_root = script_dir.parent
    data_dir = pyoptix_root / "data"
    raycast_data_path = data_dir / "raycast_data.pkl"
    raycast_mesh_path = data_dir / "raycast_mesh.ply"

    with open(raycast_data_path, "rb") as f:
        raycast_data = pickle.load(f)
    raycast_mesh = o3d.io.read_triangle_mesh(str(raycast_mesh_path))
    raycast_mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([raycast_mesh])

    rays_o = raycast_data["rays"][:, :3]
    rays_d = raycast_data["rays"][:, 3:]
    width = raycast_data["lidar_intrinsics"].horizontal_res
    height = raycast_data["lidar_intrinsics"].vertical_res
    assert len(rays_o) == len(rays_d) == width * height

    # use_real_mesh = True
    # if use_real_mesh:
    triangles = np.array(raycast_mesh.triangles, dtype=np.int32)
    vertices = np.array(raycast_mesh.vertices, dtype=np.float32)
    ########################################################################

    # Create engine and set mesh.
    oe = OptixEngine()
    start_time = time.time()
    oe.set_sensor(width=width, height=height)
    oe.set_geometry(vertices, triangles)
    im_render = oe.launch(rays_o=rays_o, rays_d=rays_d)
    print(f"Ray casting took {time.time() - start_time:.5f} seconds.")

    # Plot im_render as a point cloud. Invalid values are +inf.
    im_render = im_render.reshape((oe.height * oe.width, 3))
    im_render_mask = np.isfinite(im_render[:, 0])
    points = im_render[im_render_mask]

    max_range = 70
    lidar_center = rays_o[0]
    point_dists = np.linalg.norm(points - lidar_center, axis=1)
    points = points[point_dists < max_range]

    # Draw rays as lineset.
    # Points: all of rays_o and rays_d
    # Lines: all of rays_o -> rays_d
    ls = o3d.geometry.LineSet()
    ls_points = np.vstack((rays_o, rays_o + rays_d))
    num_lines = len(rays_o)
    ls_lines = np.vstack((np.arange(num_lines), np.arange(num_lines) + num_lines)).T
    ls.points = o3d.utility.Vector3dVector(ls_points)
    ls.lines = o3d.utility.Vector2iVector(ls_lines)

    # Remove points with coordinates smaller tha 5.
    # points = points[np.abs(points[:, 0]) > 5]

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw_geometries([pcd, raycast_mesh, ls])

    # # PIL expects [ y, x ] resolution
    # im_render = im_render.reshape((height, width, 3))
    # im_render = (im_render * 255).astype(np.uint8)
    # # PIL expects y = 0 at bottom
    # img = ImageOps.flip(Image.fromarray(im_render, "RGB"))
    # img.show()
    # img.save("triangle.png")


if __name__ == "__main__":
    main()
