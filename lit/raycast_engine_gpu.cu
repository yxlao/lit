//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <math_constants.h>
#include <optix.h>

#include "vector_math.h"

struct Params {
    unsigned int width;
    unsigned int height;
    float3* hits;            // (W, H, 3), hit coordinates
    float* incident_angles;  // (W, H), incident angle
    float3* rays_o;
    float3* rays_d;
    OptixTraversableHandle gas_handle;
};

struct RayGenData {
    // No data needed
};

struct MissData {
    float3 bg_color;
};

struct HitGroupData {
    float3* vertex_buffer;
    int3* triangle_buffer;
};

extern "C" {
__constant__ Params params;
}

static __forceinline__ __device__ void computeRay(uint3 idx,
                                                  uint3 dim,
                                                  float3& origin,
                                                  float3& direction) {
    origin = params.rays_o[idx.y * params.width + idx.x];
    direction = params.rays_d[idx.y * params.width + idx.x];
}

extern "C" __global__ void __raygen__rg() {
    // Lookup our location within the launch grid.
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen
    float3 ray_origin, ray_direction;
    computeRay(make_uint3(idx.x, idx.y, 0), dim, ray_origin, ray_direction);

    // Trace the ray against our scene hierarchy
    unsigned int p0;
    unsigned int p1;
    unsigned int p2;
    unsigned int p3;
    optixTrace(params.gas_handle,         // Traversable handle
               ray_origin,                // float3
               ray_direction,             // float3
               0.0f,                      // Min intersection distance
               1e16f,                     // Max intersection distance
               0.0f,                      // rayTime -- used for motion blur
               OptixVisibilityMask(255),  // Specify always visible
               OPTIX_RAY_FLAG_NONE,
               0,   // SBT offset   -- See SBT discussion
               1,   // SBT stride   -- See SBT discussion
               0,   // missSBTIndex -- See SBT discussion
               p0,  // optixSetPayload_0, returned from hit or miss kernel
               p1,  // optixSetPayload_1, returned from hit or miss kernel
               p2,  // optixSetPayload_2, returned from hit or miss kernel
               p3   // optixSetPayload_3, returned from hit or miss kernel
    );

    // Convert the ray cast result values back to floats.
    float3 hit = make_float3(0);
    hit.x = int_as_float(p0);
    hit.y = int_as_float(p1);
    hit.z = int_as_float(p2);
    float incident_angle = int_as_float(p3);

    // Record results in our output raster
    params.hits[idx.y * params.width + idx.x] = hit;
    params.incident_angles[idx.y * params.width + idx.x] = incident_angle;
}

extern "C" __global__ void __miss__ms() {
    MissData* miss_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    // https://stackoverflow.com/a/15514595/1255535
    optixSetPayload_0(float_as_int(CUDART_INF_F));
    optixSetPayload_1(float_as_int(CUDART_INF_F));
    optixSetPayload_2(float_as_int(CUDART_INF_F));
    optixSetPayload_3(float_as_int(0));
}

extern "C" __global__ void __closesthit__ch() {
    // Compute intersection point coordinates.
    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_direction = optixGetWorldRayDirection();

    // Get the hit distance.
    const float t = optixGetRayTmax();

    // Compute the intersection point.
    const float3 p = ray_origin + t * ray_direction;

    // Get the SBT data pointer and cast to the proper type.
    const HitGroupData* sbt_data =
            reinterpret_cast<const HitGroupData*>(optixGetSbtDataPointer());

    // Cast the integer back to a pointer.
    float3* vertex_buffer = reinterpret_cast<float3*>(sbt_data->vertex_buffer);
    int3* triangle_buffer = reinterpret_cast<int3*>(sbt_data->triangle_buffer);

    // Retrieve the index of the hit triangle.
    const unsigned int primitive_index = optixGetPrimitiveIndex();

    // Access the indices of the vertices that form this triangle.
    int3 vertex_indices = triangle_buffer[primitive_index];

    // Access vertex data.
    const float3 v0 = vertex_buffer[vertex_indices.x];
    const float3 v1 = vertex_buffer[vertex_indices.y];
    const float3 v2 = vertex_buffer[vertex_indices.z];

    // Compute normal.
    const float3 edge0 = v1 - v0;
    const float3 edge1 = v2 - v0;
    const float3 normal = normalize(cross(edge0, edge1));

    // Normalize the ray direction.
    const float3 normalized_ray_direction = normalize(ray_direction);

    // Compute the dot product of the normalized ray direction and the
    // normalized normal.
    const float dot_product = dot(normalized_ray_direction, normal);

    // Compute the incident angle using arccos.
    // Clamping the dot product to the range [-1, 1] to avoid numerical issues
    const float incident_angle = acosf(fmaxf(-1.0f, fminf(dot_product, 1.0f)));

    // Set payloads.
    optixSetPayload_0(float_as_int(p.x));
    optixSetPayload_1(float_as_int(p.y));
    optixSetPayload_2(float_as_int(p.z));
    optixSetPayload_3(float_as_int(incident_angle));
}
