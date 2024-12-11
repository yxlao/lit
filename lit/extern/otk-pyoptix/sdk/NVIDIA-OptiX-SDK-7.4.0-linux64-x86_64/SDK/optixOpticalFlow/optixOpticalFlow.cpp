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

#include <optix.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

extern "C" OptixResult runOpticalFlow( CUcontext, CUstream, OptixImage2D& flow, const OptixImage2D images[2], float& flowTime, std::string & errMessage );

//------------------------------------------------------------------------------
//
//  optixOpticalFlow -- Demonstration of OptiX optical flow
//
//------------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0 )
{
    std::cerr << "Usage  : " << argv0 << " frame1.exr frame2.exr flow.exr\n";
    std::cerr << "Calculate flow vectors between the two images, write vectors to the third file\n";
    exit( 1 );
}

// Create float OptixImage2D with given dimension and channel count. Allocate memory on device and
// copy data from host memory given in hmem to device if hmem is nonzero.

static OptixImage2D createOptixImage2D( unsigned int width, unsigned int height, unsigned int nChannels, const float* hmem = nullptr )
{
    OptixImage2D oi;

    const uint64_t frame_byte_size = width * height * nChannels * sizeof( float );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &oi.data ), frame_byte_size ) );
    if( hmem )
    {
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( oi.data ), hmem, frame_byte_size, cudaMemcpyHostToDevice ) );
    }
    oi.width              = width;
    oi.height             = height;
    oi.rowStrideInBytes   = width * nChannels * sizeof( float );
    oi.pixelStrideInBytes = nChannels * sizeof( float );
    oi.format = nChannels == 2 ? OPTIX_PIXEL_FORMAT_FLOAT2 : nChannels == 3 ? OPTIX_PIXEL_FORMAT_FLOAT3 : OPTIX_PIXEL_FORMAT_FLOAT4;
    return oi;
}

int32_t main( int32_t argc, char** argv )
{
    if( argc < 4 )
        printUsageAndExit( argv[0] );
    try
    {
        CUcontext cuCtx = 0;
        CUDA_CHECK( (cudaError_t)cuInit( 0 ) );
        CUDA_CHECK( (cudaError_t)cuCtxCreate( &cuCtx, 0, 0 ) );
        CUstream stream = 0;

        sutil::ImageBuffer frame0 = sutil::loadImage( argv[1] );
        std::cout << "\tLoaded frame0 " << argv[1] << " (" << frame0.width << "x" << frame0.height << ")" << std::endl;

        sutil::ImageBuffer frame1 = sutil::loadImage( argv[2] );
        std::cout << "\tLoaded frame1 " << argv[2] << " (" << frame1.width << "x" << frame1.height << ")" << std::endl;

        if( frame0.width != frame1.width || frame0.height != frame1.height )
        {
            std::cerr << "Input files must have the same resolution" << std::endl;
            exit( 1 );
        }
        if( !( frame0.pixel_format == sutil::FLOAT3 || frame0.pixel_format == sutil::FLOAT4 )
            || !( frame1.pixel_format == sutil::FLOAT3 || frame1.pixel_format == sutil::FLOAT4 ) )
        {
            std::cerr << "Input files must have three or four channels" << std::endl;
            exit( 1 );
        }

        OptixImage2D images[2] = {createOptixImage2D( frame0.width, frame0.height,
                                                      frame0.pixel_format == sutil::FLOAT4 ? 4 : 3, (const float*)frame0.data ),
                                  createOptixImage2D( frame1.width, frame1.height, frame1.pixel_format == sutil::FLOAT4 ? 4 : 3,
                                                      (const float*)frame1.data )};

        // We could create a 2-channel format for flow, but sutil::ImageBuffer does not support this format.
        // The optical flow implementation will leave the third channel as-is and write only the first two.
        // A fp16 format would be sufficient for the flow vectors, for simplicity we use fp32 here.
        OptixImage2D flow = createOptixImage2D( frame0.width, frame0.height, 3 );

        // The time reported does not include initialization/destruction, only flow calculation.
        // This function has been created only for demonstration purposes. A real application would
        // construct the OptixUtilOpticalFlow object once and call OptixUtilOpticalFlow::computeFlow
        // for all frames, followed by destruction of OptixUtilOpticalFlow.
        float flowTime;
        std::string errMessage;
        OptixResult res;
        res = runOpticalFlow( cuCtx, stream, flow, images, flowTime, errMessage );
        if( res != OPTIX_SUCCESS )
            std::cerr << "Error in flow calculation: " << errMessage << std::endl;

        std::cout << "\tFlow calculation        :" << std::fixed << std::setw( 8 ) << std::setprecision( 2 ) << flowTime
                  << " ms" << std::endl;

        void * hflow;
        CUDA_CHECK( (cudaError_t)cuMemAllocHost( &hflow, flow.rowStrideInBytes * flow.height * sizeof( float ) ) );
        CUDA_CHECK( (cudaError_t)cuMemcpyDtoHAsync( hflow, flow.data, flow.rowStrideInBytes * flow.height, stream ) );
        CUDA_CHECK( (cudaError_t)cuStreamSynchronize( stream ) );

        sutil::ImageBuffer flowImage = {};
        flowImage.width              = frame0.width;
        flowImage.height             = frame0.height;
        flowImage.data               = hflow;
        flowImage.pixel_format       = sutil::FLOAT3;

        std::cout << "Saving results to '" << argv[3] << "'..." << std::endl;
        sutil::saveImage( argv[3], flowImage, false );

        CUDA_CHECK( (cudaError_t)cuMemFreeHost( hflow ) );
    }
    catch( std::exception& e )
    {
        std::cerr << "ERROR: exception caught '" << e.what() << "'" << std::endl;
    }
    return 0;
}
