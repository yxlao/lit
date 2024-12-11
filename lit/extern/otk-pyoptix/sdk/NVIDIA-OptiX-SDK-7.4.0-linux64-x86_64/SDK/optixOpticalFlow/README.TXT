This sample demonstrates calculation of optical flow between two images.
Example: optixOpticalFlow \
         ../optixDenoiser/motiondata/soane-BSDF-001.exr \
         ../optixDenoiser/motiondata/soane-BSDF-002.exr flow002.exr
The flow vectors can be used for the OptiX denoiser in temporal mode.
The sample runs only on Ampere and newer GPUs.
