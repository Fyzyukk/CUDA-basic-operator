cmake_minimum_required(VERSION 3.29)
project(10gelu LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

include_directories(.)


add_executable(fused_bias_mask_scale_and_add_fp32
    11-1.fused_bias_mask_scale_and_add_fp32.cu
)
