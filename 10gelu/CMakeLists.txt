cmake_minimum_required(VERSION 3.29)
project(10gelu LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(gelu_FP16_1
    10-1.gelu_FP16_1.cu
)

add_executable(gelu_FP16_2
    10-2.gelu_FP16_2.cu
)