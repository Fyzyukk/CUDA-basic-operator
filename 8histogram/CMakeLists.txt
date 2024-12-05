cmake_minimum_required(VERSION 3.29)
project(8histogram LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(histogram_v1
    8-1.histogram.cu
)

add_executable(histogram_v2
    8-2.histogram.cu
)
