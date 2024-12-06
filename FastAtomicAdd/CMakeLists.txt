cmake_minimum_required(VERSION 3.29)
project(Elementwise LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

include_directories(.)

# v0
add_executable(fastatomicadd_baseline
    fastatomicadd_baseline.cu
)

# v1
add_executable(fastatomicadd_v1
    fastatomicadd_v1.cu
)
