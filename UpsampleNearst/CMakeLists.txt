cmake_minimum_required(VERSION 3.29)
project(Elementwise LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

include_directories(.)

# v1
add_executable(upsamplenearest_v1
    upsamplenearest_v1.cu
)

# v2
add_executable(upsamplenearest_v2
    upsamplenearest_v2.cu
)

# v3
add_executable(upsamplenearest_v3
    upsamplenearest_v3.cu
)

# v3
add_executable(upsamplenearest_v4
    upsamplenearest_v4.cu
)