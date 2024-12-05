cmake_minimum_required(VERSION 3.29)
project(12softmax LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

include_directories(.)



# 12
add_executable(softmax_v0
    softmax_v0.cu
)

# 12
add_executable(softmax_v1
    softmax_v1.cu
)

# 12
add_executable(softmax_v2
    softmax_v2.cu
)

# 12
add_executable(softmax_v3
    softmax_v3.cu
)