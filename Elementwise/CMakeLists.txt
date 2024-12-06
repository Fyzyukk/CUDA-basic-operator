cmake_minimum_required(VERSION 3.29)
project(Elementwise LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

include_directories(.)

# v0
add_executable(elementwise_relu_baseline
    elementwise_relu_baseline.cu
)

# v1
add_executable(elementwise_relu_v1
    elementwise_relu_v1.cu
)

# v2
add_executable(elementwise_relu_v2
    elementwise_relu_v2.cu
)

# v3
add_executable(elementwise_relu_v3
    elementwise_relu_v3.cu
)

# v0
add_executable(elementwise_dotmul_baseline
    elementwise_dotmul_baseline.cu
        elementwise_baseline.cu
        elementwise_dotmul_baseline_v1.cu
)

# v1
add_executable(elementwise_dotmul_v1
    elementwise_dotmul_v1.cu
)