cmake_minimum_required(VERSION 3.29)
project(9copy_if LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(copy-if
    9.copy-if.cu
)
