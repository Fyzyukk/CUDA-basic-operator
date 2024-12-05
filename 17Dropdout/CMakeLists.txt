cmake_minimum_required(VERSION 3.29)
project(17Dropout LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(dropout
    17.dropout.cu
)