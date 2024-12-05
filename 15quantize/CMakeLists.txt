cmake_minimum_required(VERSION 3.29)
project(15quantize LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(quantize
    15.quantize.cu
)

add_executable(quantize_pertensor
    15-1.quantize_pertensor.cu
)

add_executable(quantize_perchannel
    15-2.quantize_perchannel.cu
)