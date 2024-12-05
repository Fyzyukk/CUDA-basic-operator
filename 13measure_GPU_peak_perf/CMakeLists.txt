cmake_minimum_required(VERSION 3.29)
project(13measure_GPU_peak_perf LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(measure_GPU_peak_perf
    13.measure_GPU_peak_perf.cu
)
add_executable(measure_GPU_peak_perf-1
        13-1.measure_GPU_peak_perf.cu
)
add_executable(measure_GPU_peak_perf-2
        13-2.measure_GPU_peak_perf.cu
)
