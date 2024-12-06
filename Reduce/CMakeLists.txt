cmake_minimum_required(VERSION 3.29)
project(Elementwise LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

include_directories(.)

# v0
add_executable(reduce_baseline
    reduce_baseline.cu
)

# v1
add_executable(reduce_v1
    reduce_v1_interleaved_addressing.cu
)

# v2
add_executable(reduce_v2
    reduce_v2_bank_conflict.cu
)

# v3
add_executable(reduce_v3
    reduce_v3_Idel_thread.cu
)

# v4
add_executable(reduce_v4
    reduce_v4_unrolling_the_last_warp.cu
)

# v5
add_executable(reduce_v5
    reduce_v5_complete_unrolling.cu
)

# v6.0
add_executable(reduce_v6.0
    reduce_v6.0_grid&block_size.cu
)

# v6.1
add_executable(reduce_v6.1
    reduce_v6.1_grid&block_size.cu
)

# v6.2
add_executable(reduce_v6.2
    reduce_v6.2_grid&block_size.cu
)

# v7
add_executable(reduce_v7
    reduce_v7_shfl_down_sync.cu
)