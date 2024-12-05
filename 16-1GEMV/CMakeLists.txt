cmake_minimum_required(VERSION 3.29)
project(16-1GEMV LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

include_directories(.)

#16 gemv-1
add_executable(MatVec
    16.MatVec.cu
)
target_link_libraries(
    MatVec PUBLIC
    -lcudart
    -lcudadevrt)

#16 gemv-2
add_executable(VecMat
    16.VecMat.cu
)
target_link_libraries(
    VecMat PUBLIC
    -lcudart
    -lcudadevrt)