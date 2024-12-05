cmake_minimum_required(VERSION 3.29)
project(hellocuda LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

#find_package(CUDA REQUIRED)

add_executable(hellocuda
        1.hellocuda.cu
)

add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-finput-charset=UTF-8>")

#add_library(hellocuda
#        1.hellocuda.cu
#
#)

#file(GLOB plugin "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64/*.lib")
#target_link_libraries(LayerNormPlugin
#        ${plugin}
#)

# 忽略冲突的默认库
#target_link_options(1.hellocuda PRIVATE "/NODEFAULTLIB:LIBCMT")