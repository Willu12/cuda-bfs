cmake_minimum_required(VERSION 3.21)
project(cuda_bfs CUDA)

set(CMAKE_CUDA_STANDARD 23)

add_executable(cuda_bfs main.cu)

set_target_properties(cuda_bfs PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
