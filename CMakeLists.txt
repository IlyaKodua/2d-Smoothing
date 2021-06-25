cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -g -G")

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(interp2 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(interp2 example.cu interp2.cuh)
set_target_properties(
		interp2
		PROPERTIES
		CUDA_SEPARABLE_COMPILATION ON)
