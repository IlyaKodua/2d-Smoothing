cmake_minimum_required(VERSION 3.10)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options -fPIC \
-gencode arch=compute_50,code=sm_50 \
-gencode arch=compute_52,code=sm_52 \
-gencode arch=compute_60,code=sm_60 \
-gencode arch=compute_60,code=compute_60 \
-gencode=arch=compute_61,code=sm_61 \
-gencode=arch=compute_62,code=sm_62")

if (CMAKE_BUILD_TYPE STREQUAL Debug)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options -g -G")
endif()

project(Conv2d CUDA)

#find_package( OpenCV REQUIRED )
#include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CUDA_STANDARD 14)
#target_link_libraries( Conv2d ${OpenCV_LIBS} )
add_executable(Conv2d example.cu conv2d.cuh)
set_target_properties(
		Conv2d
		PROPERTIES
		CUDA_SEPARABLE_COMPILATION ON)
#		${OpenCV_LIBS} )
