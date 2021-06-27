cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -g -G")


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
