cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
Project(CUDA_Freshman CXX C CUDA)
set(CMAKE_CUDA_FLAGS "-arch=compute_52 -g -G -O3")
include_directories(./include)

add_subdirectory(helloworld)
