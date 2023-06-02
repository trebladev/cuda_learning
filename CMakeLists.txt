cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(CUDA_Freshman CXX C CUDA)
set(CMAKE_CUDA_FLAGS "-arch=compute_52 -g -G -O3")
include_directories(./include)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_subdirectory(helloworld)
add_subdirectory(check)
add_subdirectory(sum_array)
