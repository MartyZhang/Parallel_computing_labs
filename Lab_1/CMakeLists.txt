cmake_minimum_required(VERSION 3.6)
project(convolve)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -std=c11")

set(SOURCE_FILES lodepng.h lodepng.c wm.h helpers.h helpers.c convolutionmax.c)
add_executable(convolve ${SOURCE_FILES})