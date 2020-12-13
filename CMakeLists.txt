cmake_minimum_required(VERSION 3.7)

project(HistogramEqualization)

add_subdirectory("${PROJECT_SOURCE_DIR}/CPU" "${PROJECT_SOURCE_DIR}/CPU")
add_subdirectory("${PROJECT_SOURCE_DIR}/CUDANaive" "${PROJECT_SOURCE_DIR}/CUDANaive")
add_subdirectory("${PROJECT_SOURCE_DIR}/CUDAOptimized" "${PROJECT_SOURCE_DIR}/CUDAOptimized")