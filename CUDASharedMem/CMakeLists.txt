cmake_minimum_required(VERSION 3.17)
project(CUDASharedMem CUDA)

set(CMAKE_CXX_STANDARD 14)

# Set the location of the OpenCV directory
set(OpenCV_DIR "C:/DevelopmentWorkspace/SDKs/opencv/build")

find_package(OpenCV REQUIRED) # after project()

# Find OpenCV library
find_package(OpenCV 4 REQUIRED)

include_directories(./inc)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(CUDASharedMem
        src/contrast.cu
        src/utils.cu
        src/main.cu)

# Link OpenCV library
target_link_libraries(CUDASharedMem ${OpenCV_LIBS})

set_target_properties(CUDASharedMem
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
