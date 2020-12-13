cmake_minimum_required(VERSION 3.17)
project(CPU)

set(CMAKE_CXX_STANDARD 14)

# Set the location of the OpenCV directory
set(OpenCV_DIR "C:/DevelopmentWorkspace/SDKs/opencv/build")

find_package(OpenCV REQUIRED) # after project()

# Find OpenCV library
find_package(OpenCV 4 REQUIRED)

include_directories(inc)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(CPU
        src/contrast.cpp
        src/source.cpp src/utils.cpp)

# Link OpenCV library
target_link_libraries(CPU ${OpenCV_LIBS})