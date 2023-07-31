cmake_minimum_required(VERSION 3.0.2)
project(vo)

set (CMAKE_CXX_FLAGS "-lstdc++fs -std=c++17")

find_package(OpenCV 4.6.0 REQUIRED)
find_package(Eigen3 3.3.7 REQUIRED)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

message(STATUS "OpenCV path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Eigen3 path: ${EIGEN3_INCLUDE_DIR}")

add_executable(
    feature_detection
    src/feature_detection.cpp
)

target_link_libraries(
    feature_detection
    ${OpenCV_LIBRARIES}
)

add_executable(
    stereo3Dto3D
    src/stereo3Dto3D.cpp
    src/triangulate.cpp
)

target_link_libraries(
    stereo3Dto3D
    ${OpenCV_LIBRARIES}
)