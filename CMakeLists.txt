cmake_minimum_required(VERSION 3.0.2)
project(vo)

set (CMAKE_CXX_FLAGS "-lstdc++fs -std=c++17")

set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV 4.6.0 REQUIRED)
find_package(Eigen3 3.3.4 EXACT REQUIRED)
find_package(Ceres 2.2.0 EXACT REQUIRED)
find_package(Sophus REQUIRED)

include_directories(
    include
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/examples
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Sophus_INCLUDE_DIRS}
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

# add_executable(
#     stereo3Dto3D
#     src/stereo3Dto3D.cpp
#     src/triangulate.cpp
# )

# target_link_libraries(
#     stereo3Dto3D
#     ${OpenCV_LIBRARIES}
#     ${EIGEN3_LIBS}
# )


add_executable(
    orb_feature_match
    examples/orb_feature_matching.cpp 
)

target_link_libraries(
    orb_feature_match
    ${OpenCV_LIBRARIES}
)

add_executable(
    curve_fitting_ceres 
    examples/curve_fitting_ceres.cpp
)
target_link_libraries(
    curve_fitting_ceres 
    ${CERES_LIBRARIES} 
    ${EIGEN3_LIBS}
)
# add_dependencies(gsw_ceres ${bspline_traj_planner_EXPORTED_TARGETS})

add_executable(
    bundle_adjuster_ceres 
    examples/bundle_adjuster_ceres.cpp
)
target_link_libraries(
    bundle_adjuster_ceres 
    ${CERES_LIBRARIES} 
    ${EIGEN3_LIBS}
)

# add_executable(
#     bundle_adjuster_on_manifold
#     examples/bundle_adjuster_on_manifold.cc
# )
# target_link_libraries(
#     bundle_adjuster_on_manifold 
#     ${CERES_LIBRARIES} 
#     ${EIGEN3_LIBS}
# )

add_executable(
    sophus_basics
    examples/sophus_basics.cc
)

target_link_libraries(
    sophus_basics 
    ${EIGEN3_LIBS}
    Sophus::Sophus
)