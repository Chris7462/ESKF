cmake_minimum_required(VERSION 3.16.1)

project(ESKFLie)

find_package(Eigen3 REQUIRED)
find_package(manif REQUIRED)

add_executable(se2_localization_liekf
  src/se2_localization_liekf.cpp)

target_include_directories(se2_localization_liekf PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
  ${manif_INCLUDE_DIRS})
