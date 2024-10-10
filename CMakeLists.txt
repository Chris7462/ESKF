cmake_minimum_required(VERSION 3.16.1)

project(ESKFLie)

find_package(Eigen3 REQUIRED)
find_package(manif REQUIRED)

# SE2 example
add_executable(se2_localization_eskfr
  src/se2_localization_eskfr.cpp)

target_include_directories(se2_localization_eskfr PUBLIC
  ${EIGEN3_INCLUDE_DIR}
  ${manif_INCLUDE_DIRS})

# SE3 example
add_executable(se3_localization_eskfl
  src/se3_localization_eskfl.cpp)

target_include_directories(se3_localization_eskfl PUBLIC
  ${EIGEN3_INCLUDE_DIR}
  ${manif_INCLUDE_DIRS})
