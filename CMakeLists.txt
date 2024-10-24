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

# SE3 ESKF left form
add_executable(se3_localization_eskfl
  src/se3_localization_eskfl.cpp)

target_include_directories(se3_localization_eskfl PUBLIC
  ${EIGEN3_INCLUDE_DIR}
  ${manif_INCLUDE_DIRS})

# SE3 ESKF right form
add_executable(se3_localization_eskfr
  src/se3_localization_eskfr.cpp)

target_include_directories(se3_localization_eskfr PUBLIC
  ${EIGEN3_INCLUDE_DIR}
  ${manif_INCLUDE_DIRS})

# SE_2(3) ESKF mixed
add_executable(se_2_3_localization_eskf
  src/se_2_3_localization_eskf.cpp)

target_include_directories(se_2_3_localization_eskf PUBLIC
  ${EIGEN3_INCLUDE_DIR}
  ${manif_INCLUDE_DIRS})
