cmake_minimum_required(VERSION 3.5)
project(lidar_based_velocity_updater)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

ament_auto_add_library(lidar_based_velocity_updater SHARED
  src/debug_marker.cpp
  src/node.cpp
  src/adaptive_cruise_control.cpp
)

target_include_directories(lidar_based_velocity_updater
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

target_link_libraries(lidar_based_velocity_updater
    ${OpenCV_LIBRARIES}
    ${PCL_LIBRARIES}
)

rclcpp_components_register_node(lidar_based_velocity_updater
  PLUGIN "motion_planning::ObstacleStopPlannerNode"
  EXECUTABLE lidar_based_velocity_updater_node
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
  config
  launch
)
