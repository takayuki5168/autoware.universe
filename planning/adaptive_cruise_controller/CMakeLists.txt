cmake_minimum_required(VERSION 3.5)
project(adaptive_cruise_controller)

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

find_package(OpenCV REQUIRED)

ament_auto_add_library(adaptive_cruise_controller SHARED
  src/adaptive_cruise_control.cpp
  src/adaptive_cruise_control_core.cpp
  src/acc_pid.cpp
  src/debug_marker.cpp
  src/utilities.cpp
)

target_include_directories(adaptive_cruise_controller
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(adaptive_cruise_controller
  ${OpenCV_LIBRARIES}
)


rclcpp_components_register_node(adaptive_cruise_controller
  PLUGIN "motion_planning::AdaptiveCruiseControllerNode"
  EXECUTABLE adaptive_cruise_controller_node
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
