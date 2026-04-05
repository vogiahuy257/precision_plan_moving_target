cmake_minimum_required(VERSION 3.8)
project(gimbal_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

add_executable(gimbal_controller_node
  GimbalController.cpp
)

target_include_directories(gimbal_controller_node PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

ament_target_dependencies(gimbal_controller_node
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2
  tf2_geometry_msgs
)

install(TARGETS
  gimbal_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()