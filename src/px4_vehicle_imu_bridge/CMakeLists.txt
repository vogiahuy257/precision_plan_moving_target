cmake_minimum_required(VERSION 3.8)
project(px4_vehicle_imu_bridge)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(px4_msgs REQUIRED)

add_executable(px4_vehicle_imu_bridge_node
  src/px4_vehicle_imu_bridge_node.cpp
)

ament_target_dependencies(px4_vehicle_imu_bridge_node
  rclcpp
  sensor_msgs
  px4_msgs
)

install(TARGETS
  px4_vehicle_imu_bridge_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
