cmake_minimum_required(VERSION 3.8)
project(logger)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(logger_node LoggerNode.cpp)

target_include_directories(logger_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/includes>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

ament_target_dependencies(logger_node
  rclcpp
  geometry_msgs
  std_msgs
)

target_compile_features(logger_node PUBLIC cxx_std_17)

install(TARGETS logger_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY includes/
  DESTINATION include/${PROJECT_NAME}
)

install(FILES cfg/params.yaml
  DESTINATION share/${PROJECT_NAME}/cfg
)

ament_package()
