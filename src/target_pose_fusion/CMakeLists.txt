cmake_minimum_required(VERSION 3.8)
project(target_pose_fusion)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ===== Dependencies =====
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# ===== Node =====
add_executable(target_pose_fusion_node
  KalmanFilter.cpp
)

# include current folder (.hpp)
target_include_directories(target_pose_fusion_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

ament_target_dependencies(target_pose_fusion_node
  rclcpp
  geometry_msgs
  std_msgs
  px4_msgs
)

target_link_libraries(target_pose_fusion_node
  ${OpenCV_LIBS}
)

# ===== Install =====
install(TARGETS
  target_pose_fusion_node
  DESTINATION lib/${PROJECT_NAME}
)

# ===== Testing =====
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()