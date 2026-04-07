cmake_minimum_required(VERSION 3.8)
project(kalman_filter)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ===== Dependencies =====
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# ===== Node =====
add_executable(kalman_filter_node
    KalmanFilter.cpp
)

# include current folder (.hpp)
target_include_directories(kalman_filter_node PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(kalman_filter_node
    rclcpp
    std_msgs
    geometry_msgs
    px4_msgs
)

target_link_libraries(kalman_filter_node
    Eigen3::Eigen
    ${OpenCV_LIBS}
)

target_compile_features(kalman_filter_node PUBLIC cxx_std_17)

# ===== Install =====
install(TARGETS
    kalman_filter_node
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY cfg
    DESTINATION share/${PROJECT_NAME}
)

# ===== Testing =====
if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    set(ament_cmake_copyright_FOUND TRUE)
    set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif()

ament_package()