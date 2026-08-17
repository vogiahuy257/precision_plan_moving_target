cmake_minimum_required(VERSION 3.5)
project(target_drop)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter)

find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(px4_ros2_cpp REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(target_drop
    TargetDrop.cpp
    DropPred.cpp
    DropGate.cpp
)

target_include_directories(target_drop PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(target_drop Eigen3::Eigen)

ament_target_dependencies(
    target_drop
    rclcpp
    Eigen3
    px4_ros2_cpp
    px4_msgs
    geometry_msgs
    std_msgs
)

target_compile_features(target_drop PUBLIC c_std_99 cxx_std_20)

install(TARGETS target_drop DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY cfg DESTINATION share/${PROJECT_NAME}/)

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()
endif()

ament_package()
