cmake_minimum_required(VERSION 3.8)
project(ekf_filter)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(ekf_filter_node
    EKFNode.cpp
    CtraEkf.cpp
    FrameTransformer.cpp
)

target_include_directories(ekf_filter_node PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/includes>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

ament_target_dependencies(ekf_filter_node
    rclcpp
    std_msgs
    geometry_msgs
    px4_msgs
)

target_link_libraries(ekf_filter_node
    Eigen3::Eigen
)

target_compile_features(ekf_filter_node PUBLIC cxx_std_17)

install(TARGETS
    ekf_filter_node
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY includes/
    DESTINATION include/${PROJECT_NAME}
)

install(FILES
    cfg/params.yaml
    cfg/target_drop_ekf_override.yaml
    DESTINATION share/${PROJECT_NAME}/cfg
)

install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
