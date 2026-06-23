#include <iostream>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "imx219_camera_cpp/Imx219CameraNode.hpp"

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    try {
        auto node = std::make_shared<Imx219CameraNode>();
        rclcpp::spin(node);
    } catch (const std::exception &e) {
        std::cerr << "imx219_camera_node failed: " << e.what() << std::endl;
    }

    rclcpp::shutdown();
    return 0;
}
