from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = PathJoinSubstitution([
        FindPackageShare("imx219_camera_cpp"),
        "config",
        "imx219_camera.yaml",
    ])

    return LaunchDescription([
        Node(
            package="imx219_camera_cpp",
            executable="imx219_camera_node",
            name="imx219_camera_node",
            output="screen",
            parameters=[config_file],
        )
    ])
