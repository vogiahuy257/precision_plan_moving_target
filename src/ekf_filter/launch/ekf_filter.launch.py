from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="ekf_filter",
            executable="ekf_filter_node",
            name="ekf_filter",
            output="screen",
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare("ekf_filter"),
                    "cfg",
                    "params.yaml",
                ])
            ],
        )
    ])
