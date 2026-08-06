from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ============================================================
    # DEV NOTE - PROJECT CUA TOI
    # ============================================================
    # Workspace:
    #   /home/pihuy/precision_plan_moving_target
    #
    # Config VINS dang dung:
    #   /home/pihuy/precision_plan_moving_target/src/VINS-MONO-ROS2/config_pkg/config/px4/px4_config.yaml
    #
    # Camera real tren Pi5:
    #   /camera/image
    #
    # IMU da bridge tu PX4 VehicleImu sang sensor_msgs/Imu:
    #   /imu0
    #
    # File launch nay chi chay feature_tracker.
    # feature_tracker doc image_topic, max_cnt, min_dist, freq,
    # F_threshold, show_track, equalize tu file px4_config.yaml.
    #
    # Output quan trong:
    #   /feature_tracker/feature
    #   /feature_tracker/feature_img neu show_track = 1
    # ============================================================

    default_config_file = (
        '/home/pihuy/precision_plan_moving_target/src/'
        'VINS-MONO-ROS2/config_pkg/config/px4/px4_config.yaml'
    )

    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config_file,
        description='Full path to VINS PX4 YAML config file'
    )

    config_file = LaunchConfiguration('config_file')

    feature_tracker_node = Node(
        package='feature_tracker',
        executable='feature_tracker',
        name='feature_tracker',
        namespace='feature_tracker',
        output='screen',
        parameters=[{
            'config_file': config_file
        }]
    )

    return LaunchDescription([
        config_file_arg,

        LogInfo(msg=['[feature_tracker launch] config_file: ', config_file]),
        LogInfo(msg=['[feature_tracker launch] image_topic is read from YAML']),
        LogInfo(msg=['[feature_tracker launch] expected image_topic: /camera/image']),
        LogInfo(msg=['[feature_tracker launch] output topic: /feature_tracker/feature']),

        feature_tracker_node
    ])
