from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition

def generate_launch_description():
    enable_viz_arg = DeclareLaunchArgument(
        'enable_gazebo_viz',
        default_value='true',
        description='Enable Gazebo marker visualization (simulation only)'
    )

    sim_time = {'use_sim_time': True}

    return LaunchDescription([
        enable_viz_arg,

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='image_bridge',
            arguments=[
                '/world/aruco/model/x500_mono_cam_down_0/link/camera_link/sensor/imager/image@sensor_msgs/msg/Image@gz.msgs.Image'
            ],
            parameters=[sim_time],
            output='screen',
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='camera_info_bridge',
            arguments=[
                '/world/aruco/model/x500_mono_cam_down_0/link/camera_link/sensor/imager/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo'
            ],
            parameters=[sim_time],
            output='screen',
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='image_proc_bridge',
            arguments=[
                '/image_proc@sensor_msgs/msg/Image[gz.msgs.Image'
            ],
            parameters=[{
                'use_sim_time': True,
                'qos_overrides./image_proc.subscription.reliability': 'best_effort',
                'qos_overrides./image_proc.publisher.reliability': 'best_effort'
            }],
            output='screen',
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='clock_bridge',
            arguments=[
                '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
            ],
            parameters=[{'use_sim_time': True}],
            output='screen',
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='land_pad_cmd_vel_bridge',
            arguments=[
                '/land_pad/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist'
            ],
            parameters=[sim_time],
            output='screen',
        ),

        Node(
            package='aruco_tracker',
            executable='aruco_tracker',
            name='aruco_tracker',
            output='screen',
            parameters=[
                sim_time,
                PathJoinSubstitution([FindPackageShare('aruco_tracker'), 'cfg', 'params.yaml'])
            ]
        ),

        Node(
            package='precision_land',
            executable='precision_land',
            name='precision_land',
            output='screen',
            parameters=[
                sim_time
                # PathJoinSubstitution([FindPackageShare('precision_land'), 'cfg', 'params.yaml'])
            ]
        ),

        Node(
            package='precision_land_viz',
            executable='tag_pose_visualizer',
            name='tag_pose_visualizer',
            output='screen',
            parameters=[sim_time],
            condition=IfCondition(LaunchConfiguration('enable_gazebo_viz'))
        ),

        Node(
            package='kalman_filter',
            executable='kalman_filter_node',
            name='kalman_filter',
            output='screen',
            parameters=[sim_time]
        ),
    ])