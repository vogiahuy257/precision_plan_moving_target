from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # ============================================================
    # DEV NOTE - VINS PX4 REAL PI5
    # ============================================================
    # Workspace:
    #   /home/pihuy/precision_plan_moving_target
    #
    # Camera:
    #   /camera/image
    #
    # Raw PX4 IMU topic:
    #   /fmu/out/vehicle_imu
    #
    # Bridge output for VINS:
    #   /imu0
    #
    # VINS config:
    #   config_pkg/config/px4/px4_config.yaml
    #
    # Pipeline:
    #   /fmu/out/vehicle_imu
    #        ↓
    #   px4_vehicle_imu_bridge
    #        ↓
    #   /imu0
    #
    #   /camera/image + /imu0
    #        ↓
    #   feature_tracker + vins_estimator
    #        ↓
    #   /odometry
    #
    # Pose graph mac dinh tat de nhe CPU tren Pi5.
    # ============================================================

    config_pkg_path = get_package_share_directory('config_pkg')

    default_config_path = PathJoinSubstitution([
        config_pkg_path,
        'config',
        'px4',
        'px4_config.yaml'
    ])

    default_support_path = PathJoinSubstitution([
        config_pkg_path,
        'support_files'
    ])

    default_vins_folder = config_pkg_path

    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config_path,
        description='Path to px4_config.yaml'
    )

    support_file_arg = DeclareLaunchArgument(
        'support_file',
        default_value=default_support_path,
        description='Path to pose_graph support_files folder'
    )

    vins_folder_arg = DeclareLaunchArgument(
        'vins_folder',
        default_value=default_vins_folder,
        description='Path to VINS/config package folder'
    )

    run_imu_bridge_arg = DeclareLaunchArgument(
        'run_imu_bridge',
        default_value='true',
        description='Run PX4 VehicleImu to /imu0 bridge'
    )

    run_feature_tracker_arg = DeclareLaunchArgument(
        'run_feature_tracker',
        default_value='true',
        description='Run feature_tracker node'
    )

    run_vins_estimator_arg = DeclareLaunchArgument(
        'run_vins_estimator',
        default_value='true',
        description='Run vins_estimator node'
    )

    run_pose_graph_arg = DeclareLaunchArgument(
        'run_pose_graph',
        default_value='false',
        description='Run pose_graph node'
    )

    config_file = LaunchConfiguration('config_file')
    support_file = LaunchConfiguration('support_file')
    vins_folder = LaunchConfiguration('vins_folder')

    run_imu_bridge = LaunchConfiguration('run_imu_bridge')
    run_feature_tracker = LaunchConfiguration('run_feature_tracker')
    run_vins_estimator = LaunchConfiguration('run_vins_estimator')
    run_pose_graph = LaunchConfiguration('run_pose_graph')

    # ============================================================
    # PX4 VEHICLE IMU BRIDGE
    # ============================================================
    px4_vehicle_imu_bridge_node = Node(
        condition=IfCondition(run_imu_bridge),
        package='px4_vehicle_imu_bridge',
        executable='px4_vehicle_imu_bridge_node',
        name='px4_vehicle_imu_bridge_node',
        output='screen',
        parameters=[{
            'input_topic': '/fmu/out/vehicle_imu',
            'output_topic': '/imu0',
            'frame_id': 'imu_link',
            'convert_frd_to_flu': True,
            'use_ros_receive_time': False
        }]
    )

    # ============================================================
    # FEATURE TRACKER
    # ============================================================
    feature_tracker_node = Node(
        condition=IfCondition(run_feature_tracker),
        package='feature_tracker',
        executable='feature_tracker',
        name='feature_tracker',
        namespace='feature_tracker',
        output='screen',
        parameters=[{
            'config_file': config_file
        }]
    )

    # ============================================================
    # VINS ESTIMATOR
    # ============================================================
    vins_estimator_node = Node(
        condition=IfCondition(run_vins_estimator),
        package='vins_estimator',
        executable='vins_estimator',
        name='vins_estimator',
        namespace='vins_estimator',
        output='screen',
        parameters=[{
            'config_file': config_file,
            'vins_folder': vins_folder
        }]
    )

    # ============================================================
    # POSE GRAPH
    # ============================================================
    pose_graph_node = Node(
        condition=IfCondition(run_pose_graph),
        package='pose_graph',
        executable='pose_graph',
        name='pose_graph',
        namespace='pose_graph',
        output='screen',
        parameters=[{
            'config_file': config_file,
            'support_file': support_file,
            'visualization_shift_x': 0,
            'visualization_shift_y': 0,
            'skip_cnt': 0,
            'skip_dis': 0.0
        }]
    )

    return LaunchDescription([
        config_file_arg,
        support_file_arg,
        vins_folder_arg,
        run_imu_bridge_arg,
        run_feature_tracker_arg,
        run_vins_estimator_arg,
        run_pose_graph_arg,

        LogInfo(msg=['[PX4 VINS launch] config_file         : ', config_file]),
        LogInfo(msg=['[PX4 VINS launch] vins_folder         : ', vins_folder]),
        LogInfo(msg=['[PX4 VINS launch] support_file        : ', support_file]),
        LogInfo(msg=['[PX4 VINS launch] run_imu_bridge      : ', run_imu_bridge]),
        LogInfo(msg=['[PX4 VINS launch] run_feature_tracker : ', run_feature_tracker]),
        LogInfo(msg=['[PX4 VINS launch] run_vins_estimator  : ', run_vins_estimator]),
        LogInfo(msg=['[PX4 VINS launch] run_pose_graph      : ', run_pose_graph]),

        px4_vehicle_imu_bridge_node,
        feature_tracker_node,
        vins_estimator_node,
        pose_graph_node
    ])
