import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_slam   = get_package_share_directory('slam_pkg')
    pkg_gz_ros = get_package_share_directory('gazebo_ros')
    pkg_tb3    = get_package_share_directory('turtlebot3_gazebo')

    world_path = os.path.join(pkg_slam, 'worlds', 'e80.world')

    urdf_path = os.path.join(pkg_tb3, 'urdf', 'turtlebot3_burger_cam.urdf')
    with open(urdf_path, 'r') as f:
        robot_desc = f.read()

    puzzlebot_sdf = os.path.join(pkg_slam, 'models', 'puzzlebot_cam', 'model.sdf')

    # ── obstacle cylinders ────────────────────────────────────────
    cylinder_positions = [
        ((2.558355 + 0.171638), 2.195521), # 0 
        (3.136760 + 0.171638, 1.841200), # 1
        (3.020870 + 0.171638, 2.713307), # 2
        (1.033290 - 0.105715, 2.140810), # 3
        (0.364489 - 0.105715, 1.813050), # 4
        (0.498502 - 0.105715, 2.673870), # 5
        (1.294170, 4.135671), # 6 
        (1.932185, 3.294141), # 7
        (1.769580, 0.968817) # 8
    ]

    cylinder_nodes = [
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity',   f'cylinder_{i}',
                '-database', 'unit_cylinder_15_CM',
                '-x', str(x),
                '-y', str(y),
                '-z', '0.5',
            ],
            output='screen'
        )
        for i, (x, y) in enumerate(cylinder_positions)
    ]

    return LaunchDescription([
        # Gazebo server
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gz_ros, 'launch', 'gzserver.launch.py')
            ),
            launch_arguments={'world': world_path}.items()
        ),
        # Gazebo client (GUI)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gz_ros, 'launch', 'gzclient.launch.py')
            )
        ),
        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'robot_description': robot_desc,
            }]
        ),
        # Spawn PuzzleBot
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', 'puzzlebot',
                '-file',   puzzlebot_sdf,
                '-x', '0.35',
                '-y', '0.35',
                '-z', '0.02',
                '-Y', '0.0',
            ],
            output='screen'
        ),
        *cylinder_nodes,
    ])