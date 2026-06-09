import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    world_path = os.path.join(
        get_package_share_directory('slam_pkg'),
        'worlds',
        'e80.world'
    )

    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    return LaunchDescription([
        # Start Gazebo server with your world
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
            ),
            launch_arguments={'world': world_path}.items()
        ),

        # Start Gazebo client (GUI)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
            )
        ),

        # Robot state publisher
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('turtlebot3_gazebo'),
                    'launch',
                    'robot_state_publisher.launch.py'
                )
            )
        ),

        # Spawn Turtlebot3
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', 'burger',
                '-file', os.path.join(
                    get_package_share_directory('turtlebot3_gazebo'),
                    'models',
                    'turtlebot3_burger',
                    'model.sdf'
                ),
                '-x', '0.35',   
                '-y', '0.35',   
                '-z', '0.02',
                '-Y', '0.0',  
            ],
            output='screen'
        ),
    ])