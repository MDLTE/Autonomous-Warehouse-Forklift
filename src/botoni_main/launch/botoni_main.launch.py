#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    import subprocess
    ip = subprocess.check_output("hostname -I | awk '{print $1}'", shell=True).decode().strip()
    print(f"\n{'='*50}")
    print(f"  BOTONI — Dashboard en: http://{ip}:5173")
    print(f"{'='*50}\n")

    return LaunchDescription([

        # ── 0. Kill procesos anteriores ──────────────────────────────────────
        ExecuteProcess(
            cmd=['bash', '-c',
                'pkill -f web_server 2>/dev/null; '
                'pkill -f rosbridge_websocket 2>/dev/null; '
                'pkill -f web_video_server 2>/dev/null; '
                'pkill -f rosapi_node 2>/dev/null; '
                'fuser -k 9090/tcp 2>/dev/null; '
                'fuser -k 8080/tcp 2>/dev/null; '
                'exit 0'
            ],
            name='kill_old_procs',
            output='screen',
        ),

        # ── 1. poseKalman (EKF) ──────────────────────────────────────────────
        Node(
            package='puzzlebot_ros',
            executable='poseKalman',
            name='poseKalman',
            output='screen',
        ),

        # ── 2. ArUco detector ────────────────────────────────────────────────
        Node(
            package='puzzlebot_ros',
            executable='aruco_detector',
            name='aruco_detector',
            output='screen',
        ),

        # ── 3. QR detector ───────────────────────────────────────────────────
        Node(
            package='puzzlebot_ros',
            executable='qr_detector',
            name='qr_detector',
            output='screen',
        ),

        # ── 4. QR alignment ──────────────────────────────────────────────────
        Node(
            package='puzzlebot_ros',
            executable='qr_alignment_node',
            name='qr_alignment_node',
            output='screen',
        ),

        # ── 5. Door align ────────────────────────────────────────────────────
        Node(
            package='puzzlebot_ros',
            executable='door_align',
            name='door_align',
            output='screen',
        ),

        # ── 6. Navegación ────────────────────────────────────────────────────
        ExecuteProcess(
            cmd=['python3', '/home/marcelo/ros2_ws/src/nav_pkg/nav_pkg/nav_node_pb.py'],
            name='nav_node',
            output='screen',
        ),

        # ── 7. Voice node ────────────────────────────────────────────────────
        ExecuteProcess(
             cmd=['python3', '/home/marcelo/ros2_ws/src/nav_pkg/nav_pkg/voice_node.py'],
             name='voice_node',
             output='screen',
         ),

        # ── 8. UI completa (web_server, rosbridge, video, ngrok) ─────────────
        TimerAction(period=4.0, actions=[
            IncludeLaunchDescription(
                AnyLaunchDescriptionSource([
                    PathJoinSubstitution([
                        FindPackageShare('botoni_ui'),
                        'launch',
                        'botoni_ui.launch.py'
                    ])
                ])
            ),
        ]),

        # ── 9. FSM principal ─────────────────────────────────────────────────
        TimerAction(period=6.0, actions=[
            Node(
                package='botoni_main',
                executable='botoni_fsm',
                name='botoni_master',
                output='screen',
                emulate_tty=True,
            ),
        ]),

    ])