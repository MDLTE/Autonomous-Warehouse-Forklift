#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════
#  botoni_ui.launch.py — levanta TODA la interfaz de Botoni
#
#  Arranca 4 cosas:
#    1. botoni_web_server  (:5173) → sirve la página web
#    2. rosbridge_server   (:9090) → datos para la UI (topics)
#    3. web_video_server   (:8080) → cámara MJPEG/compressed
#    4. ngrok_node         → túnel HTTPS, loguea solo la URL
#
#  USO:
#     ros2 launch botoni_ui botoni_ui.launch.py
#  Luego abre desde cualquier compu en la red:
#     http://<IP_DE_ESTA_COMPU>:5173
#  O desde el cel por ngrok (ver URL en los logs)
# ═══════════════════════════════════════════════════════════════════
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    return LaunchDescription([

        # ── 1. servidor web de la interfaz ──
        Node(
            package='botoni_ui',
            executable='web_server',
            name='botoni_web_server',
            parameters=[{'port': 5173, 'address': '0.0.0.0'}],
            output='screen',
        ),

        # ── 2. rosbridge: WebSocket para los topics ──
        Node(
            package='rosbridge_server',
            executable='rosbridge_websocket',
            name='rosbridge_websocket',
            output='log',
            parameters=[{
                'port': 9090,
                'address': '0.0.0.0',
                'max_message_size': 10000000,
            }],
            ros_arguments=['--log-level', 'ERROR'],
        ),

        # ── 3. web_video_server: cámara accesible por HTTP ──
        Node(
            package='web_video_server',
            executable='web_video_server',
            name='web_video_server',
            parameters=[{'port': 8080, 'address': '0.0.0.0'}],
            output='screen',
        ),

        # ── 4. ngrok: túnel HTTPS para el cel ──
#        Node(
#            package='botoni_ui',
#            executable='ngrok_node',
#            name='ngrok_node',
#            output='screen',
#        ),
    ])