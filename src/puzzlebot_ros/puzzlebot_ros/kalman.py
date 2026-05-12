"""
kalman.py — PuzzleBot TE3003B
===============================
Extended Kalman Filter node for localization.

Prediction: encoder odometry (Dead Reckoning)
Correction: ArUco marker detections (absolute pose)

Publishes: /odom (nav_msgs/Odometry)
Subscribes: /VelocityEncR, /VelocityEncL, /marker_publisher/markers

The goto_point node reads /odom when use_ekf = True.
"""

import rclpy
from rclpy.node import Node
from rclpy import qos

import math
import numpy as np
import yaml
import os

from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion

from .my_math import wrap_to_pi


class KalmanNode(Node):

    def __init__(self):
        super().__init__('kalman')

        # ── Robot parameters ──
        self.wheel_radius = 0.05   # m
        self.wheel_base   = 0.18   # m
        self.dt           = 0.02   # 50 Hz predict rate

        # ── EKF State ──
        self.x = np.array([0.0, 0.0, 0.0])   # [x, y, θ]
        self.P = np.diag([0.01, 0.01, 0.01])  # initial covariance

        # ── Noise matrices ──
        # Q: process noise (encoder trust)
        self.Q = np.diag([0.02**2, 0.02**2, 0.01**2])
        # R: measurement noise (ArUco trust)
        self.R = np.diag([0.05**2, 0.05**2, 0.02**2])
        # H: observation matrix (ArUco measures [x, y, θ] directly)
        self.H = np.eye(3)

        # ── ArUco marker map ──
        # Format: {marker_id: [x, y, θ]}
        # Load from config or define here
        self.aruco_map = {
            0: np.array([1.0, 0.0, 0.0]),
            1: np.array([0.0, -1.0, math.pi / 2]),
        }
        self._load_aruco_map()

        # ── Encoder state ──
        self.wR = 0.0
        self.wL = 0.0

        # ── Publishers ──
        self.pub_odom = self.create_publisher(Odometry, '/odom', 10)

        # ── Subscribers ──
        self.sub_encR = self.create_subscription(
            Float32, 'VelocityEncR', self.encR_callback,
            qos.qos_profile_sensor_data)
        self.sub_encL = self.create_subscription(
            Float32, 'VelocityEncL', self.encL_callback,
            qos.qos_profile_sensor_data)

        # TODO: Subscribe to ArUco marker topic
        # self.sub_markers = self.create_subscription(
        #     MarkerArray, '/marker_publisher/markers',
        #     self.marker_callback, 10)

        # ── Timer ──
        self.timer = self.create_timer(self.dt, self.predict_and_publish)

        self.get_logger().info('EKF node started')
        self.get_logger().info(f'ArUco map: {self.aruco_map}')

    def _load_aruco_map(self):
        """Try to load ArUco map from config/aruco_map.yaml"""
        config_paths = [
            os.path.expanduser('~/ros2_ws/src/puzzlebot_ros/config/aruco_map.yaml'),
            os.path.join(os.path.dirname(__file__), '..', 'config', 'aruco_map.yaml'),
        ]
        for path in config_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        data = yaml.safe_load(f)
                    self.aruco_map = {
                        m['id']: np.array(m['position'])
                        for m in data.get('markers', [])
                    }
                    self.get_logger().info(f'Loaded ArUco map from {path}')
                    return
                except Exception as e:
                    self.get_logger().warn(f'Failed to load ArUco map: {e}')

    # ── Callbacks ──
    def encR_callback(self, msg):
        self.wR = msg.data

    def encL_callback(self, msg):
        self.wL = msg.data

    def marker_callback(self, msg):
        """
        Called when ArUco markers are detected.
        Extract robot pose from marker detection and run EKF update.
        
        TODO: Adapt to your specific marker message type.
        The key operation is:
          1. Get marker ID
          2. Look up known marker position in self.aruco_map
          3. Compute robot pose z = [x, y, θ] from the detection
          4. Call self.update(z)
        """
        # Example (adapt to actual message type):
        # for marker in msg.markers:
        #     if marker.id in self.aruco_map:
        #         z = compute_robot_pose(marker, self.aruco_map[marker.id])
        #         K = self.update(z)
        #         self.get_logger().info(
        #             f'ArUco {marker.id} correction — K diag: {np.diag(K)}')
        pass

    # ── EKF Predict ──
    def predict_and_publish(self):
        r = self.wheel_radius
        L = self.wheel_base
        dt = self.dt

        # Compute v, ω from encoders
        v = r / 2.0 * (self.wR + self.wL)
        w = r / L   * (self.wR - self.wL)

        # ── Predict state ──
        x, y, th = self.x
        self.x = np.array([
            x + v * math.cos(th) * dt,
            y + v * math.sin(th) * dt,
            wrap_to_pi(th + w * dt),
        ])

        # ── Jacobian F ──
        F = np.array([
            [1, 0, -v * math.sin(th) * dt],
            [0, 1,  v * math.cos(th) * dt],
            [0, 0,  1],
        ])

        # ── Propagate covariance ──
        self.P = F @ self.P @ F.T + self.Q

        # ── Publish /odom ──
        self.publish_odom()

    # ── EKF Update ──
    def update(self, z):
        """
        Correction step when an ArUco marker is detected.
        z: np.array([x, y, θ]) — measured robot pose in world frame
        Returns: Kalman gain K
        """
        # Innovation
        inn = z - self.H @ self.x
        inn[2] = wrap_to_pi(inn[2])

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Correct state
        self.x = self.x + K @ inn
        self.x[2] = wrap_to_pi(self.x[2])

        # Correct covariance
        self.P = (np.eye(3) - K @ self.H) @ self.P

        return K

    # ── Publish Odometry ──
    def publish_odom(self):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'

        msg.pose.pose.position.x = self.x[0]
        msg.pose.pose.position.y = self.x[1]
        msg.pose.pose.position.z = 0.0

        # Convert θ to quaternion (rotation around z)
        msg.pose.pose.orientation = self._yaw_to_quaternion(self.x[2])

        # Covariance (6x6, only top-left 3x3 meaningful for 2D)
        cov = [0.0] * 36
        cov[0]  = self.P[0, 0]   # var(x)
        cov[1]  = self.P[0, 1]   # cov(x,y)
        cov[5]  = self.P[0, 2]   # cov(x,θ)
        cov[6]  = self.P[1, 0]
        cov[7]  = self.P[1, 1]   # var(y)
        cov[11] = self.P[1, 2]   # cov(y,θ)
        cov[30] = self.P[2, 0]
        cov[31] = self.P[2, 1]
        cov[35] = self.P[2, 2]   # var(θ)
        msg.pose.covariance = cov

        self.pub_odom.publish(msg)

    def _yaw_to_quaternion(self, yaw):
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q


def main(args=None):
    rclpy.init(args=args)
    node = KalmanNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
