"""
goto_point.py — PuzzleBot TE3003B
==================================
Displaced-point trajectory controller with Dead Reckoning.

Architecture:
  /VelocityEncL + /VelocityEncR → Dead Reckoning → pose (x, y, θ)
  pose + current waypoint        → Displaced point → V, ω
  V, ω                          → /cmd_vel

When EKF is ready:
  - Set use_ekf = True
  - Node reads /odom instead of computing Dead Reckoning internally

Parameters to tune on the real robot:
  h      : displaced point distance (m)  → start at 0.05
  k      : controller gain               → start at 1.0
  v_max  : max linear velocity (m/s)
  w_max  : max angular velocity (rad/s)
  D_min  : distance tolerance for waypoint reached (m)
"""

import rclpy
from rclpy.node import Node
from rclpy import qos

import signal
import math

from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from .my_math import wrap_to_pi
from .my_math import euler_from_quaternion


class GotoPoint(Node):

    def __init__(self):
        super().__init__('goto_point')

        # ── Publishers ──
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)

        # ── Subscribers ──
        self.sub_encR = self.create_subscription(
            Float32, 'VelocityEncR', self.encR_callback,
            qos.qos_profile_sensor_data)
        self.sub_encL = self.create_subscription(
            Float32, 'VelocityEncL', self.encL_callback,
            qos.qos_profile_sensor_data)
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self.odom_callback,
            qos.qos_profile_sensor_data)

        # ── Timers ──
        self.dt_control = 0.05   # 20 Hz
        self.dt_dr      = 0.02   # 50 Hz

        self.timer_control = self.create_timer(self.dt_control, self.control_loop)
        self.timer_dr      = self.create_timer(self.dt_dr,      self.dead_reckoning_loop)

        # ── Controller parameters ──
        self.h     = 0.05    # displaced point distance (m)
        self.k     = 1.0     # proportional gain
        self.v_max = 0.20    # max linear velocity (m/s)
        self.w_max = 1.5     # max angular velocity (rad/s)
        self.D_min     = 0.07    # position tolerance (m)
        self.THETA_MIN = 0.05    # angle tolerance (rad) ≈ 3°

        # ── Robot parameters ──
        self.wheel_radius = 0.05   # m
        self.wheel_base   = 0.18   # m

        # ── State ──
        self.wR = 0.0
        self.wL = 0.0
        self.pose_x     = 0.0
        self.pose_y     = 0.0
        self.pose_theta = 0.0

        # ── EKF switch ──
        self.use_ekf = False  # Set True when kalman node is running

        # ── Waypoints: (x, y, θ) ──
        self.waypoints = [
            (0.5, 0.0, 0.0),
        ]
        self.wp_idx = 0
        self.phase = 'position'  # 'position' or 'orientation'

        self.get_logger().info(f'Waypoints: {self.waypoints}')
        self.get_logger().info(f'Mode: {"EKF (/odom)" if self.use_ekf else "Dead Reckoning"}')

    # ── Encoder callbacks ──
    def encR_callback(self, msg):
        self.wR = msg.data

    def encL_callback(self, msg):
        self.wL = msg.data

    # ── Odometry callback (EKF mode) ──
    def odom_callback(self, msg):
        if self.use_ekf:
            self.pose_x = msg.pose.pose.position.x
            self.pose_y = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            _, _, self.pose_theta = euler_from_quaternion(q.x, q.y, q.z, q.w)

    # ── Dead Reckoning (internal, when not using EKF) ──
    def dead_reckoning_loop(self):
        if self.use_ekf:
            return  # EKF provides pose via /odom

        r = self.wheel_radius
        L = self.wheel_base
        dt = self.dt_dr

        v = r / 2.0 * (self.wR + self.wL)
        w = r / L   * (self.wR - self.wL)

        self.pose_x     += v * math.cos(self.pose_theta) * dt
        self.pose_y     += v * math.sin(self.pose_theta) * dt
        self.pose_theta  = wrap_to_pi(self.pose_theta + w * dt)

    # ── Control loop ──
    def control_loop(self):
        if self.wp_idx >= len(self.waypoints):
            self.stop()
            return

        xg, yg, thg = self.waypoints[self.wp_idx]
        x  = self.pose_x
        y  = self.pose_y
        th = self.pose_theta

        if self.phase == 'position':
            # ── Displaced-point controller ──
            xh = x + self.h * math.cos(th)
            yh = y + self.h * math.sin(th)

            e1 = xg - xh
            e2 = yg - yh
            dist = math.sqrt(e1**2 + e2**2)

            if dist < self.D_min:
                self.phase = 'orientation'
                self.get_logger().info(
                    f'WP {self.wp_idx} position reached — aligning to θ={thg:.2f}')
                return

            # Invert B_h matrix
            cos_th = math.cos(th)
            sin_th = math.sin(th)
            det = self.h  # det(B_h) = h

            v = (cos_th * self.k * e1 + sin_th * self.k * e2)
            w = (-sin_th * self.k * e1 + cos_th * self.k * e2) / self.h

            # Saturate
            v = max(-self.v_max, min(self.v_max, v))
            w = max(-self.w_max, min(self.w_max, w))

            self.publish_cmd(v, w)

        elif self.phase == 'orientation':
            # ── Rotate in place to reach desired angle ──
            e_th = wrap_to_pi(thg - th)

            if abs(e_th) < self.THETA_MIN:
                self.get_logger().info(
                    f'WP {self.wp_idx} complete at ({x:.2f}, {y:.2f}, {th:.2f})')
                self.wp_idx += 1
                self.phase = 'position'

                if self.wp_idx >= len(self.waypoints):
                    self.get_logger().info('All waypoints reached!')
                    self.stop()
                return

            w = 1.5 * e_th
            w = max(-self.w_max, min(self.w_max, w))
            self.publish_cmd(0.0, w)

    def publish_cmd(self, v, w):
        msg = Twist()
        msg.linear.x  = v
        msg.angular.z = w
        self.pub_cmd.publish(msg)

    def stop(self):
        self.publish_cmd(0.0, 0.0)
        self.get_logger().info('Stopped.')

    def stop_handler(self, sig, frame):
        self.stop()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = GotoPoint()
    signal.signal(signal.SIGINT, node.stop_handler)
    rclpy.spin(node)
    node.destroy_node()


if __name__ == '__main__':
    main()
