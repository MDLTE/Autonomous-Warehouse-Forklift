"""
test_straight.py — Encoder Balance Diagnostic
==============================================
Sends a constant forward velocity and logs encoder readings
to verify that both wheels are spinning at the same rate.

If the robot drifts left/right, the encoders are unbalanced
and you may need to calibrate or replace motors.

Usage:
  ros2 run puzzlebot_ros test_straight
  (runs for 5 seconds, then prints summary)
"""

import rclpy
from rclpy.node import Node
from rclpy import qos

from std_msgs.msg import Float32
from geometry_msgs.msg import Twist


class TestStraightNode(Node):

    def __init__(self):
        super().__init__('test_straight')

        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.sub_encR = self.create_subscription(
            Float32, 'VelocityEncR', self.encR_cb, qos.qos_profile_sensor_data)
        self.sub_encL = self.create_subscription(
            Float32, 'VelocityEncL', self.encL_cb, qos.qos_profile_sensor_data)

        self.readings_R = []
        self.readings_L = []
        self.duration = 5.0  # seconds
        self.v_test = 0.15   # m/s

        self.timer = self.create_timer(0.05, self.loop)
        self.start_time = self.get_clock().now()
        self.get_logger().info(f'Starting straight test at v={self.v_test} m/s for {self.duration}s')

    def encR_cb(self, msg):
        self.readings_R.append(msg.data)

    def encL_cb(self, msg):
        self.readings_L.append(msg.data)

    def loop(self):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        if elapsed < self.duration:
            msg = Twist()
            msg.linear.x = self.v_test
            msg.angular.z = 0.0
            self.pub_cmd.publish(msg)
        else:
            # Stop
            msg = Twist()
            self.pub_cmd.publish(msg)

            # Report
            if self.readings_R and self.readings_L:
                import numpy as np
                avg_r = np.mean(self.readings_R)
                avg_l = np.mean(self.readings_L)
                ratio = avg_r / avg_l if avg_l != 0 else float('inf')
                diff = abs(avg_r - avg_l)

                self.get_logger().info(f'=== RESULTS ===')
                self.get_logger().info(f'Avg ωR: {avg_r:.4f} rad/s')
                self.get_logger().info(f'Avg ωL: {avg_l:.4f} rad/s')
                self.get_logger().info(f'Ratio R/L: {ratio:.4f}')
                self.get_logger().info(f'Difference: {diff:.4f} rad/s')

                if diff < 0.05:
                    self.get_logger().info('Wheels balanced — no PID needed')
                elif avg_r > avg_l:
                    self.get_logger().info('Right wheel faster — robot drifts right')
                else:
                    self.get_logger().info('Left wheel faster — robot drifts left')

            self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    node = TestStraightNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
