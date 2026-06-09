import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from rclpy.qos import QoSProfile, ReliabilityPolicy

class VelocityTester(Node):
    def __init__(self):
        super().__init__('velocity_tester')

        # publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # robot parameters
        self.wheel_radius = 0.045
        self.robot_width  = 0.205

        # encoder velocities (rad/s)
        self.vel_L = 0.0
        self.vel_R = 0.0
        self.enc_received = False

        # computed robot velocities
        self.current_v = 0.0
        self.current_w = 0.0

        # QoS for encoder topics
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub_L = self.create_subscription(Float32, '/VelocityEncL', self.enc_L_cb, qos)
        self.sub_R = self.create_subscription(Float32, '/VelocityEncR', self.enc_R_cb, qos)

        # data recording
        self.times      = []
        self.linear_vs  = []
        self.angular_ws = []
        self.start_time = None

        # test state machine
        self.phase       = 'wait'
        self.phase_start = None

        # test parameters
        self.target_v   = 0.3   # m/s — max linear to test
        self.target_w   = 3.14   # rad/s — max angular to test
        self.hold_time  = 2.0   # seconds to hold max velocity
        self.ramp_cmd_v = 0.3
        self.ramp_cmd_w = 6.0
        self.accel_timeout = 5.0  # seconds before giving up on reaching target

        # control loop at 20Hz
        self.create_timer(0.05, self.control_loop)

        self.get_logger().info('Velocity tester started')
        self.get_logger().info('Waiting for encoder topics /VelocityEncL and /VelocityEncR')
        self.get_logger().info('Make sure robot has space to move!')

    def enc_L_cb(self, msg: Float32):
        self.vel_L = msg.data
        self.enc_received = True
        self.compute_velocities()

    def enc_R_cb(self, msg: Float32):
        self.vel_R = msg.data
        self.enc_received = True
        self.compute_velocities()

    def compute_velocities(self):
        R = self.wheel_radius
        L = self.robot_width
        self.current_v = (self.vel_R + self.vel_L) * R / 2.0
        self.current_w = (self.vel_R - self.vel_L) * R / L

        # record data
        if self.start_time is not None:
            t = time.time() - self.start_time
            self.times.append(t)
            self.linear_vs.append(self.current_v)
            self.angular_ws.append(self.current_w)

    def send_cmd(self, v, w):
        msg = Twist()
        msg.linear.x  = v
        msg.angular.z = w
        self.cmd_pub.publish(msg)

    def stop(self):
        self.send_cmd(0.0, 0.0)

    def control_loop(self):
        if not self.enc_received:
            self.get_logger().info(
                'Waiting for encoder data...', throttle_duration_sec=2.0)
            return

        now = time.time()

        # ── WAIT ──────────────────────────────────────────────────────
        if self.phase == 'wait':
            self.stop()
            self.get_logger().info('Starting test in 3 seconds...')
            time.sleep(3.0)
            self.start_time = time.time()
            self.phase = 'accel_linear'
            self.phase_start = now
            self.get_logger().info('Phase: LINEAR ACCELERATION')

        # ── LINEAR ACCELERATION ────────────────────────────────────────
        elif self.phase == 'accel_linear':
            self.send_cmd(self.ramp_cmd_v, 0.0)

            velocity_reached = abs(self.current_v) >= self.target_v * 0.95
            timeout          = now - self.phase_start > self.accel_timeout

            if velocity_reached or timeout:
                if timeout:
                    self.get_logger().warn(
                        f'Timeout — max linear reached: {self.current_v:.3f} m/s '
                        f'(target was {self.target_v} m/s)')
                else:
                    self.get_logger().info(
                        f'Max linear reached: {self.current_v:.3f} m/s')
                self.phase = 'hold_linear'
                self.phase_start = now
                self.get_logger().info('Phase: HOLD LINEAR')

        # ── HOLD LINEAR ────────────────────────────────────────────────
        elif self.phase == 'hold_linear':
            self.send_cmd(self.ramp_cmd_v, 0.0)
            if now - self.phase_start > self.hold_time:
                self.phase = 'decel_linear'
                self.phase_start = now
                self.get_logger().info('Phase: DECEL LINEAR')

        # ── DECEL LINEAR ───────────────────────────────────────────────
        elif self.phase == 'decel_linear':
            self.stop()
            if abs(self.current_v) < 0.01:
                time.sleep(1.0)
                self.phase = 'accel_angular'
                self.phase_start = now
                self.get_logger().info('Phase: ANGULAR ACCELERATION')

        # ── ANGULAR ACCELERATION ───────────────────────────────────────
        elif self.phase == 'accel_angular':
            self.send_cmd(0.0, self.ramp_cmd_w)

            velocity_reached = abs(self.current_w) >= self.target_w * 0.95
            timeout          = now - self.phase_start > self.accel_timeout

            if velocity_reached or timeout:
                if timeout:
                    self.get_logger().warn(
                        f'Timeout — max angular reached: {self.current_w:.3f} rad/s '
                        f'(target was {self.target_w} rad/s)')
                else:
                    self.get_logger().info(
                        f'Max angular reached: {self.current_w:.3f} rad/s')
                self.phase = 'hold_angular'
                self.phase_start = now
                self.get_logger().info('Phase: HOLD ANGULAR')

        # ── HOLD ANGULAR ───────────────────────────────────────────────
        elif self.phase == 'hold_angular':
            self.send_cmd(0.0, self.ramp_cmd_w)
            if now - self.phase_start > self.hold_time:
                self.phase = 'decel_angular'
                self.phase_start = now
                self.get_logger().info('Phase: DECEL ANGULAR')

        # ── DECEL ANGULAR ──────────────────────────────────────────────
        elif self.phase == 'decel_angular':
            self.stop()
            if abs(self.current_w) < 0.05:
                self.phase = 'done'
                self.get_logger().info('Test complete — plotting results')
                self.plot_results()

        # ── DONE ───────────────────────────────────────────────────────
        elif self.phase == 'done':
            self.stop()

    def plot_results(self):
        times = np.array(self.times)
        vs    = np.array(self.linear_vs)
        ws    = np.array(self.angular_ws)

        if len(times) < 2:
            print('Not enough data to plot')
            return

        # resample to fixed 50ms intervals to eliminate noise spikes
        t_uniform  = np.arange(times[0], times[-1], 0.05)
        vs_uniform = np.interp(t_uniform, times, vs)
        ws_uniform = np.interp(t_uniform, times, ws)

        # smooth velocities with moving average before differentiating
        window = 5
        vs_smooth = np.convolve(vs_uniform, np.ones(window)/window, mode='valid')
        ws_smooth = np.convolve(ws_uniform, np.ones(window)/window, mode='valid')
        t_smooth  = t_uniform[:len(vs_smooth)]

        # compute acceleration on smoothed uniform data
        dt          = 0.05
        linear_acc  = np.diff(vs_smooth) / dt
        angular_acc = np.diff(ws_smooth) / dt
        acc_times   = t_smooth[:-1]

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle('PuzzleBot Velocity and Acceleration Test', fontsize=14)

        # linear velocity
        axes[0,0].plot(t_uniform, vs_uniform, 'b-', linewidth=2, label='measured')
        axes[0,0].plot(t_smooth,  vs_smooth,  'c-', linewidth=1, alpha=0.7, label='smoothed')
        axes[0,0].axhline(y=self.target_v, color='r', linestyle='--',
                        label=f'Target {self.target_v} m/s')
        axes[0,0].set_title('Linear Velocity')
        axes[0,0].set_xlabel('Time (s)')
        axes[0,0].set_ylabel('v (m/s)')
        axes[0,0].legend()
        axes[0,0].grid(True)

        # angular velocity
        axes[0,1].plot(t_uniform, ws_uniform, 'g-', linewidth=2, label='measured')
        axes[0,1].plot(t_smooth,  ws_smooth,  'lime', linewidth=1, alpha=0.7, label='smoothed')
        axes[0,1].axhline(y=self.target_w, color='r', linestyle='--',
                        label=f'Target {self.target_w} rad/s')
        axes[0,1].set_title('Angular Velocity')
        axes[0,1].set_xlabel('Time (s)')
        axes[0,1].set_ylabel('w (rad/s)')
        axes[0,1].legend()
        axes[0,1].grid(True)

        # linear acceleration
        axes[1,0].plot(acc_times, linear_acc, 'b-', linewidth=1.5)
        axes[1,0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[1,0].set_title('Linear Acceleration (smoothed)')
        axes[1,0].set_xlabel('Time (s)')
        axes[1,0].set_ylabel('a (m/s²)')
        axes[1,0].grid(True)

        # angular acceleration
        axes[1,1].plot(acc_times, angular_acc, 'g-', linewidth=1.5)
        axes[1,1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[1,1].set_title('Angular Acceleration (smoothed)')
        axes[1,1].set_xlabel('Time (s)')
        axes[1,1].set_ylabel('α (rad/s²)')
        axes[1,1].grid(True)

        plt.tight_layout()

        # use 95th percentile instead of max to ignore remaining noise spikes
        max_lin_acc  = np.percentile(np.abs(linear_acc),  95)
        max_ang_acc  = np.percentile(np.abs(angular_acc), 95)

        print('\n========== RESULTS ==========')
        print(f'Max linear velocity:      {np.max(np.abs(vs)):.3f} m/s')
        print(f'Max angular velocity:     {np.max(np.abs(ws)):.3f} rad/s')
        print(f'Max linear acceleration:  {max_lin_acc:.3f} m/s²')
        print(f'Max angular acceleration: {max_ang_acc:.3f} rad/s²')
        print('==============================')
        print('Copy these into DWA __init__:')
        print(f'  self.max_v           = {np.max(np.abs(vs)):.2f}')
        print(f'  self.max_w           = {np.max(np.abs(ws)):.2f}')
        print(f'  self.max_linear_acc  = {max_lin_acc:.2f}')
        print(f'  self.max_angular_acc = {max_ang_acc:.2f}')
        print('==============================\n')

        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = VelocityTester()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()