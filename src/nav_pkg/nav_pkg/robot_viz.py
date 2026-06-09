import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import threading

# ═══════════════════════════════════════════════════════════════════
#  ROBOT VISUALIZER
#  Shows in real time:
#    - Robot position and heading
#    - LiDAR scan points
#    - Detection arcs (front, sides, forklift)
#    - Current nav state
# ═══════════════════════════════════════════════════════════════════

class RobotViz(Node):

    def __init__(self):
        super().__init__('robot_viz')

        # ── robot state ───────────────────────────────────────────
        self.robot_x     = 0.0
        self.robot_y     = 0.0
        self.robot_theta = 0.0
        self.robot_v     = 0.0
        self.robot_w     = 0.0
        self.nav_state   = 'IDLE'
        self.avoid_mode  = 'clear'

        self.prev_x     = 0.0
        self.prev_y     = 0.0
        self.prev_theta = 0.0
        self.odom_dt    = 0.05

        # ── scan data ─────────────────────────────────────────────
        self.scan_angles = None
        self.scan_ranges = None

        # ── arc parameters (must match reactive_avoider) ──────────
        self.blocked_angle = math.radians(50)
        self.travel_arc    = math.radians(135)
        self.max_w         = 3.84

        # 9 arc w values — must match reactive_avoider.arc_ws
        self.arc_ws = [
            -self.max_w,
            -self.max_w * 0.67,
            -self.max_w * 0.33,
            -self.max_w * 0.10,
             0.0,
             self.max_w * 0.10,
             self.max_w * 0.33,
             self.max_w * 0.67,
             self.max_w,
        ]

        # ── chosen velocity from nav_node ─────────────────────────
        self.chosen_v = 0.0
        self.chosen_w = 0.0

        # ── subscribers ───────────────────────────────────────────
        from geometry_msgs.msg import Twist
        self.odom_sub  = self.create_subscription(
            Odometry, '/odom', self.odom_cb, 10)
        self.scan_sub  = self.create_subscription(
            LaserScan, '/scan', self.scan_cb, 10)
        self.state_sub = self.create_subscription(
            String, '/nav_status', self.state_cb, 10)
        self.cmd_sub   = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_cb, 10)

        # ── matplotlib setup ──────────────────────────────────────
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Robot Visualizer')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')

        self.viz_needed = False
        self.viz_lock   = threading.Lock()

        self.get_logger().info('Robot visualizer started')

    # ═══════════════════════════════════════════════════════════════
    #  CALLBACKS
    # ═══════════════════════════════════════════════════════════════

    def odom_cb(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self.robot_theta = self.quaternion_to_yaw(qx, qy, qz, qw)

        dx     = self.robot_x - self.prev_x
        dy     = self.robot_y - self.prev_y
        dtheta = np.arctan2(
            np.sin(self.robot_theta - self.prev_theta),
            np.cos(self.robot_theta - self.prev_theta)
        )
        self.robot_v = np.sqrt(dx**2 + dy**2) / self.odom_dt
        self.robot_w = dtheta / self.odom_dt

        self.prev_x     = self.robot_x
        self.prev_y     = self.robot_y
        self.prev_theta = self.robot_theta
        self.viz_needed = True

    def scan_cb(self, msg: LaserScan):
        self.scan_angles = np.arange(
            msg.angle_min, msg.angle_max, msg.angle_increment
        )
        self.scan_ranges = np.array(msg.ranges)

    def state_cb(self, msg: String):
        self.nav_state = msg.data

    def cmd_cb(self, msg):
        self.chosen_v = msg.linear.x
        self.chosen_w = msg.angular.z

    # ═══════════════════════════════════════════════════════════════
    #  SIMULATION
    # ═══════════════════════════════════════════════════════════════

    def simulate_arc(self, v, w, steps=15, dt=0.1):
        """Simulate one arc trajectory from current robot position."""
        traj = []
        x, y, theta = self.robot_x, self.robot_y, self.robot_theta
        for _ in range(steps):
            theta += w * dt
            theta  = math.atan2(math.sin(theta), math.cos(theta))
            x     += v * math.cos(theta) * dt
            y     += v * math.sin(theta) * dt
            traj.append((x, y))
        return traj

    # ═══════════════════════════════════════════════════════════════
    #  VISUALIZATION
    # ═══════════════════════════════════════════════════════════════

    def draw(self):
        self.ax.cla()
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')

        rx, ry, rt = self.robot_x, self.robot_y, self.robot_theta

        # ── draw arc detection zones ──────────────────────────────
        # forklift zone — red fill
        self._draw_arc_zone(rx, ry, rt,
                            -self.blocked_angle, self.blocked_angle,
                            radius=0.6, color='red', alpha=0.10, label='Forklift')

        # side zones — yellow fill
        self._draw_arc_zone(rx, ry, rt,
                            self.blocked_angle, self.travel_arc,
                            radius=0.5, color='yellow', alpha=0.12, label='Side L')
        self._draw_arc_zone(rx, ry, rt,
                            -self.travel_arc, -self.blocked_angle,
                            radius=0.5, color='yellow', alpha=0.12, label='Side R')

        # front/travel arc — blue fill
        self._draw_arc_zone(rx, ry, rt,
                            self.travel_arc, math.pi,
                            radius=0.7, color='blue', alpha=0.12, label='Travel L')
        self._draw_arc_zone(rx, ry, rt,
                            -math.pi, -self.travel_arc,
                            radius=0.7, color='blue', alpha=0.12, label='Travel R')

        # ── draw LiDAR scan points ────────────────────────────────
        if self.scan_angles is not None and self.scan_ranges is not None:
            scan_xs = []
            scan_ys = []
            for angle, dist in zip(self.scan_angles, self.scan_ranges):
                if np.isinf(dist) or dist <= 0 or dist < 0.15 or dist > 3.5:
                    continue
                world_angle = rt + angle
                scan_xs.append(rx + dist * math.cos(world_angle))
                scan_ys.append(ry + dist * math.sin(world_angle))
            if scan_xs:
                self.ax.scatter(scan_xs, scan_ys,
                                c='black', s=8, alpha=0.6, zorder=3,
                                label='LiDAR')

        # ── draw arc trajectories ─────────────────────────────────
        v_cmd = -max(abs(self.robot_v), 0.10)
        for w_arc in self.arc_ws:
            traj = self.simulate_arc(v_cmd, w_arc)
            xs   = [p[0] for p in traj]
            ys   = [p[1] for p in traj]

            # highlight chosen arc in bright green
            is_chosen = abs(w_arc - self.chosen_w) < 0.30

            if is_chosen:
                color, lw, alpha, zorder = 'lime',      3.0, 1.0, 5
            elif w_arc == 0.0:
                color, lw, alpha, zorder = 'cyan',      1.5, 0.7, 3
            else:
                color, lw, alpha, zorder = 'lightblue', 1.0, 0.4, 2

            self.ax.plot(xs, ys, color=color, linewidth=lw,
                         alpha=alpha, zorder=zorder)
            self.ax.scatter(xs[-1], ys[-1],
                            c=color, s=30 if is_chosen else 15,
                            zorder=zorder)

        # ── draw robot body ───────────────────────────────────────
        # robot as filled circle
        robot_circle = plt.Circle((rx, ry), 0.11,
                                   color='royalblue', zorder=5, alpha=0.9)
        self.ax.add_patch(robot_circle)

        # heading arrow (direction robot faces)
        head_len = 0.18
        self.ax.annotate('',
            xy=(rx + head_len * math.cos(rt),
                ry + head_len * math.sin(rt)),
            xytext=(rx, ry),
            arrowprops=dict(arrowstyle='->', color='white', lw=2),
            zorder=6
        )

        # travel direction arrow (opposite — robot drives backwards)
        travel_theta = rt + math.pi
        self.ax.annotate('',
            xy=(rx + head_len * math.cos(travel_theta),
                ry + head_len * math.sin(travel_theta)),
            xytext=(rx, ry),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            zorder=6
        )

        # ── velocity text ─────────────────────────────────────────
        self.ax.text(rx + 0.15, ry + 0.15,
                     f'v={self.robot_v:.2f}\nw={self.robot_w:.2f}',
                     fontsize=8, color='darkblue', zorder=7)

        # ── center view on robot ──────────────────────────────────
        view = 2.0
        self.ax.set_xlim(rx - view, rx + view)
        self.ax.set_ylim(ry - view, ry + view)

        # ── legend and title ──────────────────────────────────────
        self.ax.set_title(
            f'State: {self.nav_state}  |  mode: {self.avoid_mode}\n'
            f'x={rx:.2f}  y={ry:.2f}  θ={math.degrees(rt):.1f}°  '
            f'v={self.robot_v:.2f} m/s  w={self.robot_w:.2f} rad/s',
            fontsize=9
        )

        # legend entries
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red',      alpha=0.4, lw=8,  label='Forklift zone'),
            Line2D([0], [0], color='yellow',   alpha=0.4, lw=8,  label='Side zones'),
            Line2D([0], [0], color='blue',     alpha=0.4, lw=8,  label='Travel zone'),
            Line2D([0], [0], color='lime',      lw=3,             label='Chosen arc'),
            Line2D([0], [0], color='cyan',      lw=2,             label='Straight arc'),
            Line2D([0], [0], color='lightblue', lw=1,             label='Other arcs'),
            Line2D([0], [0], marker='o', color='white',
                   markerfacecolor='royalblue', markersize=10,    label='Robot'),
            Line2D([0], [0], color='white',    lw=2,             label='Heading →'),
            Line2D([0], [0], color='red',      lw=2,             label='Travel ←'),
        ]
        self.ax.legend(handles=legend_elements,
                       loc='upper right', fontsize=7, framealpha=0.8)

        plt.tight_layout()
        plt.pause(0.001)

    def _draw_arc_zone(self, cx, cy, theta,
                        angle_start, angle_end,
                        radius, color, alpha, label):
        """Draw a filled arc zone in robot frame."""
        n      = 30
        angles = np.linspace(theta + angle_start, theta + angle_end, n)
        xs     = [cx] + [cx + radius * math.cos(a) for a in angles] + [cx]
        ys     = [cy] + [cy + radius * math.sin(a) for a in angles] + [cy]
        self.ax.fill(xs, ys, color=color, alpha=alpha, zorder=1)

    def safe_draw(self):
        if self.viz_needed:
            with self.viz_lock:
                self.draw()
                self.viz_needed = False

    # ═══════════════════════════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════════════════════════

    def quaternion_to_yaw(self, qx, qy, qz, qw):
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)
    node = RobotViz()

    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True
    )
    spin_thread.start()

    plt.ion()
    try:
        while rclpy.ok():
            node.safe_draw()
            plt.pause(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()