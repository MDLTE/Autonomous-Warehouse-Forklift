import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from nav_pkg.astar import Astar
from slam_pkg.map_io import load_map, load_corridor_map
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import threading

home          = os.path.expanduser('~')
map_path      = os.path.join(home, 'ros2_ws', 'src', 'slam_pkg', 'maps', 'map')
corridor_path = os.path.join(home, 'ros2_ws', 'src', 'slam_pkg', 'maps', 'corridors')


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


# ══════════════════════════════════════════════════════════════════
#  LiDAR sector definitions — forward-driving robot, no LiDAR offset
#
#  Gap fix: front ±30° + side ±60° → continuous coverage, no blind zone
#    Front: -30° to +30°
#    Left:   30° to 150°   (center 90° ±60°)
#    Right: 210° to 330°   (center 270° ±60°, = -150° to -30°)
# ══════════════════════════════════════════════════════════════════
SECTOR_FRONT_CENTER =   0.0
SECTOR_FRONT_HW     =  30.0   # was 15° — widened to close blind zone
SECTOR_LEFT_CENTER  =  90.0
SECTOR_LEFT_HW      =  60.0   # was 45° — widened to close blind zone
SECTOR_RIGHT_CENTER = 270.0
SECTOR_RIGHT_HW     =  60.0   # was 45° — widened to close blind zone

# ══════════════════════════════════════════════════════════════════
#  Bug2 params
# ══════════════════════════════════════════════════════════════════
DANGER_DIST = 0.35   # m — was 0.25, detect obstacles earlier
WALL_DIST   = 0.20   # m — desired lateral clearance
KP_WF_LAT   = 1.5   # lateral correction gain
V_WF        = 0.12   # m/s — was 0.08, faster arc around cylinders
OMEGA_MAX   = 0.50   # rad/s — was 0.30, tighter turns around cylinders
V_MAX       = 0.22   # m/s — max linear
MLINE_TOL   = 0.06   # m — perpendicular distance to consider on M-line
MLINE_CLOSER_THRESH = 0.10   # m — must be this much closer than hit dist
HIT_POINT_MIN_DIST  = 0.30   # m — was 0.40, slightly relaxed for cylinders
PATH_CLEAR_ARC_DEG  = 45.0   # wider arc check before exiting BOUNDARY
BOUNDARY_COOLDOWN   = 50     # cycles (50 Hz = 1 s) before re-entering BOUNDARY


class NavNode(Node):

    def __init__(self):
        super().__init__('nav_node_bug2_sim')

        # ── map ───────────────────────────────────────────────────
        origin_x = 14 - 31
        origin_y = 79 - 41

        self.grid = load_map(map_path, origin_x=origin_x, origin_y=origin_y)
        try:
            self.planning_grid = load_corridor_map(
                corridor_path, map_path,
                origin_x=origin_x, origin_y=origin_y
            )
            self.get_logger().info('Corridor map loaded')
        except Exception:
            self.get_logger().warn('No corridors — using inflated map')
            self.planning_grid = self.grid.inflate_obstacles(robot_radius_m=0.15)

        self.astar = Astar(self.planning_grid)

        # ── corridor mask for visualization ───────────────────────
        try:
            from PIL import Image
            self.corridor_mask = np.array(Image.open(corridor_path + '.pgm')) == 255
        except Exception:
            self.corridor_mask = None

        # ── predefined locations ──────────────────────────────────
        self.locations = {
            'A1': (0.50, 0.50),
            'A2': (0.50, 1.50),
            'A3': (0.50, 2.50),
            'A4': (0.50, 3.50),
            'B1': (1.50, 0.50),
            'B2': (1.50, 1.50),
            'B3': (1.50, 2.50),
            'B4': (1.50, 3.50),
            'HOME': (0.35, 0.35),
        }

        # ── path ──────────────────────────────────────────────────
        self.goal           = None
        self.path           = None
        self.current_wp_idx = 0

        # ── waypoint follower params ──────────────────────────────
        self.waypoint_threshold = 0.15
        self.k_v                = 0.15
        self.k_w                = 0.5
        self.slow_down_dist     = 0.40

        # ── LiDAR sectors (updated every scan) ────────────────────
        self.front_dist = math.inf
        self.left_dist  = math.inf
        self.right_dist = math.inf
        self.latest_scan = None

        # ── Bug2 state ────────────────────────────────────────────
        self.wall_side        = 1     # +1 = left wall, -1 = right wall
        self.wall_start_dist  = None  # dist to goal when obstacle was hit
        self.hit_x            = None
        self.hit_y            = None
        self.mline_start      = None  # (x, y) at moment of impact
        self.mline_goal       = None  # (x, y) of current goal

        # ── robot state ───────────────────────────────────────────
        self.initial_x    = 0.35
        self.initial_y    = 0.35
        self.robot_x      = self.initial_x
        self.robot_y      = self.initial_y
        self.robot_theta  = 0.0
        self.prev_x       = 0.0
        self.prev_y       = 0.0
        self.prev_theta   = 0.0
        self.robot_v      = 0.0
        self.robot_w      = 0.0
        self.odomReceived = False
        self.odom_dt      = 0.05
        self.dist_traveled = 0.0
        self.min_travel_detect = 0.20

        # ── state machine — IDLE → FOLLOWING → BOUNDARY → REACHED → IDLE
        self.state = 'IDLE'
        self.boundary_cooldown = 0

        # ── publishers / subscribers ──────────────────────────────
        self.cmd_pub    = self.create_publisher(Twist,  '/cmd_vel',    10)
        self.status_pub = self.create_publisher(String, '/nav_status', 10)
        self.state_pub  = self.create_publisher(String, '/nav_state',  10)
        self.odom_sub   = self.create_subscription(Odometry,  '/odom', self.odom_cb, 10)
        self.scan_sub   = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.goal_sub   = self.create_subscription(String, '/nav_goal', self.goal_cb, 10)

        self.dt = 0.02   # 50 Hz — matches bug2_monitor
        self.create_timer(self.dt, self.control_loop)

        # ── matplotlib ────────────────────────────────────────────
        self.viz_lock    = threading.Lock()
        self.viz_counter = 0
        self.viz_needed  = True
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('button_press_event', self.on_map_click)
        plt.ion()
        plt.show(block=False)
        plt.pause(0.1)

        self.get_logger().info(
            'Nav node Bug2 SIM started — forward drive, A* + Bug2 (bug2_monitor logic)')

    # ═══════════════════════════════════════════════════════════════
    #  CALLBACKS
    # ═══════════════════════════════════════════════════════════════

    def odom_cb(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x + self.initial_x
        self.robot_y = msg.pose.pose.position.y + self.initial_y
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self.robot_theta  = self.quaternion_to_yaw(qx, qy, qz, qw)
        self.odomReceived = True
        dx = self.robot_x - self.prev_x
        dy = self.robot_y - self.prev_y
        dtheta = np.arctan2(
            np.sin(self.robot_theta - self.prev_theta),
            np.cos(self.robot_theta - self.prev_theta)
        )
        self.robot_v        = np.sqrt(dx**2 + dy**2) / self.odom_dt
        self.robot_w        = dtheta / self.odom_dt
        self.dist_traveled += math.sqrt(dx**2 + dy**2)
        self.prev_x, self.prev_y, self.prev_theta = (
            self.robot_x, self.robot_y, self.robot_theta)

    def scan_cb(self, msg: LaserScan):
        if not self.odomReceived:
            return
        self.latest_scan = msg
        self._update_sectors()

    def goal_cb(self, msg: String):
        location_id = msg.data.strip().upper()
        if location_id not in self.locations:
            self.get_logger().warn(
                f'Unknown location: {location_id} — '
                f'available: {list(self.locations.keys())}')
            return
        wx, wy = self.locations[location_id]
        self.get_logger().info(f'Goal: {location_id} → ({wx:.2f}, {wy:.2f})')
        self.goal = (wx, wy)
        self._plan_path()

    # ═══════════════════════════════════════════════════════════════
    #  MAP CLICK → A*
    # ═══════════════════════════════════════════════════════════════

    def on_map_click(self, event):
        if event.inaxes != self.ax:
            return
        gx = int(event.xdata)
        gy = int(event.ydata)
        wx = (gx - self.grid.center_x) * self.grid.resolution
        wy = (gy - self.grid.center_y) * self.grid.resolution
        self.get_logger().info(f'Goal clicked: ({wx:.2f}, {wy:.2f})')
        self.goal = (wx, wy)
        self._plan_path()

    def _plan_path(self):
        if self.goal is None:
            return
        if not self.odomReceived:
            self.get_logger().warn('Waiting for odometry')
            return
        self.path = self.astar.find_path((self.robot_x, self.robot_y), self.goal)
        if self.path is None:
            self.get_logger().warn('No path found')
            self.state = 'IDLE'
            return
        self.boundary_cooldown = 0
        self.current_wp_idx = 0
        self.dist_traveled  = 0.0
        self._reset_bug2()
        self.state = 'FOLLOWING'
        self.get_logger().info(f'Path: {len(self.path)} waypoints')
        self.viz_needed = True

    # ═══════════════════════════════════════════════════════════════
    #  LIDAR SECTORS — from bug2_monitor, no angle offset
    # ═══════════════════════════════════════════════════════════════

    def _get_sector(self, center_deg, half_width_deg):
        if self.latest_scan is None:
            return math.inf
        msg = self.latest_scan
        n   = len(msg.ranges)
        if n == 0:
            return math.inf

        center_rad = math.radians(center_deg)
        hw_rad     = math.radians(half_width_deg)
        range_min  = max(msg.range_min, 0.01)
        range_max  = msg.range_max if msg.range_max > 0 else 12.0

        vals = []
        for k in range(n):
            angle_k = msg.angle_min + k * msg.angle_increment
            if abs(wrap_angle(angle_k - center_rad)) <= hw_rad:
                r = msg.ranges[k]
                if math.isfinite(r) and range_min <= r <= range_max:
                    vals.append(r)
        return min(vals) if vals else math.inf

    def _update_sectors(self):
        self.front_dist = self._get_sector(SECTOR_FRONT_CENTER, SECTOR_FRONT_HW)
        self.left_dist  = self._get_sector(SECTOR_LEFT_CENTER,  SECTOR_LEFT_HW)
        self.right_dist = self._get_sector(SECTOR_RIGHT_CENTER, SECTOR_RIGHT_HW)

    # ═══════════════════════════════════════════════════════════════
    #  BUG2 HELPERS
    # ═══════════════════════════════════════════════════════════════

    def _reset_bug2(self):
        self.wall_side       = 1
        self.wall_start_dist = None
        self.hit_x           = None
        self.hit_y           = None
        self.mline_start     = None
        self.mline_goal      = None

    def _dist_to_goal(self):
        if self.goal is None:
            return math.inf
        return math.hypot(self.robot_x - self.goal[0],
                          self.robot_y - self.goal[1])

    def _on_mline(self):
        """
        True when robot is within MLINE_TOL of the M-line segment
        and between start and goal (not behind start).
        Taken directly from bug2_monitor._on_mline().
        """
        if self.mline_start is None or self.mline_goal is None:
            return False
        sx, sy = self.mline_start
        gx, gy = self.mline_goal
        lx, ly = gx - sx, gy - sy
        length = math.hypot(lx, ly)
        if length < 1e-6:
            return False
        nx, ny = -ly / length, lx / length
        rx, ry = self.robot_x - sx, self.robot_y - sy
        perp   = abs(rx * nx + ry * ny)
        proj   = rx * (lx / length) + ry * (ly / length)
        return perp < MLINE_TOL and 0.0 <= proj <= length

    def _path_clear_wide(self):
        """
        Check ±PATH_CLEAR_ARC_DEG around front for obstacles.
        Stricter than just front_dist — prevents exit when obstacle
        is just outside the ±15° trigger cone.
        """
        return self._get_sector(0.0, PATH_CLEAR_ARC_DEG) > DANGER_DIST

    # ═══════════════════════════════════════════════════════════════
    #  WAYPOINT FOLLOWER — forward drive
    # ═══════════════════════════════════════════════════════════════

    def follow_waypoint(self):
        if self.path is None or len(self.path) == 0:
            return 0.0, 0.0

        target = self.path[0]
        dx     = target[0] - self.robot_x
        dy     = target[1] - self.robot_y
        dist   = math.sqrt(dx**2 + dy**2)

        if dist < self.waypoint_threshold:
            self.path.pop(0)
            if len(self.path) == 0:
                self.state = 'REACHED'
                return 0.0, 0.0
            target = self.path[0]
            dx     = target[0] - self.robot_x
            dy     = target[1] - self.robot_y
            dist   = math.sqrt(dx**2 + dy**2)

        angle_to_target = math.atan2(dy, dx)
        # forward robot — effective heading = robot_theta (no +π)
        angular_error = math.atan2(
            math.sin(angle_to_target - self.robot_theta),
            math.cos(angle_to_target - self.robot_theta)
        )

        turn_factor = max(0.2, 1.0 - abs(angular_error) / math.pi)
        dist_factor = min(1.0, dist / self.slow_down_dist)

        v = self.k_v * turn_factor * dist_factor   # positive — forward
        w = self.k_w * angular_error

        v = max(0.0, min(V_MAX, v))
        w = clamp(w, -OMEGA_MAX, OMEGA_MAX)
        return v, w

    # ═══════════════════════════════════════════════════════════════
    #  BUG2 WALL FOLLOWING — from bug2_monitor._wall_follow_cmd()
    #  Signs are identical to bug2_monitor (forward robot, no flip)
    # ═══════════════════════════════════════════════════════════════

    def follow_boundary(self):
        cmd     = Twist()
        lateral = self.left_dist if self.wall_side == 1 else self.right_dist
        lat_err = lateral - WALL_DIST

        # Sub-case 1: corner — spin in place
        if self.front_dist < DANGER_DIST * 0.7:
            cmd.linear.x  = 0.0
            cmd.angular.z = -self.wall_side * 0.15
            return cmd

        # Sub-case 2: front close — slow and turn
        if self.front_dist < DANGER_DIST * 1.2:
            cmd.linear.x  = 0.04
            cmd.angular.z = -self.wall_side * 0.20
            return cmd

        # Sub-case 3: normal advance with lateral correction
        cmd.linear.x  = V_WF
        cmd.angular.z = clamp(
            self.wall_side * KP_WF_LAT * lat_err,
            -OMEGA_MAX, OMEGA_MAX)
        return cmd

    # ═══════════════════════════════════════════════════════════════
    #  STATE MACHINE
    # ═══════════════════════════════════════════════════════════════

    def control_loop(self):
        if not self.odomReceived or self.latest_scan is None:
            return

        # Publish state every cycle
        state_msg = String()
        state_msg.data = self.state
        self.state_pub.publish(state_msg)

        # ── IDLE ──────────────────────────────────────────────────
        if self.state == 'IDLE':
            self.publish_cmd(0.0, 0.0)

        # ── FOLLOWING ─────────────────────────────────────────────
        elif self.state == 'FOLLOWING':
            self.get_logger().info('FOLLOWING', throttle_duration_sec=1.0)

            # guard — skip obstacle check until moved enough
            if self.dist_traveled < self.min_travel_detect:
                v, w = self.follow_waypoint()
                self.publish_cmd(v, w)
                return

            # decrement cooldown each cycle
            if self.boundary_cooldown > 0:
                self.boundary_cooldown -= 1

            if self.front_dist < DANGER_DIST and self.boundary_cooldown == 0:
                self.wall_side       = 1 if self.left_dist >= self.right_dist else -1
                self.wall_start_dist = self._dist_to_goal()
                self.hit_x           = self.robot_x
                self.hit_y           = self.robot_y
                self.mline_start     = (self.robot_x, self.robot_y)
                self.mline_goal      = self.goal
                side_str = 'left' if self.wall_side == 1 else 'right'
                self.get_logger().info(
                    f'Obstacle (front={self.front_dist:.2f}m) → '
                    f'BOUNDARY ({side_str} wall)')
                self.state = 'BOUNDARY'
                return

            v, w = self.follow_waypoint()
            self.publish_cmd(v, w)

            self.viz_counter += 1
            if self.viz_counter % 20 == 0:
                self.viz_needed = True

        # ── BOUNDARY (Bug2) ───────────────────────────────────────
        elif self.state == 'BOUNDARY':
            self.get_logger().info('BOUNDARY', throttle_duration_sec=1.0)

            dist_now     = self._dist_to_goal()
            on_mline     = self._on_mline()
            closer       = dist_now < (self.wall_start_dist - MLINE_CLOSER_THRESH)
            path_clear   = self._path_clear_wide()   # ±45° arc, stricter than front only
            moved_enough = (
                self.hit_x is not None and
                math.hypot(self.robot_x - self.hit_x,
                           self.robot_y - self.hit_y) > HIT_POINT_MIN_DIST
            )

            self.get_logger().info(
                f'mline={on_mline} closer={closer} clear={path_clear} '
                f'moved={moved_enough} dist={dist_now:.2f} '
                f'start={self.wall_start_dist:.2f} '
                f'F={self.front_dist:.2f} L={self.left_dist:.2f} '
                f'R={self.right_dist:.2f}',
                throttle_duration_sec=0.5)

            if on_mline and closer and path_clear and moved_enough:
                self.get_logger().info('M-line reached — resuming FOLLOWING')
                self._reset_bug2()
                self.boundary_cooldown = BOUNDARY_COOLDOWN
                resume_idx         = self._nearest_waypoint_ahead_idx()
                self.path          = self.path[resume_idx:]
                self.dist_traveled = self.min_travel_detect
                self.state         = 'FOLLOWING'
                return

            cmd = self.follow_boundary()
            self.publish_cmd(cmd.linear.x, cmd.angular.z)

            self.viz_counter += 1
            if self.viz_counter % 20 == 0:
                self.viz_needed = True

        # ── REACHED ───────────────────────────────────────────────
        elif self.state == 'REACHED':
            self.get_logger().info('REACHED')
            self.publish_cmd(0.0, 0.0)
            self.publish_status('AREA_REACHED')
            self.viz_needed = True
            self.state      = 'IDLE'

    # ═══════════════════════════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════════════════════════

    def _nearest_waypoint_ahead_idx(self):
        if not self.path:
            return 0
        # forward robot — travel direction = robot_theta
        for i, wp in enumerate(self.path):
            dx = wp[0] - self.robot_x
            dy = wp[1] - self.robot_y
            diff = abs(math.atan2(
                math.sin(math.atan2(dy, dx) - self.robot_theta),
                math.cos(math.atan2(dy, dx) - self.robot_theta)
            ))
            if diff <= math.pi / 2:
                return i
        return len(self.path) - 1

    def go_to(self, location_id):
        location_id = location_id.strip().upper()
        if location_id not in self.locations:
            self.get_logger().warn(f'Unknown location: {location_id}')
            return False
        wx, wy = self.locations[location_id]
        self.goal = (wx, wy)
        self._plan_path()
        return True

    def publish_cmd(self, v, w):
        msg           = Twist()
        msg.linear.x  = v
        msg.angular.z = w
        self.cmd_pub.publish(msg)

    def publish_status(self, status):
        msg      = String()
        msg.data = status
        self.status_pub.publish(msg)

    def quaternion_to_yaw(self, qx, qy, qz, qw):
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    # ═══════════════════════════════════════════════════════════════
    #  VISUALIZATION
    # ═══════════════════════════════════════════════════════════════

    def visualize(self):
        self.ax.cla()

        prob_map = 1 - (1 / (1 + np.exp(self.grid.grid)))
        gray     = 1.0 - prob_map
        rgb      = np.stack([gray, gray, gray], axis=-1)

        if self.corridor_mask is not None:
            rgb[self.corridor_mask, 0] = 0.0
            rgb[self.corridor_mask, 1] = 0.75
            rgb[self.corridor_mask, 2] = 0.2

        self.ax.imshow(rgb, origin='lower')

        if self.path:
            path_gx = [int(wx / self.grid.resolution + self.grid.center_x)
                       for wx, wy in self.path]
            path_gy = [int(wy / self.grid.resolution + self.grid.center_y)
                       for wx, wy in self.path]
            self.ax.scatter(path_gx, path_gy, c='cyan', s=15, alpha=0.9, zorder=3)

        # M-line — orange dashed while in BOUNDARY
        if self.state == 'BOUNDARY' and self.mline_start and self.mline_goal:
            sx, sy = self.mline_start
            gx, gy = self.mline_goal
            s_gx = int(sx / self.grid.resolution + self.grid.center_x)
            s_gy = int(sy / self.grid.resolution + self.grid.center_y)
            g_gx = int(gx / self.grid.resolution + self.grid.center_x)
            g_gy = int(gy / self.grid.resolution + self.grid.center_y)
            self.ax.plot([s_gx, g_gx], [s_gy, g_gy],
                         color='orange', linewidth=1.5,
                         linestyle='--', alpha=0.8, zorder=3)

        robot_gx = int(self.robot_x / self.grid.resolution + self.grid.center_x)
        robot_gy = int(self.robot_y / self.grid.resolution + self.grid.center_y)
        self.ax.scatter(robot_gx, robot_gy, c='red', s=100, marker='*', zorder=5)
        self.ax.arrow(
            robot_gx, robot_gy,
            10 * np.cos(self.robot_theta),
            10 * np.sin(self.robot_theta),
            head_width=4, head_length=4, fc='red', ec='red', zorder=5)

        if self.goal:
            goal_gx = int(self.goal[0] / self.grid.resolution + self.grid.center_x)
            goal_gy = int(self.goal[1] / self.grid.resolution + self.grid.center_y)
            self.ax.scatter(goal_gx, goal_gy,
                            c='blue', s=150, marker='x', zorder=5)

        self.ax.set_title(
            f'State: {self.state} | '
            f'{len(self.path) if self.path else 0} wps | '
            f'F={self.front_dist:.2f} L={self.left_dist:.2f} R={self.right_dist:.2f}'
        )
        plt.pause(0.001)

    def safe_visualize(self):
        if self.viz_needed:
            with self.viz_lock:
                self.visualize()
                self.viz_needed = False


def main(args=None):
    rclpy.init(args=args)
    node = NavNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            node.safe_visualize()
            plt.pause(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_cmd(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
    