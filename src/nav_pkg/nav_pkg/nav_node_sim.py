import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from nav_pkg.astar import Astar
from nav_pkg.dwa import DWA
from slam_pkg.map_io import load_map, load_corridor_map
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import threading

home          = os.path.expanduser('~')
map_path      = os.path.join(home, 'ros2_ws', 'src', 'slam_pkg', 'maps', 'map')
corridor_path = os.path.join(home, 'ros2_ws', 'src', 'slam_pkg', 'maps', 'corridors')


class DWAFwd(DWA):
    def __init__(self):
        super().__init__()
        self.max_v =  0.30
        self.min_v = -0.30
        self.max_w =  3.84
        self.min_w = -3.84
        self.max_linear_acc  = 0.43
        self.max_angular_acc = 6.92
        self.dt           = 0.1
        self.predict_time = 2.0
        self.v_resolution = 0.05
        self.w_resolution = math.radians(5.0)
        self.to_goal_cost_gain  = 0.2
        self.speed_cost_gain    = 0.5
        self.obstacle_cost_gain = 0.2
        self.robot_radius = 0.13
        self.avoid_max_w = 0.8

    def calc_to_goal_cost(self, trajectory, goal):
        dx          = goal[0] - trajectory[-1, 0]
        dy          = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        effective_yaw = trajectory[-1, 2]
        cost_angle    = error_angle - effective_yaw
        heading_cost  = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))
        start_dist = math.sqrt(
            (trajectory[0, 0] - goal[0])**2 +
            (trajectory[0, 1] - goal[1])**2
        )
        end_dist = math.sqrt(
            (trajectory[-1, 0] - goal[0])**2 +
            (trajectory[-1, 1] - goal[1])**2
        )
        retreat_penalty = max(0.0, end_dist - start_dist) * 2.0
        return heading_cost + retreat_penalty

    def get_obstacle_points(self, robot_x, robot_y, robot_theta, scan_angles, scan_ranges):
        points = []
        for beam_angle, distance in zip(scan_angles, scan_ranges):
            if np.isinf(distance) or distance <= 0 or distance >= 3.5:
                continue
            if distance < 0.15:
                continue
            world_angle = robot_theta + beam_angle
            wx = robot_x + distance * np.cos(world_angle)
            wy = robot_y + distance * np.sin(world_angle)
            points.append((wx, wy))
        return points

    def compute(self, robot_x, robot_y, robot_theta, robot_v, robot_w,
                goal, scan_angles, scan_ranges, avoiding=False):
        obstacles = self.get_obstacle_points(
            robot_x, robot_y, robot_theta, scan_angles, scan_ranges
        )
        if avoiding:
            v_min, v_max = 0.05, self.max_v
            w_min, w_max = -self.avoid_max_w, self.avoid_max_w
        else:
            v_min, v_max, w_min, w_max = self.get_dynamic_window(robot_v, robot_w)

        v_values = np.arange(v_min, v_max, self.v_resolution)
        w_values = np.arange(w_min, w_max, self.w_resolution)

        best_cost = float('inf')
        best_v    = 0.0
        best_w    = 0.0

        for v in v_values:
            for w in w_values:
                trajectory = self.predict_trajectory(
                    robot_x, robot_y, robot_theta, robot_v, robot_w, v, w
                )
                to_goal_cost  = self.to_goal_cost_gain  * self.calc_to_goal_cost(trajectory, goal)
                speed_cost    = self.speed_cost_gain    * self.calc_speed_cost(trajectory)
                obstacle_cost = self.obstacle_cost_gain * self.calc_obstacle_cost(trajectory, obstacles)
                final_cost    = to_goal_cost + speed_cost + obstacle_cost
                if final_cost == float('inf'):
                    continue
                if final_cost < best_cost:
                    best_cost = final_cost
                    best_v    = v
                    best_w    = w

        return best_v, best_w


class NavNode(Node):

    def __init__(self):
        super().__init__('nav_node_sim')

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
        self.dwa   = DWAFwd()

        try:
            from PIL import Image
            self.corridor_mask = np.array(Image.open(corridor_path + '.pgm')) == 255
        except Exception:
            self.corridor_mask = None

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

        self.goal           = None
        self.path           = None
        self.current_wp_idx = 0

        self.waypoint_threshold = 0.15
        self.k_v                = 0.3
        self.k_w                = 0.5
        self.slow_down_dist     = 0.40

        self.obstacle_threshold = 0.40
        self.narrow_trigger_arc = math.radians(15)
        self.wide_exit_arc      = math.radians(60)

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
        self.latest_scan  = None

        self.state         = 'IDLE'
        self.clear_counter = 0
        self.stuck_counter = 0

        self.dist_traveled     = 0.0
        self.min_travel_detect = 0.20

        # ── publishers / subscribers ──────────────────────────────
        self.cmd_pub    = self.create_publisher(Twist,  '/cmd_vel',    10)
        self.status_pub = self.create_publisher(String, '/nav_status', 10)
        self.state_pub  = self.create_publisher(String, '/nav_state',  10)
        self.odom_sub   = self.create_subscription(Odometry,  '/odom', self.odom_cb, 10)
        self.scan_sub   = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.goal_sub      = self.create_subscription(String, '/nav_goal', self.goal_cb, 10)
        self.goal_pose_sub = self.create_subscription(Point,  '/nav_goal_pose', self.goal_pose_cb, 10)

        self.dt = 0.1
        self.create_timer(self.dt, self.control_loop)

        self.viz_lock    = threading.Lock()
        self.viz_counter = 0
        self.viz_needed  = True
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('button_press_event', self.on_map_click)
        plt.ion()
        plt.show(block=False)
        plt.pause(0.1)

        self.get_logger().info('Nav node SIM started — forward drive, A* + DWA, full LiDAR')

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
        dx     = self.robot_x - self.prev_x
        dy     = self.robot_y - self.prev_y
        dtheta = np.arctan2(
            np.sin(self.robot_theta - self.prev_theta),
            np.cos(self.robot_theta - self.prev_theta)
        )
        self.robot_v        = np.sqrt(dx**2 + dy**2) / self.odom_dt
        self.robot_w        = dtheta / self.odom_dt
        self.dist_traveled += math.sqrt(dx**2 + dy**2)
        self.prev_x, self.prev_y, self.prev_theta = (
            self.robot_x, self.robot_y, self.robot_theta
        )

    def scan_cb(self, msg: LaserScan):
        if not self.odomReceived:
            return
        self.latest_scan = msg

    def goal_cb(self, msg: String):
        location_id = msg.data.strip().upper()
        if location_id not in self.locations:
            self.get_logger().warn(
                f'Unknown location ID: {location_id} — '
                f'available: {list(self.locations.keys())}'
            )
            return
        wx, wy = self.locations[location_id]
        self.get_logger().info(f'Received goal: {location_id} → ({wx:.2f}, {wy:.2f})')
        self.goal = (wx, wy)
        self._plan_path()

    def goal_pose_cb(self, msg: Point):
        """Accept raw (x, y) goal from external node via /nav_goal_pose."""
        wx, wy = float(msg.x), float(msg.y)
        self.get_logger().info(f'Goal pose received: ({wx:.2f}, {wy:.2f})')
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
        self.get_logger().info(f'Goal: ({wx:.2f}, {wy:.2f})')
        self.goal = (wx, wy)
        self._plan_path()

    def _plan_path(self):
        if self.goal is None:
            return
        self.path = self.astar.find_path((self.robot_x, self.robot_y), self.goal)
        if self.path is None:
            self.get_logger().warn('No path found')
            self.state = 'IDLE'
            return
        self.current_wp_idx = 0
        self.clear_counter  = 0
        self.stuck_counter  = 0
        self.dist_traveled  = 0.0
        self.state          = 'FOLLOWING'
        self.get_logger().info(f'Path: {len(self.path)} waypoints')
        self.viz_needed = True

    # ═══════════════════════════════════════════════════════════════
    #  OBSTACLE DETECTION
    # ═══════════════════════════════════════════════════════════════

    def check_obstacle_narrow(self, scan_angles, scan_ranges):
        for angle, distance in zip(scan_angles, scan_ranges):
            if abs(angle) > self.narrow_trigger_arc:
                continue
            if np.isinf(distance) or distance <= 0 or distance < 0.15:
                continue
            if distance < self.obstacle_threshold:
                return True
        return False

    def check_obstacle_wide(self, scan_angles, scan_ranges):
        for angle, distance in zip(scan_angles, scan_ranges):
            if abs(angle) > self.wide_exit_arc:
                continue
            if np.isinf(distance) or distance <= 0 or distance < 0.15:
                continue
            if distance < self.obstacle_threshold:
                return True
        return False

    # ═══════════════════════════════════════════════════════════════
    #  WAYPOINT FOLLOWER
    # ═══════════════════════════════════════════════════════════════

    def follow_waypoint(self):
        if self.path is None or len(self.path) == 0:
            return 0.0, 0.0

        target = self.path[0]
        dx     = target[0] - self.robot_x
        dy     = target[1] - self.robot_y

        angle_to_target = math.atan2(dy, dx)
        angular_error   = math.atan2(
            math.sin(angle_to_target - self.robot_theta),
            math.cos(angle_to_target - self.robot_theta)
        )
        dist = math.sqrt(dx**2 + dy**2)

        if dist < self.waypoint_threshold:
            self.path.pop(0)
            self.current_wp_idx = 0
            if len(self.path) == 0:
                self.state = 'REACHED'
                return 0.0, 0.0
            target = self.path[0]
            dx     = target[0] - self.robot_x
            dy     = target[1] - self.robot_y
            dist   = math.sqrt(dx**2 + dy**2)

        turn_factor = max(0.2, 1.0 - abs(angular_error) / math.pi)
        dist_factor = min(1.0, dist / self.slow_down_dist)

        v = self.k_v * turn_factor * dist_factor
        w = self.k_w * angular_error

        v = max(0.0, min(self.dwa.max_v, v))
        w = max(-self.dwa.max_w, min(self.dwa.max_w, w))
        return v, w

    # ═══════════════════════════════════════════════════════════════
    #  DWA
    # ═══════════════════════════════════════════════════════════════

    def run_dwa(self, scan_angles, scan_ranges):
        if self.goal is None:
            return 0.0, 0.0
        return self.dwa.compute(
            self.robot_x, self.robot_y, self.robot_theta,
            self.robot_v, self.robot_w,
            self.goal, scan_angles, scan_ranges,
            avoiding=True
        )

    # ═══════════════════════════════════════════════════════════════
    #  STATE MACHINE
    # ═══════════════════════════════════════════════════════════════

    def control_loop(self):
        if not self.odomReceived or self.latest_scan is None:
            return

        scan_angles_raw = np.arange(
            self.latest_scan.angle_min,
            self.latest_scan.angle_max,
            self.latest_scan.angle_increment
        )
        scan_angles = (scan_angles_raw + math.pi) % (2 * math.pi) - math.pi
        scan_ranges = np.array(self.latest_scan.ranges)

        # publish current state every cycle
        state_msg = String()
        state_msg.data = self.state
        self.state_pub.publish(state_msg)

        # ── IDLE ──────────────────────────────────────────────────
        if self.state == 'IDLE':
            self.publish_cmd(0.0, 0.0)

        # ── FOLLOWING ─────────────────────────────────────────────
        elif self.state == 'FOLLOWING':
            self.get_logger().info('FOLLOWING', throttle_duration_sec=1.0)

            if self.dist_traveled < self.min_travel_detect:
                v, w = self.follow_waypoint()
                self.publish_cmd(v, w)
                return

            if self.check_obstacle_narrow(scan_angles, scan_ranges):
                self.state         = 'AVOIDING'
                self.clear_counter = 0
                self.stuck_counter = 0
                self.get_logger().info('Obstacle detected — switching to DWA')
                return

            v, w = self.follow_waypoint()
            self.publish_cmd(v, w)

            self.viz_counter += 1
            if self.viz_counter % 10 == 0:
                self.viz_needed = True

        # ── AVOIDING ──────────────────────────────────────────────
        elif self.state == 'AVOIDING':
            self.get_logger().info('AVOIDING', throttle_duration_sec=1.0)

            if not self.check_obstacle_wide(scan_angles, scan_ranges):
                self.clear_counter += 1
                if self.clear_counter > 10:
                    self.get_logger().info('Obstacle cleared — resuming')
                    resume_idx          = self._nearest_waypoint_ahead_idx()
                    self.path           = self.path[resume_idx:]
                    self.current_wp_idx = 0
                    self.dist_traveled  = self.min_travel_detect
                    self.state          = 'FOLLOWING'
                    self.clear_counter  = 0
                    self.stuck_counter  = 0
                    self.viz_needed     = True
                return
            else:
                self.clear_counter = 0

            v, w = self.run_dwa(scan_angles, scan_ranges)

            if v == 0.0 and w == 0.0:
                self.stuck_counter += 1
                if self.stuck_counter > 10:
                    self.get_logger().warn('DWA stuck — spinning to escape')
                    self.publish_cmd(0.0, 0.5)
                return

            self.stuck_counter = 0
            self.publish_cmd(v, w)

            self.viz_counter += 1
            if self.viz_counter % 10 == 0:
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
        for i, wp in enumerate(self.path):
            dx          = wp[0] - self.robot_x
            dy          = wp[1] - self.robot_y
            angle_to_wp = math.atan2(dy, dx)
            diff        = abs(math.atan2(
                math.sin(angle_to_wp - self.robot_theta),
                math.cos(angle_to_wp - self.robot_theta)
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

        if self.path is not None:
            path_gx = [int(wx / self.grid.resolution + self.grid.center_x)
                       for wx, wy in self.path]
            path_gy = [int(wy / self.grid.resolution + self.grid.center_y)
                       for wx, wy in self.path]
            self.ax.scatter(path_gx, path_gy, c='cyan', s=15, alpha=0.9, zorder=3)

        robot_gx = int(self.robot_x / self.grid.resolution + self.grid.center_x)
        robot_gy = int(self.robot_y / self.grid.resolution + self.grid.center_y)
        self.ax.scatter(robot_gx, robot_gy, c='red', s=100, marker='*', zorder=5)
        self.ax.arrow(
            robot_gx, robot_gy,
            10 * np.cos(self.robot_theta),
            10 * np.sin(self.robot_theta),
            head_width=4, head_length=4, fc='red', ec='red', zorder=5
        )

        if self.goal is not None:
            goal_gx = int(self.goal[0] / self.grid.resolution + self.grid.center_x)
            goal_gy = int(self.goal[1] / self.grid.resolution + self.grid.center_y)
            self.ax.scatter(goal_gx, goal_gy, c='blue', s=150, marker='x', zorder=5)

        self.ax.set_title(
            f'State: {self.state} | '
            f'{len(self.path) if self.path else 0} waypoints remaining'
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