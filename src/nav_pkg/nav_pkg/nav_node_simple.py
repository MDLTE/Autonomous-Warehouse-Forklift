import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from nav_pkg.astar import Astar
from slam_pkg.map_io import load_map, load_corridor_map
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import json
import threading
import sys
import tty
import termios

home          = os.path.expanduser('~')
map_path      = os.path.join(home, 'ros2_ws', 'src', 'slam_pkg', 'maps', 'map')
corridor_path = os.path.join(home, 'ros2_ws', 'src', 'slam_pkg', 'maps', 'corridors')
wp_path       = os.path.join(home, 'ros2_ws', 'src', 'slam_pkg', 'maps',
                              'search_waypoints.json')

# ── sweep angle ───────────────────────────────────────────────────
_S = math.radians(35)   # sweep half-arc

# ── heading sequences per waypoint ───────────────────────────────
# Each entry is a list of (label, world_angle_rad).
# 'aruco' headings are for position confirmation.
# 'sweep' headings are the camera sweep positions.
# Angles follow math convention: 0=east, 90=north, 180=west, -90=south.
_N  =  math.pi / 2          # north  90°
_E  =  0.0                  # east    0°
_S_ = -math.pi / 2          # south -90°
_W  =  math.pi              # west  180°
_NE =  math.pi / 4          # NE    45°
_NW =  3 * math.pi / 4      # NW   135°

HEADING_SEQUENCES = {
    'mission_1': [
        # FOURCONVEYORS: aruco east, sweep north
        [('aruco', _E),
         ('sweep', _N),
         ('sweep', _N - _S),
         ('sweep', _N + _S)],
    ],
    'mission_2': [
        # CENTERRIGHTSHELF-RIGHTSIDE: aruco NE, sweep west
        [('aruco', _NE),
         ('sweep', _W),
         ('sweep', _W - _S),
         ('sweep', _W + _S)],
        # CENTERSHELVES-CENTER: aruco N, sweep east, aruco N, sweep west
        [('aruco', _N),
         ('sweep', _E),
         ('sweep', _E - _S),
         ('sweep', _E + _S),
         ('aruco', _N),
         ('sweep', _W),
         ('sweep', _W - _S),
         ('sweep', _W + _S)],
        # CENTERLEFTSHELF-LEFTSIDE: aruco NW, sweep east
        [('aruco', _NW),
         ('sweep', _E),
         ('sweep', _E - _S),
         ('sweep', _E + _S)],
        # CENTERBOTTOMSHELF-TOPSIDE: aruco north, sweep south
        [('aruco', _N),
         ('sweep', _S_),
         ('sweep', _S_ - _S),
         ('sweep', _S_ + _S)],
        # CENTERBOTTOMSHELF-BOTTOMSIDE: aruco east, sweep north
        [('aruco', _E),
         ('sweep', _N),
         ('sweep', _N - _S),
         ('sweep', _N + _S)],
    ],
}


class NavNodeSimple(Node):

    def __init__(self):
        super().__init__('nav_node_simple')

        # ── map ───────────────────────────────────────────────────
        origin_x = 21
        origin_y = 83
        self.grid = load_map(map_path, origin_x=origin_x, origin_y=origin_y)
        try:
            self.planning_grid = load_corridor_map(
                corridor_path, map_path,
                origin_x=origin_x, origin_y=origin_y)
            self.get_logger().info('Corridor map loaded')
        except Exception:
            self.get_logger().warn('No corridors — using inflated map')
            self.planning_grid = self.grid.inflate_obstacles(robot_radius_m=0.15)
        self.astar = Astar(self.planning_grid)

        try:
            from PIL import Image
            self.corridor_mask = np.array(
                Image.open(corridor_path + '.pgm')) == 255
        except Exception:
            self.corridor_mask = None

        # ── search waypoints ──────────────────────────────────────
        with open(wp_path, 'r') as f:
            self.search_waypoints = json.load(f)
        self.get_logger().info(
            f'Search waypoints loaded — '
            f'M1: {len(self.search_waypoints["mission_1"])} wps  '
            f'M2: {len(self.search_waypoints["mission_2"])} wps')

        # ── path and goal ─────────────────────────────────────────
        self.goal = None
        self.path = None

        # ── Pure Pursuit params ───────────────────────────────────
        self.waypoint_threshold = 0.15
        self.k_v                = 0.3
        self.k_w                = 1.0
        self.slow_down_dist     = 0.40
        self.lookahead_dist     = 0.4
        self.turn_in_place      = math.radians(45)

        self.declare_parameter('max_v', 0.15)
        self.declare_parameter('max_w', 0.30)
        self.max_v = float(self.get_parameter('max_v').value)
        self.max_w = float(self.get_parameter('max_w').value)

        # ── sweep params ──────────────────────────────────────────
        self.declare_parameter('dwell_cycles',   15)   # cycles at each heading (10Hz → 1.5s)
        self.declare_parameter('heading_tol_deg', 3.0) # deg — "close enough" to heading
        self.dwell_cycles  = int(self.get_parameter('dwell_cycles').value)
        self.heading_tol   = math.radians(
            float(self.get_parameter('heading_tol_deg').value))

        # ── localization guard ────────────────────────────────────
        self.declare_parameter('use_loc_guard',       True)
        self.declare_parameter('heading_std_max_deg', 25.0)
        self.declare_parameter('pose_std_max',        1.0)
        self.use_loc_guard   = bool(self.get_parameter('use_loc_guard').value)
        self.heading_std_max = math.radians(
            float(self.get_parameter('heading_std_max_deg').value))
        self.pose_std_max = float(self.get_parameter('pose_std_max').value)
        self.heading_std  = float('inf')
        self.pose_std_xy  = float('inf')
        self.localizing   = False

        self.declare_parameter('debug', False)
        self.debug = bool(self.get_parameter('debug').value)

        # ── robot state ───────────────────────────────────────────
        self.robot_x     = 0.0
        self.robot_y     = 0.0
        self.robot_theta = 0.0
        self.odomReceived = False

        # ── state machine ─────────────────────────────────────────
        # IDLE → FOLLOWING (manual) → REACHED → IDLE
        # IDLE → SEARCHING (search trigger) → SWEEPING → SEARCHING → ... → IDLE
        self.state = 'IDLE'

        # ── search state ──────────────────────────────────────────
        self.search_mission     = None   # 'mission_1' or 'mission_2'
        self.search_wp_idx      = 0
        self.heading_queue      = []     # [(label, angle_rad), ...]
        self.heading_idx        = 0
        self.dwell_counter      = 0

        # ── publishers / subscribers ──────────────────────────────
        self.cmd_pub    = self.create_publisher(Twist,  '/cmd_vel',    10)
        self.status_pub = self.create_publisher(String, '/nav_status', 10)
        self.odom_sub   = self.create_subscription(
            Odometry, '/odom', self.odom_cb, 10)
        self.trigger_sub = self.create_subscription(
            String, '/search_trigger', self.search_trigger_cb, 10)

        self.dt = 0.1
        self.create_timer(self.dt, self.control_loop)

        # ── visualization ─────────────────────────────────────────
        self.viz_lock   = threading.Lock()
        self.viz_needed = True
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('button_press_event', self.on_map_click)
        plt.ion()
        plt.show(block=False)
        plt.pause(0.1)

        self.get_logger().info(
            'nav_node_simple listo — A* + Pure Pursuit + search routing\n'
            '  trigger: ros2 topic pub /search_trigger std_msgs/String '
            '"data: T1"  (or T2)')

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
        self.robot_theta  = self.quaternion_to_yaw(qx, qy, qz, qw)
        self.odomReceived = True
        cov = msg.pose.covariance
        self.pose_std_xy = math.sqrt(max(0.0, cov[0] + cov[7]))
        self.heading_std = math.sqrt(max(0.0, cov[35]))

    def on_map_click(self, event):
        if event.inaxes != self.ax:
            return
        gx = int(event.xdata)
        gy = int(event.ydata)
        wx = (gx - self.grid.center_x) * self.grid.resolution
        wy = (gy - self.grid.center_y) * self.grid.resolution
        self.get_logger().info(f'Meta: ({wx:.2f}, {wy:.2f})')
        self.goal = (wx, wy)
        self._plan_path()

    def search_trigger_cb(self, msg: String):
        trigger = msg.data.strip().upper()
        if trigger not in ('T1', 'T2'):
            self.get_logger().warn(
                f'Unknown trigger: {trigger} — use T1 or T2')
            return
        if self.state not in ('IDLE',):
            self.get_logger().warn(
                f'Trigger ignored — robot is busy ({self.state})')
            return
        mission = 'mission_1' if trigger == 'T1' else 'mission_2'
        self.get_logger().info(
            f'[SEARCH] Trigger {trigger} → {mission} '
            f'({len(self.search_waypoints[mission])} waypoints)')
        self._start_search(mission)

    # ═══════════════════════════════════════════════════════════════
    #  PATH PLANNING
    # ═══════════════════════════════════════════════════════════════

    def _plan_path(self):
        if self.goal is None:
            return
        if not self.odomReceived:
            self.get_logger().warn('Esperando odometria')
            return
        self.path = self.astar.find_path((self.robot_x, self.robot_y), self.goal)
        if self.path is None:
            self.get_logger().warn('No se encontro camino')
            self.state = 'IDLE'
            return
        self.state = 'FOLLOWING'
        self.get_logger().info(f'Camino: {len(self.path)} waypoints')
        self.viz_needed = True

    def _plan_path_for_search(self):
        """Plan path to current search waypoint and enter SEARCHING."""
        wps = self.search_waypoints[self.search_mission]
        if self.search_wp_idx >= len(wps):
            self._search_done()
            return
        wp = wps[self.search_wp_idx]
        self.goal = (wp['x'], wp['y'])
        self.path = self.astar.find_path(
            (self.robot_x, self.robot_y), self.goal)
        if self.path is None:
            self.get_logger().warn(
                f'[SEARCH] No path to wp {self.search_wp_idx} '
                f'({wp["label"]}) — skipping')
            self.search_wp_idx += 1
            self._plan_path_for_search()
            return
        self.state = 'SEARCHING'
        self.get_logger().info(
            f'[SEARCH] Moving to wp {self.search_wp_idx}: '
            f'{wp["label"]} ({wp["x"]:.2f}, {wp["y"]:.2f})')
        self.viz_needed = True

    # ═══════════════════════════════════════════════════════════════
    #  SEARCH ROUTING
    # ═══════════════════════════════════════════════════════════════

    def _start_search(self, mission):
        self.search_mission = mission
        self.search_wp_idx  = 0
        self._plan_path_for_search()

    def _setup_sweep(self):
        """Load heading sequence for the current search waypoint."""
        seqs        = HEADING_SEQUENCES[self.search_mission]
        self.heading_queue = seqs[self.search_wp_idx]
        self.heading_idx   = 0
        self.dwell_counter = 0
        wp = self.search_waypoints[self.search_mission][self.search_wp_idx]
        self.get_logger().info(
            f'[SWEEP] wp {self.search_wp_idx} ({wp["label"]}) — '
            f'{len(self.heading_queue)} headings: '
            + ', '.join(f'{k}({math.degrees(a):.0f}°)'
                        for k, a in self.heading_queue))

    def _advance_search_waypoint(self):
        """Move to the next search waypoint or finish."""
        self.search_wp_idx += 1
        wps = self.search_waypoints[self.search_mission]
        if self.search_wp_idx >= len(wps):
            self._search_done()
        else:
            self._plan_path_for_search()

    def _search_done(self):
        self.get_logger().info(
            f'[SEARCH] Mission {self.search_mission} complete')
        self.publish_status('SEARCH_DONE')
        self.state         = 'IDLE'
        self.search_mission = None
        self.viz_needed    = True

    # ═══════════════════════════════════════════════════════════════
    #  PURE PURSUIT
    # ═══════════════════════════════════════════════════════════════

    def _get_lookahead_point(self):
        L  = self.lookahead_dist
        px = self.robot_x
        py = self.robot_y
        for i, (wx, wy) in enumerate(self.path):
            seg = math.hypot(wx - px, wy - py)
            if seg >= L:
                t  = L / seg
                lx = px + t * (wx - px)
                ly = py + t * (wy - py)
                return lx, ly, max(0, i - 1)
            L -= seg
            px, py = wx, wy
        return self.path[-1][0], self.path[-1][1], len(self.path) - 1

    def follow_waypoint(self):
        if self.path is None or len(self.path) == 0:
            return 0.0, 0.0

        gx, gy       = self.path[-1]
        dist_to_goal = math.hypot(gx - self.robot_x, gy - self.robot_y)
        if dist_to_goal < self.waypoint_threshold:
            self.path = []
            return 0.0, 0.0   # caller decides next state

        lx, ly, prune_idx = self._get_lookahead_point()
        if prune_idx > 0:
            self.path = self.path[prune_idx:]

        dx    = lx - self.robot_x
        dy    = ly - self.robot_y
        alpha = math.atan2(dy, dx) - self.robot_theta
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))

        w = max(-self.max_w, min(self.max_w, self.k_w * alpha))

        if abs(alpha) > self.turn_in_place:
            v = 0.0
        else:
            v = self.k_v * min(1.0, dist_to_goal / self.slow_down_dist)
            v = v * max(0.0, math.cos(alpha))
            v = max(0.0, min(self.max_v, v))

        if self.debug:
            phase = 'TURN ' if abs(alpha) > self.turn_in_place else 'DRIVE'
            self.get_logger().info(
                f'[DBG] {phase} alpha={math.degrees(alpha):.0f} '
                f'v={v:.2f} w={w:.2f} dist={dist_to_goal:.2f}',
                throttle_duration_sec=0.5)

        return v, w

    def rotate_to_heading(self, target_rad):
        """
        Rotate in place toward target_rad (world frame).
        Returns (v=0, w) and True when within heading_tol.
        """
        error = math.atan2(
            math.sin(target_rad - self.robot_theta),
            math.cos(target_rad - self.robot_theta))
        at_heading = abs(error) < self.heading_tol
        w = 0.0 if at_heading else max(
            -self.max_w, min(self.max_w, self.k_w * error))
        return 0.0, w, at_heading

    # ═══════════════════════════════════════════════════════════════
    #  STATE MACHINE
    # ═══════════════════════════════════════════════════════════════

    def control_loop(self):
        if not self.odomReceived:
            return

        # ── localization guard ────────────────────────────────────
        guarded_states = ('FOLLOWING', 'SEARCHING', 'SWEEPING')
        if (self.use_loc_guard and self.state in guarded_states
                and not self.localization_confident()):
            self.publish_cmd(0.0, 0.0)
            if not self.localizing:
                self.localizing = True
                self.publish_status('LOCALIZING')
            self.get_logger().warn(
                f'Pose incierta (sigma_th={math.degrees(self.heading_std):.0f} deg, '
                f'sigma_xy={self.pose_std_xy:.2f} m) - esperando ArUco',
                throttle_duration_sec=2.0)
            self.viz_needed = True
            return
        if self.localizing:
            self.localizing = False
            self.get_logger().info('Pose recuperada - reanudando')

        # ── IDLE ──────────────────────────────────────────────────
        if self.state == 'IDLE':
            self.publish_cmd(0.0, 0.0)

        # ── FOLLOWING (manual map click) ──────────────────────────
        elif self.state == 'FOLLOWING':
            gx, gy = (self.path[-1] if self.path
                      else (self.robot_x, self.robot_y))
            dist   = math.hypot(gx - self.robot_x, gy - self.robot_y)
            if not self.path or dist < self.waypoint_threshold:
                self.get_logger().info('REACHED')
                self.publish_cmd(0.0, 0.0)
                self.publish_status('AREA_REACHED')
                self.state      = 'IDLE'
                self.viz_needed = True
                return
            v, w = self.follow_waypoint()
            self.publish_cmd(v, w)
            self.viz_needed = True

        # ── SEARCHING (moving to search waypoint) ─────────────────
        elif self.state == 'SEARCHING':
            self.get_logger().info('SEARCHING', throttle_duration_sec=2.0)
            v, w = self.follow_waypoint()

            if not self.path:   # follow_waypoint cleared path → arrived
                self.publish_cmd(0.0, 0.0)
                self.get_logger().info(
                    f'[SEARCH] Arrived at wp {self.search_wp_idx} '
                    f'({self.search_waypoints[self.search_mission][self.search_wp_idx]["label"]})')
                self.state = 'SWEEPING'
                self._setup_sweep()
                return

            self.publish_cmd(v, w)
            self.viz_needed = True

        # ── SWEEPING (heading sequence at waypoint) ───────────────
        elif self.state == 'SWEEPING':

            if self.heading_idx >= len(self.heading_queue):
                # all headings done — move to next waypoint
                self.publish_cmd(0.0, 0.0)
                self.get_logger().info(
                    f'[SWEEP] wp {self.search_wp_idx} complete')
                self._advance_search_waypoint()
                return

            kind, target_rad = self.heading_queue[self.heading_idx]

            v, w, at_heading = self.rotate_to_heading(target_rad)
            self.publish_cmd(v, w)

            if at_heading:
                self.dwell_counter += 1
                if self.dwell_counter >= self.dwell_cycles:
                    self.get_logger().info(
                        f'[SWEEP] {kind} {math.degrees(target_rad):.0f}° done '
                        f'(heading {self.heading_idx + 1}/'
                        f'{len(self.heading_queue)})')
                    self.heading_idx  += 1
                    self.dwell_counter = 0
            else:
                self.dwell_counter = 0   # reset dwell if heading is lost

            self.viz_needed = True

        # ── REACHED (handled inline in FOLLOWING above) ───────────

    # ═══════════════════════════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════════════════════════

    def emergency_stop(self):
        self.publish_cmd(0.0, 0.0)
        self.state          = 'IDLE'
        self.path           = None
        self.goal           = None
        self.search_mission = None
        self.heading_queue  = []
        self.viz_needed     = True
        self.get_logger().warn('STOP — motores detenidos')

    def localization_confident(self):
        return (self.heading_std <= self.heading_std_max
                and self.pose_std_xy <= self.pose_std_max)

    def publish_cmd(self, v, w):
        msg = Twist()
        msg.linear.x  = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def publish_status(self, status):
        msg = String()
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
        gray = 1.0 - prob_map
        rgb  = np.stack([gray, gray, gray], axis=-1)
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
            self.ax.scatter(path_gx, path_gy,
                            c='cyan', s=15, alpha=0.9, zorder=3)

        # search waypoints (all of them for current mission)
        if self.search_mission:
            wps = self.search_waypoints[self.search_mission]
            for i, wp in enumerate(wps):
                gx = int(wp['x'] / self.grid.resolution + self.grid.center_x)
                gy = int(wp['y'] / self.grid.resolution + self.grid.center_y)
                color = 'lime' if i == self.search_wp_idx else 'yellow'
                self.ax.scatter(gx, gy, c=color, s=80,
                                marker='D', zorder=4, alpha=0.8)
                self.ax.text(gx + 4, gy + 4, str(i),
                             fontsize=7, color=color, zorder=4)

        robot_gx = int(self.robot_x / self.grid.resolution + self.grid.center_x)
        robot_gy = int(self.robot_y / self.grid.resolution + self.grid.center_y)
        self.ax.scatter(robot_gx, robot_gy,
                        c='red', s=100, marker='*', zorder=5)
        self.ax.arrow(
            robot_gx, robot_gy,
            10 * np.cos(self.robot_theta),
            10 * np.sin(self.robot_theta),
            head_width=4, head_length=4, fc='red', ec='red', zorder=5)

        if self.goal is not None:
            goal_gx = int(self.goal[0] / self.grid.resolution + self.grid.center_x)
            goal_gy = int(self.goal[1] / self.grid.resolution + self.grid.center_y)
            self.ax.scatter(goal_gx, goal_gy,
                            c='blue', s=150, marker='x', zorder=5)

        # sweep heading indicator
        if self.state == 'SWEEPING' and self.heading_idx < len(self.heading_queue):
            kind, target_rad = self.heading_queue[self.heading_idx]
            self.ax.arrow(
                robot_gx, robot_gy,
                14 * math.cos(target_rad),
                14 * math.sin(target_rad),
                head_width=3, head_length=3,
                fc='orange', ec='orange', zorder=6, alpha=0.8)

        state_label = self.state
        if self.state == 'SWEEPING' and self.heading_queue:
            kind, ang = self.heading_queue[
                min(self.heading_idx, len(self.heading_queue) - 1)]
            state_label = (f'SWEEPING [{self.heading_idx + 1}/'
                           f'{len(self.heading_queue)}] '
                           f'{kind} {math.degrees(ang):.0f}°')
        elif self.state == 'SEARCHING' and self.search_mission:
            wp = self.search_waypoints[self.search_mission][self.search_wp_idx]
            state_label = f'SEARCHING → {wp["label"]}'

        self.ax.set_title(
            f'State: {state_label}'
            f'{"  (LOCALIZANDO)" if self.localizing else ""} | '
            f'{len(self.path) if self.path else 0} wp | '
            f'sigma_th={math.degrees(self.heading_std):.0f} deg')
        plt.pause(0.001)

    def safe_visualize(self):
        if self.viz_needed:
            with self.viz_lock:
                self.visualize()
                self.viz_needed = False


def keyboard_listener(node):
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        print('\r[nav_node_simple] S=stop  Q=quit', flush=True)
        while rclpy.ok():
            ch = sys.stdin.read(1)
            if ch in ('s', 'S'):
                node.emergency_stop()
            elif ch in ('q', 'Q', '\x03'):
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main(args=None):
    rclpy.init(args=args)
    node = NavNodeSimple()

    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    kb_thread = threading.Thread(
        target=keyboard_listener, args=(node,), daemon=True)
    kb_thread.start()

    try:
        while rclpy.ok():
            node.safe_visualize()
            plt.pause(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_cmd(0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()