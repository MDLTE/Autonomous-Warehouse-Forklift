import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from nav_pkg.astar import Astar
from nav_pkg.dwa import DWA
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

_S = math.radians(35)
_N  =  math.pi / 2
_E  =  0.0
_S_ = -math.pi / 2
_W  =  math.pi
_NE =  math.pi / 4
_NW =  3 * math.pi / 4

# ── Drop-off waypoints: va aqui, voltea al sur, entra a HANDOFF ───
# triggers D4 / D3 / D2 — misma logica que el viejo "Logo"
# Todos pasan PRIMERO por APPROACH_POINT para forzar la entrada por ese lado.
APPROACH_POINT = (2.88, 1.39)
DROPOFF_GOALS = {
    'D4': (2.94, 0.59),
    'D3': (2.53, 0.61),
    'D2': (2.03, 0.55),
}
LOGO_HEADING = _S_   # sur

HEADING_SEQUENCES = {
    'mission_1': [
        [('aruco', _E), ('sweep', _N), ('sweep', _N - _S), ('sweep', _N + _S)],
    ],
    'mission_2': [
        [('aruco', _NE), ('sweep', _W), ('sweep', _W - _S), ('sweep', _W + _S)],
        [('aruco', _N), ('sweep', _E), ('sweep', _E - _S), ('sweep', _E + _S),
         ('aruco', _N), ('sweep', _W), ('sweep', _W - _S), ('sweep', _W + _S)],
        [('aruco', _NW), ('sweep', _E), ('sweep', _E - _S), ('sweep', _E + _S)],
        [('aruco', _N), ('sweep', _S_), ('sweep', _S_ - _S), ('sweep', _S_ + _S)],
        [('aruco', _E), ('sweep', _N), ('sweep', _N - _S), ('sweep', _N + _S)],
    ],
}


class NavNodeSimple(Node):

    def __init__(self):
        super().__init__('nav_node_simple')

        origin_x = 21
        origin_y = 83
        self.grid = load_map(map_path, origin_x=origin_x, origin_y=origin_y)
        try:
            self.planning_grid = load_corridor_map(
                corridor_path, map_path, origin_x=origin_x, origin_y=origin_y)
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

        self.dwa = DWA()

        with open(wp_path, 'r') as f:
            self.search_waypoints = json.load(f)
        self.get_logger().info(
            f'Search waypoints — M1:{len(self.search_waypoints["mission_1"])} '
            f'M2:{len(self.search_waypoints["mission_2"])}')

        self.goal = None
        self.path = None

        self.waypoint_threshold = 0.15
        self.k_v                = 0.15
        self.k_w                = 1.0
        self.slow_down_dist     = 0.40
        self.lookahead_dist     = 0.4
        self.turn_in_place      = math.radians(45)

        self.declare_parameter('max_v', 0.15)
        self.declare_parameter('max_w', 0.30)
        self.max_v = float(self.get_parameter('max_v').value)
        self.max_w = float(self.get_parameter('max_w').value)
        self.dwa.max_v       = self.max_v
        self.dwa.min_v       = -self.max_v
        self.dwa.max_w       = self.max_w
        self.dwa.min_w       = -self.max_w
        self.dwa.avoid_max_w = self.max_w

        self.declare_parameter('front_detect_arc_deg',  13.0)
        self.declare_parameter('front_detect_dist',      0.65)
        self.declare_parameter('clear_arc_deg',          35.0)
        self.declare_parameter('avoid_commit_cycles',    40)
        self.declare_parameter('clear_resume',           15)
        self.declare_parameter('min_obstacle_dist',      0.23)
        self.front_detect_arc    = math.radians(
            float(self.get_parameter('front_detect_arc_deg').value))
        self.front_detect_dist   = float(self.get_parameter('front_detect_dist').value)
        self.clear_arc           = math.radians(
            float(self.get_parameter('clear_arc_deg').value))
        self.avoid_commit_cycles = int(self.get_parameter('avoid_commit_cycles').value)
        self.clear_resume        = int(self.get_parameter('clear_resume').value)
        self.min_obstacle_dist   = float(self.get_parameter('min_obstacle_dist').value)

        self.declare_parameter('clear_dist_factor', 1.5)
        self.clear_dist_factor = float(self.get_parameter('clear_dist_factor').value)

        self.declare_parameter('dwell_cycles',    15)
        self.declare_parameter('heading_tol_deg',  3.0)
        self.dwell_cycles = int(self.get_parameter('dwell_cycles').value)
        self.heading_tol  = math.radians(
            float(self.get_parameter('heading_tol_deg').value))

        self.declare_parameter('use_loc_guard',       True)
        self.declare_parameter('heading_std_max_deg', 25.0)
        self.declare_parameter('pose_std_max',         1.0)
        self.use_loc_guard   = bool(self.get_parameter('use_loc_guard').value)
        self.heading_std_max = math.radians(
            float(self.get_parameter('heading_std_max_deg').value))
        self.pose_std_max = float(self.get_parameter('pose_std_max').value)
        self.heading_std  = float('inf')
        self.pose_std_xy  = float('inf')
        self.localizing   = False

        self.declare_parameter('debug', False)
        self.debug = bool(self.get_parameter('debug').value)

        self.robot_x      = 0.0
        self.robot_y      = 0.0
        self.robot_theta  = 0.0
        self.odomReceived = False
        self.scan_angles  = None
        self.scan_ranges  = None

        self.state        = 'IDLE'
        self.resume_state = 'FOLLOWING'
        self.logo_reached_counter = 0

        self.clear_counter = 0
        self.stuck_counter = 0
        self.avoid_cycles  = 0
        self.avoid_count   = 0

        self.search_mission = None
        self.search_wp_idx  = 0
        self.heading_queue  = []
        self.heading_idx    = 0
        self.dwell_counter  = 0
        self.qr_confirmed   = False
        self.handoff_done   = False
        self.dropoff_label  = ''
        self.dropoff_final  = None
        self.dropoff_stage  = 'approach'

        # status republish
        self.last_status        = ''
        self.status_republish_t = 0.0

        self.cmd_pub = self.create_publisher(Twist, '/nav/cmd_vel', 10)
        self.status_pub  = self.create_publisher(String, '/nav_status', 10)
        self.odom_sub    = self.create_subscription(
            Odometry, '/odom', self.odom_cb, 10)
        self.scan_sub    = self.create_subscription(
            LaserScan, '/scan', self.scan_cb, 10)
        self.trigger_sub = self.create_subscription(
            String, '/search_trigger', self.search_trigger_cb, 10)
        self.qr_sub      = self.create_subscription(
            Bool, '/qr_detected', self.qr_cb, 10)
        self.logo_sub      = self.create_subscription(
            Bool, '/logo_detected', self.qr_cb, 10)

        self.create_subscription(String, '/fsm_cmd', self._cb_fsm_cmd, 10)
        self._task_status_pub = self.create_publisher(String, '/task_status', 10)

        self.dt = 0.1
        self.create_timer(self.dt, self.control_loop)

        self.viz_lock   = threading.Lock()
        self.viz_needed = True
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('button_press_event', self.on_map_click)
        plt.ion()
        plt.show(block=False)
        plt.pause(0.1)

        self.get_logger().info(
            'nav_node_simple listo — A* + Pure Pursuit + DWA + search + Logo\n'
            '  /search_trigger: T1 | T2 | D4 | D3 | D2\n'
            '  /qr_detected: true para detener sweep')

    # ── CALLBACKS ──────────────────────────────────────────────────

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

    def scan_cb(self, msg: LaserScan):
        n   = len(msg.ranges)
        ang = msg.angle_min + math.pi + np.arange(n) * msg.angle_increment
        self.scan_angles = np.arctan2(np.sin(ang), np.cos(ang))
        self.scan_ranges = np.array(msg.ranges, dtype=float)
        self.viz_needed  = True

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
    def _cb_fsm_cmd(self, msg):
        cmd = msg.data.strip()
        up  = cmd.upper()
        if up == 'T1':
            if self.state != 'IDLE':
                self.get_logger().warn(f'[FSM_CMD] T1 ignorado — ocupado ({self.state})')
                return
            self.get_logger().info('[FSM_CMD] T1 → mission_1')
            self._start_search('mission_1')
        elif up == 'T2':
            if self.state != 'IDLE':
                self.get_logger().warn(f'[FSM_CMD] T2 ignorado — ocupado ({self.state})')
                return
            self.get_logger().info('[FSM_CMD] T2 → mission_2')
            self._start_search('mission_2')
        elif up.startswith('T6:'):
            empresa = cmd.split(':', 1)[1].strip()
            dropoff = self._empresa_to_dropoff(empresa)
            if self.state != 'IDLE':
                self.get_logger().warn(f'[FSM_CMD] T6 ignorado — ocupado ({self.state})')
                return
            self.get_logger().info(f'[FSM_CMD] T6 → dropoff {dropoff} ({empresa})')
            self._start_dropoff(dropoff)
        elif up == 'STOP':
            self.get_logger().info('[FSM_CMD] STOP → emergency_stop')
            self.emergency_stop()
            self._pub_task_status('FAILED', 'STOP')

    def _empresa_to_dropoff(self, empresa: str) -> str:
        """Mapea nombre de empresa a puerta D4/D3/D2."""
        m = {
            'Emezon': 'D4', 'emezon': 'D4', 'EMEZON': 'D4',
            'Walmar': 'D3', 'walmar': 'D3', 'WALMAR': 'D3',
            'Popsi':  'D2', 'popsi':  'D2', 'POPSI':  'D2',
        }
        return m.get(empresa, 'D4')  # default D4 si no reconoce

    def _pub_task_status(self, status: str, data: str = ''):
        msg = String()
        msg.data = f'NAV:{status}:{data}'
        self._task_status_pub.publish(msg)
        self.get_logger().info(f'[task_status] NAV:{status}:{data}')
    def search_trigger_cb(self, msg: String):
        trigger = msg.data.strip()
        up = trigger.upper()
        if self.state != 'IDLE':
            self.get_logger().warn(f'Trigger ignorado — ocupado ({self.state})')
            return
        if up == 'T1':
            self._start_search('mission_1')
        elif up == 'T2':
            self._start_search('mission_2')
        elif up in DROPOFF_GOALS:
            self._start_dropoff(up)
        else:
            self.get_logger().warn(f'Trigger desconocido: {trigger}')

    def qr_cb(self, msg: Bool):
        """QR confirmado por nodo externo. Solo actua en SWEEPING."""
        if msg.data and self.state == 'SWEEPING':
            self.qr_confirmed = True
            self.get_logger().info('[QR] Confirmacion recibida!')



    # ── STATUS ─────────────────────────────────────────────────────

    def publish_status(self, status):
        """Publica en transicion + republica cada ~1s para los que lleguen tarde."""
        now = self.get_clock().now().nanoseconds * 1e-9
        if status != self.last_status or (now - self.status_republish_t) > 1.0:
            msg = String()
            msg.data = status
            self.status_pub.publish(msg)
            self.last_status        = status
            self.status_republish_t = now

    # ── OBSTACLE DETECTION ─────────────────────────────────────────

    def _obstacle_in_arc(self, arc, dist):
        if self.scan_angles is None:
            return False, None
        a = self.scan_angles
        r = self.scan_ranges
        m = ((np.abs(a) <= arc) & np.isfinite(r)
             & (r > self.min_obstacle_dist) & (r < dist))
        if not np.any(m):
            return False, None
        idx = np.where(m)[0]
        nearest = idx[np.argmin(r[idx])]
        return True, (math.degrees(a[nearest]), float(r[nearest]))

    # ── PATH PLANNING ──────────────────────────────────────────────

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
        self.resume_state  = 'FOLLOWING'
        self.clear_counter = 0
        self.stuck_counter = 0
        self.avoid_cycles  = 0
        self.state = 'FOLLOWING'
        self.get_logger().info(f'Camino: {len(self.path)} waypoints')
        self.viz_needed = True

    def _prune_behind(self, path):
        while len(path) > 1:
            dx  = path[0][0] - self.robot_x
            dy  = path[0][1] - self.robot_y
            dot = (math.cos(self.robot_theta) * dx
                   + math.sin(self.robot_theta) * dy)
            if dot < 0:
                path.pop(0)
            else:
                break
        return path

    def _plan_path_for_search(self):
        wps = self.search_waypoints[self.search_mission]
        if self.search_wp_idx >= len(wps):
            self._search_done()
            return
        wp   = wps[self.search_wp_idx]
        self.goal = (wp['x'], wp['y'])
        self.path = self.astar.find_path(
            (self.robot_x, self.robot_y), self.goal)
        if self.path is None:
            self.get_logger().warn(
                f'[SEARCH] Sin camino a wp {self.search_wp_idx} '
                f'({wp["label"]}) — saltando')
            self.search_wp_idx += 1
            self._plan_path_for_search()
            return
        self.path = self._prune_behind(self.path)
        self.resume_state  = 'SEARCHING'
        self.clear_counter = 0
        self.stuck_counter = 0
        self.avoid_cycles  = 0
        self.state = 'SEARCHING'
        self.get_logger().info(
            f'[SEARCH] Rumbo a wp {self.search_wp_idx}: '
            f'{wp["label"]} ({wp["x"]:.2f}, {wp["y"]:.2f})')
        self.viz_needed = True

    def _replan_to_goal(self):
        """OPCION A: replantea A* desde la posicion actual hacia la meta."""
        if self.goal is None:
            return False
        new_path = self.astar.find_path(
            (self.robot_x, self.robot_y), self.goal)
        if new_path is None:
            self.get_logger().warn('[REPLAN] A* no encontro camino nuevo')
            return False
        self.path = self._prune_behind(new_path)
        self.get_logger().info(
            f'[REPLAN] Camino nuevo: {len(self.path)} waypoints')
        return True

    # ── DROP-OFF (D4 / D3 / D2) ────────────────────────────────────

    def _start_dropoff(self, label):
        """D4/D3/D2: PRIMERO va a APPROACH_POINT, luego al drop-off,
        voltea al sur, entra a HANDOFF."""
        self.dropoff_label = label
        self.dropoff_final = DROPOFF_GOALS[label]
        self.dropoff_stage = 'approach'   # approach → final
        self.goal = APPROACH_POINT
        self.path = self.astar.find_path(
            (self.robot_x, self.robot_y), self.goal)
        if self.path is None:
            self.get_logger().warn(f'[{label}] Sin camino al punto de aproximacion')
            self.state = 'IDLE'
            return
        self.path = self._prune_behind(self.path)
        self.resume_state  = 'LOGO_GOING'
        self.clear_counter = 0
        self.stuck_counter = 0
        self.avoid_cycles  = 0
        self.state = 'LOGO_GOING'
        self.get_logger().info(
            f'[{label}] Rumbo al punto de aproximacion '
            f'({APPROACH_POINT[0]:.2f}, {APPROACH_POINT[1]:.2f})')
        self.viz_needed = True

    # ── SEARCH ROUTING ─────────────────────────────────────────────

    def _start_search(self, mission):
        self.search_mission = mission
        self.search_wp_idx  = 0
        self.get_logger().info(
            f'[SEARCH] {mission} ({len(self.search_waypoints[mission])} wps)')
        self._plan_path_for_search()

    def _setup_sweep(self):
        seqs = HEADING_SEQUENCES[self.search_mission]
        self.heading_queue = seqs[self.search_wp_idx]
        self.heading_idx   = 0
        self.dwell_counter = 0
        self.qr_confirmed  = False
        wp = self.search_waypoints[self.search_mission][self.search_wp_idx]
        self.get_logger().info(
            f'[SWEEP] wp {self.search_wp_idx} ({wp["label"]}) — '
            f'{len(self.heading_queue)} headings')

    def _advance_search_waypoint(self):
        self.search_wp_idx += 1
        wps = self.search_waypoints[self.search_mission]
        if self.search_wp_idx >= len(wps):
            self._search_done()
        else:
            self._plan_path_for_search()

    def _search_done(self):
        self.get_logger().info(f'[SEARCH] Mision {self.search_mission} completa')
        self.publish_status('SEARCH_DONE')
        self.state          = 'IDLE'
        self.search_mission = None
        self.viz_needed     = True

    def _qr_found_handoff(self):
        """QR detectado en sweep — para en seco y cede control (HANDOFF)."""
        self.publish_cmd(0.0, 0.0)   # un solo stop
        wp = self.search_waypoints[self.search_mission][self.search_wp_idx]
        self.get_logger().info(
            f'[QR] Detectado en wp {self.search_wp_idx} ({wp["label"]}) '
            f'— HANDOFF, cediendo control')
        self.publish_status('QR_FOUND')
        self._pub_task_status('QR_FOUND')
        self.search_mission = None
        self.qr_confirmed   = False
        self.heading_queue  = []
        self.state          = 'IDLE'
        self.viz_needed     = True

    # ── PURE PURSUIT ───────────────────────────────────────────────

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
            return 0.0, 0.0
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
        return v, w

    def rotate_to_heading(self, target_rad):
        error = math.atan2(
            math.sin(target_rad - self.robot_theta),
            math.cos(target_rad - self.robot_theta))
        at_heading = abs(error) < self.heading_tol
        w = 0.0 if at_heading else max(
            -self.max_w, min(self.max_w, self.k_w * error))
        return 0.0, w, at_heading

    # ── STATE MACHINE ──────────────────────────────────────────────

    def control_loop(self):
        if not self.odomReceived:
            return

        guarded = ('FOLLOWING', 'SEARCHING', 'SWEEPING', 'AVOIDING',
                   'LOGO_GOING', 'LOGO_APPROACH_TURN')
        if (self.use_loc_guard and self.state in guarded
                and not self.localization_confident()):
            self.publish_cmd(0.0, 0.0)
            if not self.localizing:
                self.localizing = True
            self.publish_status('LOCALIZING')
            self.get_logger().warn(
                f'Pose incierta — sigma_th='
                f'{math.degrees(self.heading_std):.0f}deg '
                f'sigma_xy={self.pose_std_xy:.2f}m',
                throttle_duration_sec=2.0)
            self.viz_needed = True
            return
        if self.localizing:
            self.localizing = False
            self.get_logger().info('Pose recuperada - reanudando')

        # ── IDLE ──────────────────────────────────────────────────
        if self.state == 'IDLE':
            self.publish_cmd(0.0, 0.0)
            self.publish_status('IDLE')

        # ── FOLLOWING ─────────────────────────────────────────────
        elif self.state == 'FOLLOWING':
            self.publish_status('FOLLOWING')
            found, info = self._obstacle_in_arc(
                self.front_detect_arc, self.front_detect_dist)
            if found:
                ang, dist = info
                self.avoid_count += 1
                self.get_logger().warn(
                    f'[AVOID#{self.avoid_count}] obstaculo {ang:+.0f}°/{dist:.2f}m')
                self.resume_state  = 'FOLLOWING'
                self.state         = 'AVOIDING'
                self.clear_counter = 0
                self.stuck_counter = 0
                self.avoid_cycles  = 0
                return
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

        # ── SEARCHING ─────────────────────────────────────────────
        elif self.state == 'SEARCHING':
            wp = self.search_waypoints[self.search_mission][self.search_wp_idx]
            self.publish_status(f'SEARCHING:{wp["label"]}')
            found, info = self._obstacle_in_arc(
                self.front_detect_arc, self.front_detect_dist)
            if found:
                ang, dist = info
                self.avoid_count += 1
                self.get_logger().warn(
                    f'[AVOID#{self.avoid_count}] obstaculo {ang:+.0f}°/{dist:.2f}m '
                    f'(en busqueda)')
                self.resume_state  = 'SEARCHING'
                self.state         = 'AVOIDING'
                self.clear_counter = 0
                self.stuck_counter = 0
                self.avoid_cycles  = 0
                return
            v, w = self.follow_waypoint()
            if not self.path:
                self.publish_cmd(0.0, 0.0)
                self.get_logger().info(
                    f'[SEARCH] Llegue a wp {self.search_wp_idx} ({wp["label"]})')
                self.state = 'SWEEPING'
                self._setup_sweep()
                return
            self.publish_cmd(v, w)
            self.viz_needed = True

        # ── AVOIDING ──────────────────────────────────────────────
        elif self.state == 'AVOIDING':
            self.publish_status('AVOIDING')
            self.avoid_cycles += 1
            clear_check_dist = self.front_detect_dist * self.clear_dist_factor
            clear, _ = self._obstacle_in_arc(self.clear_arc, clear_check_dist)
            if self.avoid_cycles > self.avoid_commit_cycles and not clear:
                self.clear_counter += 1
                if self.clear_counter > self.clear_resume:
                    self.get_logger().info(
                        f'[AVOID#{self.avoid_count}] despejado — '
                        f'replaneando hacia {self.resume_state}')
                    if self._replan_to_goal():
                        self.state = self.resume_state
                    else:
                        self.clear_counter = 0
                        return
                    self.clear_counter = 0
                    self.stuck_counter = 0
                    return
            else:
                self.clear_counter = 0

            v, w = self.dwa.compute(
                self.robot_x, self.robot_y, self.robot_theta,
                self.max_v, 0.0, self.goal,
                self.scan_angles, self.scan_ranges, avoiding=True)

            if self.debug:
                self.get_logger().info(
                    f'[AVOID#{self.avoid_count}] cyc={self.avoid_cycles} '
                    f'v={v:.2f} w={math.degrees(w):+.0f}° '
                    f'nobs={self.dwa.last_n_obs} '
                    f'blocked={int(self.dwa.last_all_blocked)}',
                    throttle_duration_sec=0.3)

            if v == 0.0 and w == 0.0:
                self.stuck_counter += 1
                if self.stuck_counter > 10:
                    self.publish_cmd(0.0, self.max_w)
                    return
            else:
                self.stuck_counter = 0
            self.publish_cmd(v, w)
            self.viz_needed = True

        # ── SWEEPING ──────────────────────────────────────────────
        elif self.state == 'SWEEPING':
            # QR detectado — para en seco y HANDOFF
            if self.qr_confirmed:
                self._qr_found_handoff()
                return

            wp = self.search_waypoints[self.search_mission][self.search_wp_idx]
            self.publish_status(f'SWEEPING:{wp["label"]}')

            if self.heading_idx >= len(self.heading_queue):
                self.publish_cmd(0.0, 0.0)
                self.get_logger().info(f'[SWEEP] wp {self.search_wp_idx} completo')
                self._advance_search_waypoint()
                return
            kind, target_rad = self.heading_queue[self.heading_idx]
            v, w, at_heading = self.rotate_to_heading(target_rad)
            self.publish_cmd(v, w)
            if at_heading:
                self.dwell_counter += 1
                if self.dwell_counter >= self.dwell_cycles:
                    self.get_logger().info(
                        f'[SWEEP] {kind} {math.degrees(target_rad):.0f}° '
                        f'({self.heading_idx + 1}/{len(self.heading_queue)})')
                    self.heading_idx  += 1
                    self.dwell_counter = 0
            else:
                self.dwell_counter = 0
            self.viz_needed = True

        # ── LOGO_GOING (viaja al drop-off, luego voltea al sur) ───
        elif self.state == 'LOGO_GOING':
            self.publish_status(f'LOGO_GOING:{self.dropoff_label}')
            found, info = self._obstacle_in_arc(
                self.front_detect_arc, self.front_detect_dist)
            if found:
                ang, dist = info
                self.avoid_count += 1
                self.get_logger().warn(
                    f'[AVOID#{self.avoid_count}] obstaculo {ang:+.0f}°/{dist:.2f}m '
                    f'({self.dropoff_label})')
                self.resume_state  = 'LOGO_GOING'
                self.state         = 'AVOIDING'
                self.clear_counter = 0
                self.stuck_counter = 0
                self.avoid_cycles  = 0
                return
            v, w = self.follow_waypoint()
            if not self.path:
                self.publish_cmd(0.0, 0.0)
                if self.dropoff_stage == 'approach':
                    # llego al punto de aproximacion — voltea al sur ANTES de seguir
                    self.get_logger().info(
                        f'[{self.dropoff_label}] En aproximacion — '
                        f'volteando al sur antes de seguir')
                    self.state = 'LOGO_APPROACH_TURN'
                    return
                else:
                    # llego al drop-off final — voltea al sur
                    self.get_logger().info(
                        f'[{self.dropoff_label}] Llegue al final — volteando al sur')
                    self.state = 'LOGO_TURN'
                    return
            self.publish_cmd(v, w)
            self.viz_needed = True

        # ── LOGO_APPROACH_TURN (voltea al sur en el approach) ─────
        elif self.state == 'LOGO_APPROACH_TURN':
            self.publish_status(f'LOGO_APPROACH_TURN:{self.dropoff_label}')
            v, w, at_heading = self.rotate_to_heading(LOGO_HEADING)
            self.publish_cmd(v, w)
            if at_heading:
                self.publish_cmd(0.0, 0.0)
                # ya orientado al sur — ahora replanea al drop-off final
                self.dropoff_stage = 'final'
                self.goal = self.dropoff_final
                new_path = self.astar.find_path(
                    (self.robot_x, self.robot_y), self.goal)
                if new_path is None:
                    self.get_logger().warn(
                        f'[{self.dropoff_label}] Sin camino al drop-off final')
                    self.state = 'IDLE'
                    return
                self.path = self._prune_behind(new_path)
                self.get_logger().info(
                    f'[{self.dropoff_label}] Orientado al sur — rumbo al final '
                    f'({self.goal[0]:.2f}, {self.goal[1]:.2f})')
                self.state = 'LOGO_GOING'
            self.viz_needed = True

        # ── LOGO_TURN (voltea al sur, luego HANDOFF) ──────────────
        elif self.state == 'LOGO_TURN':
            self.publish_status(f'LOGO_TURN:{self.dropoff_label}')
            v, w, at_heading = self.rotate_to_heading(LOGO_HEADING)
            self.publish_cmd(v, w)
            if at_heading:
                self.publish_cmd(0.0, 0.0)
                self.get_logger().info(
                    f'[{self.dropoff_label}] Orientado al sur — HANDOFF')
                self.publish_status(f'LOGO_REACHED:{self.dropoff_label}')
                self.handoff_done = False
                self.logo_reached_counter = 20
                self.state = 'LOGO_REACHED_WAIT'
                self.viz_needed = True
            self.viz_needed = True

        # ── LOGO_REACHED_WAIT ─────────────────────────────
        elif self.state == 'LOGO_REACHED_WAIT':

            self.publish_cmd(0.0, 0.0)

            self.publish_status(
                f'LOGO_REACHED:{self.dropoff_label}'
            )

            self.logo_reached_counter -= 1

            if self.logo_reached_counter <= 0:
                self.get_logger().info(
                    f'[{self.dropoff_label}] Entrando a HANDOFF'
                )

                self._pub_task_status('LOGO_REACHED')
                self.state = 'IDLE'
                return

            self.viz_needed = True

        
        

    # ── HELPERS ────────────────────────────────────────────────────

    def emergency_stop(self):
        self.publish_cmd(0.0, 0.0)
        self.state          = 'IDLE'
        self.path           = None
        self.goal           = None
        self.search_mission = None
        self.heading_queue  = []
        self.qr_confirmed   = False
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

    def quaternion_to_yaw(self, qx, qy, qz, qw):
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    # ── VISUALIZATION ──────────────────────────────────────────────

    def _scan_points(self):
        obs = []
        near = []
        if self.scan_angles is None or not self.odomReceived:
            return obs, near
        for beam, d in zip(self.scan_angles, self.scan_ranges):
            if not math.isfinite(d) or d <= 0 or d >= 5.0:
                continue
            wx = self.robot_x + d * math.cos(self.robot_theta + beam)
            wy = self.robot_y + d * math.sin(self.robot_theta + beam)
            if d < self.min_obstacle_dist:
                near.append((wx, wy))
            else:
                obs.append((wx, wy))
        return obs, near

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

        res = self.grid.resolution
        cx  = self.grid.center_x
        cy  = self.grid.center_y

        def to_grid(pts):
            if not pts:
                return [], []
            return ([int(wx / res + cx) for wx, wy in pts],
                    [int(wy / res + cy) for wx, wy in pts])

        obs, near = self._scan_points()
        gxs, gys = to_grid(near)
        if gxs:
            self.ax.scatter(gxs, gys, c='#888888', s=3, alpha=0.4, zorder=2)
        gxs, gys = to_grid(obs)
        if gxs:
            self.ax.scatter(gxs, gys, c='red', s=5, alpha=0.7, zorder=3,
                            label=f'lidar ({len(obs)})')

        if self.path:
            path_gx = [int(wx / res + cx) for wx, wy in self.path]
            path_gy = [int(wy / res + cy) for wx, wy in self.path]
            self.ax.scatter(path_gx, path_gy, c='cyan', s=15, alpha=0.9, zorder=4)

        if self.search_mission:
            wps = self.search_waypoints[self.search_mission]
            for i, wp in enumerate(wps):
                gx = int(wp['x'] / res + cx)
                gy = int(wp['y'] / res + cy)
                color = 'lime' if i == self.search_wp_idx else 'yellow'
                self.ax.scatter(gx, gy, c=color, s=80, marker='D',
                                zorder=5, alpha=0.8)
                self.ax.text(gx + 4, gy + 4, str(i), fontsize=7,
                             color=color, zorder=5)

        robot_gx = int(self.robot_x / res + cx)
        robot_gy = int(self.robot_y / res + cy)
        self.ax.scatter(robot_gx, robot_gy, c='red', s=100, marker='*', zorder=6)
        self.ax.arrow(robot_gx, robot_gy,
                      10 * np.cos(self.robot_theta),
                      10 * np.sin(self.robot_theta),
                      head_width=4, head_length=4, fc='red', ec='red', zorder=6)

        if self.goal is not None:
            goal_gx = int(self.goal[0] / res + cx)
            goal_gy = int(self.goal[1] / res + cy)
            self.ax.scatter(goal_gx, goal_gy, c='blue', s=150, marker='x', zorder=6)

        if self.state == 'SWEEPING' and self.heading_idx < len(self.heading_queue):
            _, target_rad = self.heading_queue[self.heading_idx]
            self.ax.arrow(robot_gx, robot_gy,
                          14 * math.cos(target_rad),
                          14 * math.sin(target_rad),
                          head_width=3, head_length=3,
                          fc='orange', ec='orange', zorder=7, alpha=0.8)

        label = self.state
        if self.state == 'SWEEPING' and self.heading_queue:
            k, a = self.heading_queue[
                min(self.heading_idx, len(self.heading_queue) - 1)]
            label = (f'SWEEPING [{self.heading_idx + 1}/'
                     f'{len(self.heading_queue)}] {k} {math.degrees(a):.0f}°')
        elif self.state == 'SEARCHING' and self.search_mission:
            wp = self.search_waypoints[self.search_mission][self.search_wp_idx]
            label = f'SEARCHING → {wp["label"]}'
        elif self.state == 'AVOIDING':
            label = (f'AVOIDING [cyc {self.avoid_cycles}/{self.avoid_commit_cycles}] '
                     f'→ {self.resume_state}')

        self.ax.set_title(
            f'State: {label}'
            f'{"  (LOCALIZANDO)" if self.localizing else ""} | '
            f'σ_th={math.degrees(self.heading_std):.0f}°')
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