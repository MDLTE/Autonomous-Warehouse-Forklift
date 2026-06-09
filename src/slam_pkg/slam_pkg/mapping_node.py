import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
import math
from slam_pkg.occupancy_grid import OccupancyGrid
from slam_pkg.MCL import MCL
import matplotlib.pyplot as plt
import os
import threading

user = os.environ.get('USER')


# ── Loop closure params ───────────────────────────────────────────
LC_KEYFRAME_DIST    = 0.20   # dense coverage in small space
LC_SEARCH_RADIUS    = 0.35   # 1.75 × keyframe dist, covers drift
LC_MIN_KEYFRAME_AGE = 10     # 10 × 0.20 = 2m minimum loop — avoids self-match on straights
LC_CORR_THRESHOLD   = 0.78   # slightly relaxed since scan geometry varies in corridors
LC_INJECT_FRACTION  = 0.15   # keep conservative
LC_INJECT_SIGMA_XY  = 0.06   # tighter — small world, keyframe poses are more reliable
LC_INJECT_SIGMA_TH  = 0.08   # tighter for same reason


def scan_correlation(ranges_a, ranges_b, range_max):
    """
    Normalised cross-correlation between two range arrays.
    Clamps inf/nan to range_max, normalises, returns score in [0, 1].
    """
    def clean(r):
        a = np.array(r, dtype=np.float32)
        a = np.where(np.isfinite(a), a, range_max)
        a = np.clip(a, 0.0, range_max)
        return a / range_max

    a = clean(ranges_a)
    b = clean(ranges_b)

    if len(a) != len(b):
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-6:
        return 0.0
    return float(np.dot(a, b) / denom)


class MappingNode(Node):

    def __init__(self):
        super().__init__('mapping_node')

        self.declare_parameter('mode', 'mapping')
        mode = self.get_parameter('mode').value
        self.get_logger().info(f'Starting in {mode} mode')

        self.robot_x     = 0.0
        self.robot_y     = 0.0
        self.robot_theta = 0.0

        self.prev_x     = 0.0
        self.prev_y     = 0.0
        self.prev_theta = 0.0

        self.prev_map_x     = 0.0
        self.prev_map_y     = 0.0
        self.prev_map_theta = 0.0

        self.v      = 0.0
        self.w      = 0.0
        self.odom_dt = 0.05

        self.odomReceived = False

        if mode == 'localization':
            from slam_pkg.map_io import load_map
            map_path = f'/home/{user}/ros2_ws/src/slam_pkg/maps/map'
            self.grid = load_map(map_path)
            self.get_logger().info('Map loaded')
            self.mcl = MCL(
                n_particles=1000,
                initial_x=None, initial_y=None, initial_theta=None,
                grid=self.grid, global_init=True)
        else:
            self.grid = OccupancyGrid(
                rows=590, cols=409, resolution=0.01,
                origin_x=14, origin_y=79)
            self.mcl = MCL(
                n_particles=1000,
                initial_x=0.35, initial_y=0.35, initial_theta=0.0,
                grid=self.grid, global_init=False)

        # ── loop closure keyframe store ───────────────────────────
        # Each entry: {'x': float, 'y': float, 'theta': float, 'ranges': list}
        self.keyframes       = []
        self.lc_dist_accum   = 0.0     # accumulated travel since last keyframe
        self.lc_prev_x       = 0.35
        self.lc_prev_y       = 0.35
        self.lc_events       = 0       # total loop closures detected

        # ── visualization ─────────────────────────────────────────
        self.viz_lock   = threading.Lock()
        self.viz_needed = False
        self.viz_data   = None          # snapshot passed to viz thread

        # ── autosave timer (every 30 s, mapping mode only) ────────
        if mode == 'mapping':
            self.create_timer(30.0, self.autosave)

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_cb, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_cb, 10)

    # ═══════════════════════════════════════════════════════════════
    #  CALLBACKS
    # ═══════════════════════════════════════════════════════════════

    def scan_cb(self, msg: LaserScan):
        if not self.odomReceived:
            return

        mode = self.get_parameter('mode').value

        scan_angles = np.arange(
            msg.angle_min, msg.angle_max, msg.angle_increment)
        # normalize to [-π, π] — works for both 0→2π and -π→π scan formats
        scan_angles = (scan_angles + math.pi) % (2 * math.pi) - math.pi
        scan_ranges = msg.ranges

        # MCL update
        self.mcl.update_weights(scan_angles, scan_ranges, msg.range_max)
        self.mcl.resample()
        mcl_x, mcl_y, mcl_theta = self.mcl.estimate()

        if mode == 'mapping':
            angle_moved = abs(np.arctan2(
                np.sin(mcl_theta - self.prev_map_theta),
                np.cos(mcl_theta - self.prev_map_theta)))
            self.prev_map_theta = mcl_theta

            if angle_moved > 0.15:
                self.prev_map_x = mcl_x
                self.prev_map_y = mcl_y
                return

            robot_still    = abs(self.v) < 0.02 and abs(self.w) < 0.05
            moving_straight = abs(self.v) >= 0.02 and abs(self.w) < 0.05
            dist_moved = math.hypot(mcl_x - self.prev_map_x,
                                    mcl_y - self.prev_map_y)

            if robot_still or (moving_straight and dist_moved > 0.02):
                self.prev_map_x = mcl_x
                self.prev_map_y = mcl_y
                self.grid.update_map_with_scan(
                    mcl_x, mcl_y, mcl_theta,
                    scan_angles, scan_ranges, range_max=msg.range_max)

            # loop closure check
            self._update_loop_closure(
                mcl_x, mcl_y, mcl_theta,
                scan_ranges, msg.range_max)

        # hand snapshot to viz thread (non-blocking)
        self.viz_data = {
            'grid':      self.grid.grid.copy(),
            'particles': [(p.x, p.y) for p in self.mcl.particles],
            'mcl_x':     mcl_x,
            'mcl_y':     mcl_y,
            'mcl_theta': mcl_theta,
            'lc_events': self.lc_events,
            'keyframes': [(kf['x'], kf['y']) for kf in self.keyframes],
        }
        self.viz_needed = True

    def odom_cb(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

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
            np.cos(self.robot_theta - self.prev_theta))

        self.v = np.sqrt(dx**2 + dy**2) / self.odom_dt
        self.w = dtheta / self.odom_dt

        self.mcl.predict(dx, dy, dtheta)

        self.prev_x     = self.robot_x
        self.prev_y     = self.robot_y
        self.prev_theta = self.robot_theta

    # ═══════════════════════════════════════════════════════════════
    #  LOOP CLOSURE
    # ═══════════════════════════════════════════════════════════════

    def _update_loop_closure(self, x, y, theta, ranges, range_max):
        """
        Two-phase loop closure for MCL-based SLAM:

        Phase 1 — Keyframe management:
          Store a keyframe (pose + scan snapshot) every LC_KEYFRAME_DIST metres
          of travel. Keyframes build a sparse history of visited poses.

        Phase 2 — Closure detection:
          When the robot's current estimated position is within LC_SEARCH_RADIUS
          of an old keyframe (skipping the LC_MIN_KEYFRAME_AGE most recent to
          avoid self-matching), compute the scan correlation between the current
          scan and the stored keyframe scan. If the correlation exceeds
          LC_CORR_THRESHOLD, inject LC_INJECT_FRACTION of particles near the
          keyframe pose with Gaussian noise. This biases the particle cloud
          toward the historically consistent pose without discarding the current
          estimate entirely.
        """
        # ── Phase 1: keyframe storage ─────────────────────────────
        travel = math.hypot(x - self.lc_prev_x, y - self.lc_prev_y)
        self.lc_dist_accum += travel
        self.lc_prev_x, self.lc_prev_y = x, y

        if self.lc_dist_accum >= LC_KEYFRAME_DIST:
            self.keyframes.append({
                'x': x, 'y': y, 'theta': theta,
                'ranges': list(ranges)
            })
            self.lc_dist_accum = 0.0

        # ── Phase 2: closure detection ────────────────────────────
        if len(self.keyframes) <= LC_MIN_KEYFRAME_AGE:
            return

        candidates = self.keyframes[:-LC_MIN_KEYFRAME_AGE]

        for kf in candidates:
            dist = math.hypot(x - kf['x'], y - kf['y'])
            if dist > LC_SEARCH_RADIUS:
                continue

            score = scan_correlation(ranges, kf['ranges'], range_max)
            if score < LC_CORR_THRESHOLD:
                continue

            # good match — inject particles near the keyframe pose
            n_inject = int(len(self.mcl.particles) * LC_INJECT_FRACTION)
            inject_indices = np.random.choice(
                len(self.mcl.particles), n_inject, replace=False)

            for idx in inject_indices:
                p = self.mcl.particles[idx]
                p.x     = kf['x'] + np.random.normal(0, LC_INJECT_SIGMA_XY)
                p.y     = kf['y'] + np.random.normal(0, LC_INJECT_SIGMA_XY)
                p.theta = kf['theta'] + np.random.normal(0, LC_INJECT_SIGMA_TH)
                p.weight = 1.0 / len(self.mcl.particles)

            self.lc_events += 1
            self.get_logger().info(
                f'Loop closure #{self.lc_events} | '
                f'score={score:.3f} dist={dist:.2f}m | '
                f'kf pose=({kf["x"]:.2f},{kf["y"]:.2f}) | '
                f'injected {n_inject} particles')
            break   # one closure per scan cycle is enough

    # ═══════════════════════════════════════════════════════════════
    #  VISUALIZATION — runs in main thread, never in callbacks
    # ═══════════════════════════════════════════════════════════════

    def safe_visualize(self):
        if not self.viz_needed:
            return
        with self.viz_lock:
            data = self.viz_data
            self.viz_needed = False

        if data is None:
            return

        prob_map = 1 - (1 / (1 + np.exp(data['grid'])))
        plt.cla()
        plt.imshow(prob_map, cmap='gray_r', origin='lower')

        # particles
        res = self.grid.resolution
        cx  = self.grid.center_x
        cy  = self.grid.center_y

        if data['particles']:
            gxs = [int(px / res + cx) for px, _ in data['particles']]
            gys = [int(py / res + cy) for _, py in data['particles']]
            plt.scatter(gxs, gys, c='blue', s=2, alpha=0.4, zorder=2)

        # keyframes (small orange dots)
        for kx, ky in data['keyframes']:
            plt.scatter(
                int(kx / res + cx), int(ky / res + cy),
                c='orange', s=8, alpha=0.6, zorder=3)

        # estimated robot pose
        mgx = int(data['mcl_x'] / res + cx)
        mgy = int(data['mcl_y'] / res + cy)
        plt.scatter(mgx, mgy, c='red', s=50, marker='*', zorder=5)
        plt.arrow(
            mgx, mgy,
            8 * np.cos(data['mcl_theta']),
            8 * np.sin(data['mcl_theta']),
            head_width=3, head_length=3, fc='red', ec='red', zorder=5)

        plt.title(
            f'MCL SLAM | keyframes: {len(data["keyframes"])} | '
            f'loop closures: {data["lc_events"]}')
        plt.pause(0.001)

    # ═══════════════════════════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════════════════════════

    def autosave(self):
        mode = self.get_parameter('mode').value
        if mode != 'mapping':
            return
        map_path = f'/home/{user}/ros2_ws/src/slam_pkg/maps/map_new'
        self.grid.save_map(map_path)
        self.get_logger().info('Map autosaved')

    def quaternion_to_yaw(self, qx, qy, qz, qw):
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None):
    plt.ion()
    rclpy.init(args=args)
    node = MappingNode()

    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            node.safe_visualize()
            plt.pause(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        mode = node.get_parameter('mode').value
        if mode == 'mapping':
            map_path = f'/home/{user}/ros2_ws/src/slam_pkg/maps/map_new'
            node.grid.save_map(map_path)
            print(f'Map saved to {map_path}')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()