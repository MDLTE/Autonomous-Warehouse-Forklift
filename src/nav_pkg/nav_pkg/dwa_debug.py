import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_pkg.dwa import DWA
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
import numpy as np
import math
import threading


class DWADebugViz(Node):
    # diagnostico estacionario: el carrito NO se mueve. Lee el scan, corre el
    # DWA en modo avoiding asumiendo v actual = 0.15, y dibuja:
    #   - puntos de obstaculo que SI cuenta (negro)
    #   - puntos filtrados por cercania = auto-estructura (rojo)
    #   - todas las trayectorias candidatas (gris) y la elegida (verde)
    # todo en el marco del robot (robot en el origen, mirando +x).

    def __init__(self):
        super().__init__('dwa_debug_viz')

        self.dwa = DWA()

        # params para que coincida con el nav node real
        self.declare_parameter('max_v', 0.15)
        self.declare_parameter('max_w', 0.30)
        self.declare_parameter('assume_v', 0.15)            # v actual asumida
        self.declare_parameter('goal_dist', 2.0)            # meta recta enfrente
        self.declare_parameter('front_detect_arc_deg', 10.0)
        self.declare_parameter('front_detect_dist', 0.50)
        self.max_v     = float(self.get_parameter('max_v').value)
        self.max_w     = float(self.get_parameter('max_w').value)
        self.assume_v  = float(self.get_parameter('assume_v').value)
        self.goal_dist = float(self.get_parameter('goal_dist').value)
        self.front_arc = math.radians(
            float(self.get_parameter('front_detect_arc_deg').value))
        self.front_dist = float(self.get_parameter('front_detect_dist').value)

        self.dwa.max_v = self.max_v
        self.dwa.min_v = -self.max_v
        self.dwa.max_w = self.max_w
        self.dwa.min_w = -self.max_w
        self.dwa.avoid_max_w = self.max_w

        # meta recta enfrente (en el marco del robot)
        self.goal = (self.goal_dist, 0.0)

        self.scan_angles = None
        self.scan_ranges = None

        # resultados que calcula el timer y dibuja el main loop
        self.lock      = threading.Lock()
        self.fresh     = False
        self.kept_pts  = []
        self.near_pts  = []
        self.cands     = []      # (traj, feasible)
        self.best_traj = None
        self.best_v    = 0.0
        self.best_w    = 0.0
        self.front_hit = False
        self.front_dmin = float('inf')

        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_timer(0.1, self.compute)

        # figura
        self.fig, self.ax = plt.subplots(figsize=(8, 7))
        plt.ion()
        plt.show(block=False)
        plt.pause(0.1)

        self.get_logger().info(
            f'dwa_debug_viz listo — meta a {self.goal_dist} m, v asumida {self.assume_v}, '
            f'fork_sector {math.degrees(self.dwa.fork_sector_inner):.0f}-'
            f'{math.degrees(self.dwa.fork_sector_outer):.0f} deg @ {self.dwa.fork_filter_dist} m')

    def scan_cb(self, msg: LaserScan):
        # mismo shift que el nav node: frente = raw 180, corre +pi y envuelve
        n = len(msg.ranges)
        ang = msg.angle_min + math.pi + np.arange(n) * msg.angle_increment
        ang = np.arctan2(np.sin(ang), np.cos(ang))
        self.scan_angles = ang
        self.scan_ranges = np.array(msg.ranges, dtype=float)

    def _split_points(self):
        # separa puntos en kept (los que ve el DWA) vs filtrados por cercania
        # (auto-estructura), con el MISMO criterio que get_obstacle_points
        kept, near = [], []
        a = self.scan_angles
        r = self.scan_ranges
        for beam, d in zip(a, r):
            if not np.isfinite(d) or d <= 0 or d >= 3.5:
                continue
            in_fork = (self.dwa.fork_sector_inner <= abs(beam)
                       <= self.dwa.fork_sector_outer)
            thr = self.dwa.fork_filter_dist if in_fork else self.dwa.base_filter_dist
            pt = (d * math.cos(beam), d * math.sin(beam))
            if d < thr:
                near.append(pt)
            else:
                kept.append(pt)
        return kept, near

    def _front_info(self):
        # rayo mas cercano dentro del cono frontal (solo para el titulo)
        a = self.scan_angles
        r = self.scan_ranges
        m = (np.abs(a) <= self.front_arc) & np.isfinite(r) \
            & (r > 0.10) & (r < self.front_dist)
        if not np.any(m):
            return False, float('inf')
        return True, float(np.min(np.where(m, r, np.inf)))

    def compute(self):
        if self.scan_angles is None:
            return

        kept, near = self._split_points()
        obstacles  = self.dwa.get_obstacle_points(
            0.0, 0.0, 0.0, self.scan_angles, self.scan_ranges)
        front_hit, front_dmin = self._front_info()

        # mismo muestreo que el DWA en avoiding=True
        v_vals = np.arange(0.05, self.dwa.max_v, self.dwa.v_resolution)
        w_vals = np.arange(-self.dwa.avoid_max_w, self.dwa.avoid_max_w,
                           self.dwa.w_resolution)

        cands     = []
        best_cost = float('inf')
        best_traj = None
        best_v = best_w = 0.0

        for v in v_vals:
            for w in w_vals:
                t_pred = min(self.dwa.predict_time_max,
                             max(self.dwa.predict_time_min,
                                 self.dwa.lookahead_horizon / max(v, 1e-3)))
                traj = self.dwa.predict_trajectory(
                    0.0, 0.0, 0.0, self.assume_v, 0.0, v, w, predict_time=t_pred)
                cg = self.dwa.to_goal_cost_gain  * \
                    self.dwa.calc_to_goal_cost(traj, self.goal)
                cs = self.dwa.speed_cost_gain    * \
                    self.dwa.calc_speed_cost(traj)
                co = self.dwa.obstacle_cost_gain * \
                    self.dwa.calc_obstacle_cost(traj, obstacles)
                total    = cg + cs + co
                feasible = math.isfinite(total)
                cands.append((traj, feasible))
                if feasible and total < best_cost:
                    best_cost = total
                    best_traj = traj
                    best_v, best_w = v, w

        with self.lock:
            self.kept_pts   = kept
            self.near_pts   = near
            self.cands      = cands
            self.best_traj  = best_traj
            self.best_v     = best_v
            self.best_w     = best_w
            self.front_hit  = front_hit
            self.front_dmin = front_dmin
            self.fresh      = True

    def draw(self):
        with self.lock:
            if not self.fresh:
                return
            kept   = list(self.kept_pts)
            near   = list(self.near_pts)
            cands  = list(self.cands)
            best   = self.best_traj
            bv, bw = self.best_v, self.best_w
            fhit   = self.front_hit
            fdmin  = self.front_dmin
            self.fresh = False

        self.ax.cla()
        R = max(self.goal_dist + 0.3, 2.0)
        self.ax.set_xlim(-0.8, R)
        self.ax.set_ylim(-R / 1.6, R / 1.6)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('x adelante (m)')
        self.ax.set_ylabel('y izquierda (m)')

        # cono de deteccion frontal (lo que dispara AVOIDING)
        self.ax.add_patch(Wedge((0, 0), self.front_dist,
                                -math.degrees(self.front_arc),
                                math.degrees(self.front_arc),
                                color='gold', alpha=0.20))
        # zona de auto-filtro: solo los lobulos del montacargas se cortan a
        # fork_filter_dist; el resto solo se limpia de ruido (base_filter_dist)
        ei = math.degrees(self.dwa.fork_sector_inner)
        eo = math.degrees(self.dwa.fork_sector_outer)
        for t1, t2 in [(ei, eo), (-eo, -ei)]:
            self.ax.add_patch(Wedge((0, 0), self.dwa.fork_filter_dist, t1, t2,
                                    color='red', alpha=0.18, zorder=0))
        self.ax.add_patch(Circle((0, 0), self.dwa.base_filter_dist,
                                 color='red', alpha=0.08, zorder=0))

        # candidatas
        for traj, feasible in cands:
            if feasible:
                self.ax.plot(traj[:, 0], traj[:, 1],
                             color='0.6', lw=0.8, alpha=0.6, zorder=2)
            else:
                self.ax.plot(traj[:, 0], traj[:, 1],
                             color='red', lw=0.6, alpha=0.25,
                             ls=':', zorder=1)

        # elegida
        if best is not None:
            self.ax.plot(best[:, 0], best[:, 1],
                         color='limegreen', lw=3, zorder=4,
                         label=f'elegida v={bv:.2f} w={math.degrees(bw):+.0f}deg')

        # obstaculos
        if kept:
            k = np.array(kept)
            self.ax.scatter(k[:, 0], k[:, 1], c='black', s=10,
                            zorder=3, label='obst (DWA lo ve)')
        if near:
            e = np.array(near)
            self.ax.scatter(e[:, 0], e[:, 1], c='red', s=10, marker='x',
                            zorder=3, label='filtrado (montacargas/ruido)')

        # robot + radio de seguridad + meta
        self.ax.add_patch(Circle((0, 0), self.dwa.robot_radius,
                                 fill=False, ec='blue', ls='--', alpha=0.5))
        self.ax.arrow(0, 0, 0.25, 0, head_width=0.06, fc='blue', ec='blue', zorder=5)
        self.ax.scatter([self.goal[0]], [self.goal[1]],
                        c='blue', marker='x', s=120, zorder=5, label='meta')

        n_near  = len(near)
        blocked = best is None
        self.ax.set_title(
            f'front={"SI" if fhit else "no"}@{fdmin:.2f}m | '
            f'obst_vistos={len(kept)} filtrados={n_near} | '
            f'{"BLOQUEADO (todo choca)" if blocked else f"elige v={bv:.2f} w={math.degrees(bw):+.0f}deg"}')
        self.ax.legend(loc='upper left', fontsize=8)
        plt.pause(0.001)


def main(args=None):
    rclpy.init(args=args)
    node = DWADebugViz()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            node.draw()
            plt.pause(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()