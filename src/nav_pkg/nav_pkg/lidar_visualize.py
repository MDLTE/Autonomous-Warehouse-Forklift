import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import math
import threading
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Mide que sectores del LiDAR bloquea el montacargas y la ventana frontal.
# Pon el robot en espacio ABIERTO (nada a <1.5m). Los haces del montacargas
# regresan corto en TODOS los scans (es la estructura del robot); el resto
# regresa lejos.
#
# El frente del robot esta en raw 180 (LiDAR montado al reves, offset 180).
# Se acumula por BIN de 1 grado (0..359). bin index == grado, frente = bin 180.
#
# Zonas de interes visualizadas:
#   CYAN    ±20° alrededor del frente (160°-200°) — cono de deteccion frontal
#   NARANJA 20°-40° a cada lado del frente (140°-160° y 200°-220°) — zona de exclusion del montacargas

NBINS = 360

# ── zonas de interes (en grados, frame shifted — frente = 180°) ───
FRONT_ARC_DEG    = 20   # cono frontal: 180° ± 20°
PILLAR_INNER_DEG = 20   # inicio zona montacargas desde frente
PILLAR_OUTER_DEG = 40   # fin zona montacargas desde frente


class LidarScanCheck(Node):

    def __init__(self):
        super().__init__('lidar_scan_check')

        self.declare_parameter('blocked_thresh', 0.5)
        self.declare_parameter('warmup', 60)
        self.declare_parameter('show_view', True)
        self.blocked_thresh = float(self.get_parameter('blocked_thresh').value)
        self.warmup         = int(self.get_parameter('warmup').value)
        self.show_view      = bool(self.get_parameter('show_view').value)

        self.sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)

        self.n_short   = np.zeros(NBINS)
        self.n_invalid = np.zeros(NBINS)
        self.n_total   = np.zeros(NBINS)
        self.range_sum = np.zeros(NBINS)
        self.range_cnt = np.zeros(NBINS)
        self.n_scans   = 0
        self.blocked   = None
        self.last_th   = None
        self.last_r    = None
        self.info_printed = False
        self.reported     = False
        self.lock = threading.Lock()

        self.get_logger().info(
            f'lidar_scan_check — robot en espacio ABIERTO (nada a <1.5m). '
            f'umbral={self.blocked_thresh}m, warmup={self.warmup} scans. '
            f'frente del robot = bin 180 grados')

    def scan_cb(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=float)
        n      = len(ranges)
        angles = msg.angle_min + np.arange(n) * msg.angle_increment

        if not self.info_printed:
            self.info_printed = True
            self.get_logger().info(
                f'scan: N={n} pts | angle_min={math.degrees(msg.angle_min):.1f} '
                f'angle_max={math.degrees(msg.angle_max):.1f} '
                f'inc={math.degrees(msg.angle_increment):.2f} deg | '
                f'range=[{msg.range_min:.2f}, {msg.range_max:.2f}]')

        deg  = np.degrees(angles) % 360.0
        bins = np.clip(deg.astype(int), 0, NBINS - 1)

        valid = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
        short = valid & (ranges < self.blocked_thresh)

        np.add.at(self.n_total,   bins, 1.0)
        np.add.at(self.n_short,   bins, short.astype(float))
        np.add.at(self.n_invalid, bins, (~valid).astype(float))
        np.add.at(self.range_sum, bins[valid], ranges[valid])
        np.add.at(self.range_cnt, bins[valid], 1.0)
        self.n_scans += 1

        with self.lock:
            self.last_th = angles[valid]
            self.last_r  = ranges[valid]

        if self.n_scans >= self.warmup and not self.reported:
            self.reported = True
            self.report()

    def report(self):
        tot = np.maximum(self.n_total, 1.0)
        short_frac   = self.n_short / tot
        invalid_frac = self.n_invalid / tot
        mean_range   = self.range_sum / np.maximum(self.range_cnt, 1.0)

        seen         = self.n_total > 0
        self.blocked = (short_frac > 0.7) & seen
        no_return    = (invalid_frac > 0.7) & seen

        self.get_logger().info('===================== RESULTADO =====================')

        sectors = self._runs(self.blocked)
        if not sectors:
            self.get_logger().info('No hay sectores BLOQUEADOS (corto constante)')
        for a, b in sectors:
            seg = mean_range[a:b + 1]
            seg = seg[seg > 0]
            md  = float(np.mean(seg)) if len(seg) else 0.0
            self.get_logger().info(
                f'BLOQUEADO (montacargas): {a} a {b} grados '
                f'(~{b - a + 1} grados, dist media {md:.2f} m)')

        for a, b in self._runs(no_return):
            self.get_logger().info(
                f'SIN RETORNO: {a} a {b} grados (bloqueo muy pegado o sin pared)')

        fi = 180
        if self.blocked[fi]:
            self.get_logger().warn(
                'OJO: el frente (180) sale BLOQUEADO. '
                'despeja el frente o ajusta blocked_thresh.')
        lo, hi = fi, fi
        while lo - 1 >= 0 and not self.blocked[lo - 1]:
            lo -= 1
        while hi + 1 < NBINS and not self.blocked[hi + 1]:
            hi += 1
        self.get_logger().info(
            f'VENTANA FRONTAL ABIERTA: {lo} a {hi} grados | ancho ~{hi - lo} grados')
        self.get_logger().info('=====================================================')
        self.get_logger().info(
            f'Zona frontal configurada:   {180 - FRONT_ARC_DEG}° a '
            f'{180 + FRONT_ARC_DEG}° (±{FRONT_ARC_DEG}° — CYAN)')
        self.get_logger().info(
            f'Zona exclusion montacargas: {180 - PILLAR_OUTER_DEG}° a '
            f'{180 - PILLAR_INNER_DEG}° y '
            f'{180 + PILLAR_INNER_DEG}° a '
            f'{180 + PILLAR_OUTER_DEG}° (NARANJA)')
        self.get_logger().info('(la grafica resalta en rojo lo bloqueado)')

    def _runs(self, mask):
        runs = []
        i, n = 0, len(mask)
        while i < n:
            if mask[i]:
                j = i
                while j + 1 < n and mask[j + 1]:
                    j += 1
                runs.append((i, j))
                i = j + 1
            else:
                i += 1
        return runs

    def draw(self):
        with self.lock:
            if self.last_th is None:
                return
            th = self.last_th.copy()
            r  = self.last_r.copy()

        self.ax.cla()
        self.ax.set_theta_zero_location('E')
        self.ax.set_theta_direction(1)

        rmax = float(np.nanmax(r)) if len(r) else 1.5

        # ── zones of interest ──────────────────────────────────────
        front_center = math.radians(180)
        front_hw     = math.radians(FRONT_ARC_DEG)
        pillar_inner = math.radians(PILLAR_INNER_DEG)
        pillar_outer = math.radians(PILLAR_OUTER_DEG)

        # front detection cone — cyan ±20° around 180°
        theta_front = np.linspace(front_center - front_hw,
                                  front_center + front_hw, 60)
        self.ax.fill_between(theta_front, 0, rmax,
                             alpha=0.18, color='cyan', zorder=1)

        # left pillar exclusion: 140°-160° (= 180° - 40° to 180° - 20°)
        theta_lp = np.linspace(front_center - pillar_outer,
                               front_center - pillar_inner, 40)
        self.ax.fill_between(theta_lp, 0, rmax,
                             alpha=0.25, color='orange', zorder=1)

        # right pillar exclusion: 200°-220° (= 180° + 20° to 180° + 40°)
        theta_rp = np.linspace(front_center + pillar_inner,
                               front_center + pillar_outer, 40)
        self.ax.fill_between(theta_rp, 0, rmax,
                             alpha=0.25, color='orange', zorder=1)

        # boundary lines
        for deg, color, ls in [
            (180 - FRONT_ARC_DEG,    'cyan',   '--'),
            (180 + FRONT_ARC_DEG,    'cyan',   '--'),
            (180 - PILLAR_OUTER_DEG, 'orange', ':'),
            (180 - PILLAR_INNER_DEG, 'orange', ':'),
            (180 + PILLAR_INNER_DEG, 'orange', ':'),
            (180 + PILLAR_OUTER_DEG, 'orange', ':'),
        ]:
            self.ax.plot([math.radians(deg), math.radians(deg)],
                         [0, rmax], color=color, lw=1.2, linestyle=ls, zorder=2)

        # ── scan data ─────────────────────────────────────────────
        self.ax.scatter(th, r, s=4, c='black', alpha=0.5, zorder=3)

        if self.blocked is not None:
            bins  = np.clip((np.degrees(th) % 360).astype(int), 0, NBINS - 1)
            bmask = self.blocked[bins]
            self.ax.scatter(th[bmask], r[bmask], s=10, c='red', zorder=4)

        # robot front line
        self.ax.plot([front_center, front_center], [0, rmax],
                     c='lime', lw=2, zorder=5, label='Frente (180°)')

        # legend
        legend_patches = [
            mpatches.Patch(color='cyan',   alpha=0.5,
                           label=f'Cono frontal ±{FRONT_ARC_DEG}°'),
            mpatches.Patch(color='orange', alpha=0.5,
                           label=f'Exclusion montacargas {PILLAR_INNER_DEG}°-{PILLAR_OUTER_DEG}°'),
            mpatches.Patch(color='red',    alpha=0.8,
                           label='Bloqueado (montacargas)'),
        ]
        self.ax.legend(handles=legend_patches, loc='upper right',
                       fontsize=7, bbox_to_anchor=(1.3, 1.1))

        self.ax.set_title(
            f'LiDAR — scans: {self.n_scans} | frente = linea verde (180°)\n'
            f'CYAN: cono ±{FRONT_ARC_DEG}°   '
            f'NARANJA: exclusion {PILLAR_INNER_DEG}°-{PILLAR_OUTER_DEG}°',
            fontsize=9)
        plt.pause(0.001)


def main(args=None):
    rclpy.init(args=args)
    node = LidarScanCheck()

    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    if node.show_view:
        node.fig = plt.figure(figsize=(8, 8))
        node.ax  = node.fig.add_subplot(111, projection='polar')
        plt.ion()
        plt.show(block=False)

    try:
        while rclpy.ok():
            if node.show_view:
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