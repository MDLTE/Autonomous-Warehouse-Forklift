#!/usr/bin/env python3
"""
poseKalmanSim.py — EKF de localizacion del PuzzleBot (simulacion)

Cambio grande respecto a la version anterior:
  - Modelo de observacion RANGE-BEARING (el estandar del curso). Cada marcador
    es un landmark de posicion conocida; medimos rango y bearing y corregimos.
    YA NO se usa gamma ni se reconstruye la pose desde un solo marcador, que
    era lo que hacia que al girar en el eje el robot "orbitara" el marcador.
  - La R de cada correccion usa el range_sigma_m que publica el detector
    (crece con la distancia) -> los marcadores lejanos pesan menos solos.
  - Sin corte duro de distancia: lejos = R grande, no descarte.
  - Dibuja la elipse de covarianza (2-sigma) del EKF.

Sub: /aruco/detections (String/JSON), /odom (Odometry, world-frame en sim)
Pub: /odom_ekf (Odometry)
"""

import json
import math

import cv2
import numpy as np
import rclpy
from rclpy import qos
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String


# posiciones (x, y) de los marcadores del mundo e80. el bearing-only no
# necesita la orientacion del marcador, solo donde esta.
ARUCO_MAP = {
    4:  (0.000,    1.050),
    0:  (0.000,    3.786),
    1:  (1.610,    4.847741),
    2:  (3.655820, 3.757),
    3:  (3.655820, 1.090),
    6:  (1.217200, 2.499262),
    5:  (2.401577, 2.497429),
    7:  (1.214891, 3.553349),
    8:  (2.397084, 3.551018),
    9:  (1.344316, 1.508095),
    10: (2.401642, 1.508212),
}

# ruido de proceso (prediccion). tunables.
Q_TRANS_K = 0.10    # m de sigma por m avanzado
Q_TRANS_B = 0.005
Q_ROT_K   = 0.15    # rad de sigma por rad girado
Q_ROT_B   = 0.01

# ruido de medicion
SIGMA_R_FLOOR = 0.03            # piso del sigma de rango (m)
SIGMA_BEARING = math.radians(2.0)  # sigma del bearing
BEARING_SIGN  = -1.0            # yaw_deg(+)=marcador a la derecha -> bearing(-)
GATE_CHI2_2D  = 9.21           # chi2 2 GL 99%
MAX_USE_RANGE = 4.0            # arriba de esto ni lo usamos (puro ruido)

# visualizacion
_WX_MIN, _WX_MAX = -0.3, 4.1
_WY_MIN, _WY_MAX = -0.3, 5.2
MAP_SIZE, MAP_PADDING = 700, 50
_WX_RANGE = max(_WX_MAX - _WX_MIN, 1.0)
_WY_RANGE = max(_WY_MAX - _WY_MIN, 1.0)
COL_BG, COL_GRID = (30, 30, 30), (55, 55, 55)
COL_ARUCO_UNK, COL_ARUCO_VIS = (100, 100, 100), (50, 220, 50)
COL_ODO, COL_KF, COL_LINE, COL_ELLIPSE = (255, 180, 50), (50, 180, 255), (100, 180, 100), (180, 140, 255)


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_from_quat(q):
    return math.atan2(2.0 * q.w * q.z, 1.0 - 2.0 * q.z * q.z)


# ══════════════════════════════════════════════════════════════════
#  EKF range-bearing
# ══════════════════════════════════════════════════════════════════

class EKF:
    def __init__(self):
        self.x = np.zeros(3)
        self.P = np.diag([1.0, 1.0, 1.0])
        self.initialized = False

    def init_pose(self, x, y, th):
        self.x = np.array([x, y, th], dtype=float)
        self.P = np.diag([0.1, 0.1, 0.2])
        self.initialized = True

    def predict(self, v, w, dt):
        if not self.initialized or dt <= 0:
            return
        th = self.x[2]
        if abs(w) < 1e-6:
            dx = v * dt * math.cos(th)
            dy = v * dt * math.sin(th)
            dth = 0.0
            Fx = np.array([[1, 0, -v * dt * math.sin(th)],
                           [0, 1,  v * dt * math.cos(th)],
                           [0, 0, 1]])
        else:
            r = v / w
            dth = w * dt
            th2 = th + dth
            dx = r * (math.sin(th2) - math.sin(th))
            dy = r * (math.cos(th) - math.cos(th2))
            Fx = np.array([[1, 0, r * (math.cos(th2) - math.cos(th))],
                           [0, 1, r * (math.sin(th2) - math.sin(th))],
                           [0, 0, 1]])

        self.x[0] += dx
        self.x[1] += dy
        self.x[2] = wrap_angle(th + dth)

        trans = abs(v) * dt
        rot = abs(w) * dt
        Qd = np.diag([(Q_TRANS_K * trans + Q_TRANS_B) ** 2,
                      (Q_TRANS_K * trans + Q_TRANS_B) ** 2,
                      (Q_ROT_K * rot + Q_ROT_B) ** 2])
        self.P = Fx @ self.P @ Fx.T + Qd

    def update_landmark(self, mx, my, z_r, z_b, sigma_r):
        # un marcador: medicion z = [rango, bearing]
        if not self.initialized:
            return None
        x, y, th = self.x
        dx = mx - x
        dy = my - y
        q = dx * dx + dy * dy
        r = math.sqrt(q)
        if r < 1e-3:
            return None

        z_hat = np.array([r, wrap_angle(math.atan2(dy, dx) - th)])
        H = np.array([[-dx / r, -dy / r,  0.0],
                      [ dy / q, -dx / q, -1.0]])
        R = np.diag([sigma_r ** 2, SIGMA_BEARING ** 2])

        innov = np.array([z_r - z_hat[0], wrap_angle(z_b - z_hat[1])])
        S = H @ self.P @ H.T + R
        Sinv = np.linalg.inv(S)

        d2 = float(innov @ Sinv @ innov)
        if d2 > GATE_CHI2_2D:
            return False

        K = self.P @ H.T @ Sinv
        self.x = self.x + K @ innov
        self.x[2] = wrap_angle(self.x[2])
        self.P = (np.eye(3) - K @ H) @ self.P
        return True

    @property
    def state(self):
        return float(self.x[0]), float(self.x[1]), float(self.x[2])


# ══════════════════════════════════════════════════════════════════
#  Visualizacion
# ══════════════════════════════════════════════════════════════════

def world_to_map(wx, wy):
    scale = min((MAP_SIZE - 2 * MAP_PADDING) / _WX_RANGE,
                (MAP_SIZE - 2 * MAP_PADDING) / _WY_RANGE)
    px = int(MAP_PADDING + (wx - _WX_MIN) * scale)
    py = int(MAP_SIZE - MAP_PADDING - (wy - _WY_MIN) * scale)
    return px, py


def _scale():
    return min((MAP_SIZE - 2 * MAP_PADDING) / _WX_RANGE,
               (MAP_SIZE - 2 * MAP_PADDING) / _WY_RANGE)


def draw_cov_ellipse(canvas, x, y, P2, color):
    # elipse 2-sigma de la covarianza de posicion
    try:
        vals, vecs = np.linalg.eigh(P2)
    except np.linalg.LinAlgError:
        return
    vals = np.clip(vals, 1e-6, None)
    ang = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    axes = (int(2 * math.sqrt(vals[0]) * _scale()),
            int(2 * math.sqrt(vals[1]) * _scale()))
    cx, cy = world_to_map(x, y)
    # el eje y de la imagen va al reves, por eso -ang
    cv2.ellipse(canvas, (cx, cy), axes, -ang, 0, 360, color, 1, cv2.LINE_AA)


def draw_map(kf, odo, trail_kf, trail_odo, visible, meas_r):
    canvas = np.full((MAP_SIZE, MAP_SIZE, 3), COL_BG, dtype=np.uint8)

    x = 0.0
    while x <= _WX_MAX:
        gx, _ = world_to_map(x, _WY_MIN)
        cv2.line(canvas, (gx, MAP_PADDING), (gx, MAP_SIZE - MAP_PADDING), COL_GRID, 1)
        x += 0.5
    y = 0.0
    while y <= _WY_MAX:
        _, gy = world_to_map(_WX_MIN, y)
        cv2.line(canvas, (MAP_PADDING, gy), (MAP_SIZE - MAP_PADDING, gy), COL_GRID, 1)
        y += 0.5

    for pt in trail_odo:
        cv2.circle(canvas, world_to_map(pt[0], pt[1]), 2, COL_ODO, -1)
    for pt in trail_kf:
        cv2.circle(canvas, world_to_map(pt[0], pt[1]), 2, COL_KF, -1)

    for mid, (mx, my) in ARUCO_MAP.items():
        px, py = world_to_map(mx, my)
        color = COL_ARUCO_VIS if mid in visible else COL_ARUCO_UNK
        cv2.rectangle(canvas, (px - 8, py - 8), (px + 8, py + 8), color, -1)
        cv2.putText(canvas, str(mid), (px + 10, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    kx, ky, kth = kf.state
    rxk, ryk = world_to_map(kx, ky)
    for mid in visible:
        if mid in ARUCO_MAP:
            mpx, mpy = world_to_map(*ARUCO_MAP[mid])
            cv2.line(canvas, (rxk, ryk), (mpx, mpy), COL_LINE, 1, cv2.LINE_AA)
            if mid in meas_r:
                lx, ly = (rxk + mpx) // 2, (ryk + mpy) // 2
                cv2.putText(canvas, f"{meas_r[mid]:.2f}m", (lx + 3, ly - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 255, 180), 1)

    draw_cov_ellipse(canvas, kx, ky, kf.P[:2, :2], COL_ELLIPSE)

    ox, oy, oth = odo
    rxo, ryo = world_to_map(ox, oy)
    cv2.circle(canvas, (rxo, ryo), 8, COL_ODO, -1)
    cv2.arrowedLine(canvas, (rxo, ryo),
                    (int(rxo + 16 * math.cos(oth)), int(ryo - 16 * math.sin(oth))),
                    (255, 220, 80), 1, cv2.LINE_AA, tipLength=0.4)

    cv2.circle(canvas, (rxk, ryk), 11, COL_KF, -1)
    cv2.arrowedLine(canvas, (rxk, ryk),
                    (int(rxk + 22 * math.cos(kth)), int(ryk - 22 * math.sin(kth))),
                    (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.35)

    cv2.putText(canvas, "EKF (azul)   Odometria (naranja)", (MAP_PADDING, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    cv2.putText(canvas, f"KF:  x={kx:.2f} y={ky:.2f} th={math.degrees(kth):.1f}",
                (10, MAP_SIZE - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_KF, 1)
    cv2.putText(canvas, f"Odo: x={ox:.2f} y={oy:.2f} th={math.degrees(oth):.1f}",
                (10, MAP_SIZE - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_ODO, 1)
    return canvas


# ══════════════════════════════════════════════════════════════════
#  Nodo
# ══════════════════════════════════════════════════════════════════

class PoseKalmanSimNode(Node):
    def __init__(self):
        super().__init__('pose_kalman_sim')

        self.pose_pub = self.create_publisher(Odometry, '/odom_ekf', 10)
        self.create_subscription(String, '/aruco/detections', self.aruco_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, qos.qos_profile_sensor_data)

        self.latest_v = 0.0
        self.latest_w = 0.0
        self.odom_pose = None       # pose cruda de /odom (world-frame en sim)
        self.last_dets = []

        self.ekf = EKF()
        # odometria pura (dead-reckoning) para comparar contra el EKF
        self.odo = np.zeros(3)
        self.odo_ready = False

        self.trail_kf, self.trail_odo = [], []
        self.MAX_TRAIL = 500
        self.visible, self.meas_r = [], {}

        self.last_time = self.get_clock().now()
        self.create_timer(0.05, self.loop)
        self.get_logger().info(f'PoseKalman SIM (range-bearing) | IDs={list(ARUCO_MAP)}')

    def odom_cb(self, msg: Odometry):
        self.latest_v = msg.twist.twist.linear.x
        self.latest_w = msg.twist.twist.angular.z
        p = msg.pose.pose
        self.odom_pose = (p.position.x, p.position.y, yaw_from_quat(p.orientation))

    def aruco_cb(self, msg: String):
        try:
            self.last_dets = json.loads(msg.data)
        except json.JSONDecodeError:
            self.last_dets = []

    def init_from_odom(self):
        # /odom en este sim arranca en la pose real del mundo -> init valido
        if self.odom_pose is None:
            return
        self.ekf.init_pose(*self.odom_pose)
        self.odo = np.array(self.odom_pose, dtype=float)
        self.odo_ready = True
        self.get_logger().info(
            f'init desde /odom: x={self.odom_pose[0]:.2f} y={self.odom_pose[1]:.2f} '
            f'th={math.degrees(self.odom_pose[2]):.1f}')

    def predict_step(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now
        if dt <= 0 or dt > 0.5:
            return
        v, w = self.latest_v, self.latest_w
        self.ekf.predict(v, w, dt)
        # dead-reckoning para la traza naranja
        th = self.odo[2]
        if abs(w) < 1e-6:
            self.odo[0] += v * dt * math.cos(th)
            self.odo[1] += v * dt * math.sin(th)
        else:
            r = v / w
            th2 = th + w * dt
            self.odo[0] += r * (math.sin(th2) - math.sin(th))
            self.odo[1] += r * (math.cos(th) - math.cos(th2))
            self.odo[2] = wrap_angle(th2)

    def correct_step(self):
        self.visible, self.meas_r = [], {}
        for d in self.last_dets:
            mid = d.get('id')
            rng = d.get('distance_m')
            yaw = d.get('yaw_deg')
            if mid not in ARUCO_MAP or rng is None or yaw is None:
                continue
            if rng > MAX_USE_RANGE:
                continue
            self.visible.append(mid)
            self.meas_r[mid] = rng

            z_r = rng
            z_b = BEARING_SIGN * math.radians(yaw)
            sigma_r = max(d.get('range_sigma_m', SIGMA_R_FLOOR), SIGMA_R_FLOOR)
            mx, my = ARUCO_MAP[mid]
            self.ekf.update_landmark(mx, my, z_r, z_b, sigma_r)

    def publish_pose(self):
        if not self.ekf.initialized:
            return
        kx, ky, kth = self.ekf.state
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.pose.pose.position.x = kx
        msg.pose.pose.position.y = ky
        msg.pose.pose.orientation.z = math.sin(kth / 2)
        msg.pose.pose.orientation.w = math.cos(kth / 2)
        # covarianza (x, y, yaw) en el bloque 6x6
        P = self.ekf.P
        cov = [0.0] * 36
        cov[0], cov[1], cov[5] = P[0, 0], P[0, 1], P[0, 2]
        cov[6], cov[7], cov[11] = P[1, 0], P[1, 1], P[1, 2]
        cov[30], cov[31], cov[35] = P[2, 0], P[2, 1], P[2, 2]
        msg.pose.covariance = cov
        self.pose_pub.publish(msg)

    def loop(self):
        if not self.ekf.initialized:
            self.init_from_odom()
            return

        self.predict_step()
        self.correct_step()
        self.publish_pose()

        kx, ky, _ = self.ekf.state
        self.trail_kf.append((kx, ky))
        self.trail_odo.append((self.odo[0], self.odo[1]))
        self.trail_kf = self.trail_kf[-self.MAX_TRAIL:]
        self.trail_odo = self.trail_odo[-self.MAX_TRAIL:]

        img = draw_map(self.ekf, tuple(self.odo),
                       self.trail_kf, self.trail_odo, self.visible, self.meas_r)
        cv2.imshow('PoseKalman SIM', img)
        if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
            raise KeyboardInterrupt


def main(args=None):
    rclpy.init(args=args)
    node = PoseKalmanSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()