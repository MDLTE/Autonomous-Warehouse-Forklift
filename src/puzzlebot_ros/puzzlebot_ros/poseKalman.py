#!/usr/bin/env python3
"""
poseKalman.py — EKF de localizacion Puzzlebot (robot REAL)

Portado de lo que sirvio en sim: modelo de observacion RANGE-BEARING (sin
gamma en las correcciones). Cada marcador es un landmark de posicion conocida;
medimos rango y bearing y corregimos. Eso arregla el bug del "orbitar".

Adaptado al robot real:
  - Odometria desde encoders (/VelocityEncR, /VelocityEncL, rad/s).
  - Publica en /odom (el EKF ES la odometria del robot real) + twist + cov.
  - Init sin suponer heading: del primer marcador saca pose completa
    (heading via gamma de solvePnP, posicion via rango+bearing). Gamma se usa
    SOLO en el instante del init, no en las correcciones continuas.
  - Offset de camara cam_forward=0.066: la medicion sale de la camara, que va
    adelante del eje de giro, asi que el modelo de observacion predice desde
    la posicion de la camara.
  - Piso al sigma de rango (el detector sub-estima de cerca).
  - Gate chi-cuadrado para tirar lecturas basura.

Mapeo extra (NO corrigen el EKF, se ubican con la pose ya estimada):
  - Logos: posicion DESCONOCIDA. Se proyectan a mundo con la pose del EKF y, si
    aterrizan sobre un espacio de puerta FIJO, suman un voto a esa puerta.
  - QR: posicion DESCONOCIDA. Se proyectan a mundo y se promedian (EMA), se
    dibujan como estrellas y se publican en /qr/map.

Sub: /aruco/detections, /logo_detections, /qr/detections (String/JSON),
     /VelocityEncR, /VelocityEncL (Float32)
Pub: /odom (Odometry), /logo/door_map (String), /qr/map (String)
"""

import json
import math
from collections import Counter

import cv2
import numpy as np
import rclpy
from rclpy import qos
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, String
from sensor_msgs.msg import CompressedImage


# mapa real medido (esquina inferior izquierda = origen). (x, y, yaw_normal)
# arena 3.6645 x 4.863 m
ARUCO_MAP = {
    4:  (0.0000, 1.046,  0.0),
    0:  (0.0000, 3.835,  0.0),
    1:  (2.0300, 4.863, -math.pi/2),
    2:  (3.6645, 3.835,  math.pi),
    3:  (3.6645, 1.046,  math.pi),
    6:  (1.2430, 2.505, -math.pi/2),
    5:  (2.3795, 2.505, -math.pi/2),
    7:  (1.2430, 3.565,  math.pi/2),
    8:  (2.3795, 3.565,  math.pi/2),
    9:  (0.3650, 0.000,  math.pi/2),
    10: (2.8755, 0.000,  math.pi/2),
}

# robot real — encoders CALIBRADOS (radio y base medidos, mismos que el nodo de alineacion).
# el radio real (~0.0391) es ~18% menor que el nominal 0.0475: antes el modelo
# sobre-estimaba la distancia ~22% y por eso el EKF desconfiaba del dead-reckoning.
WHEEL_RADIUS = 0.0391
WHEEL_BASE   = 0.180
CAM_FORWARD  = 0.066          # lente adelante del eje de giro (m)

# ruido de proceso — con los encoders ya calibrados el dead-reckoning es mucho
# mas fiel, asi que bajamos Q: la P crece mas lento sin ArUco y el filtro confia
# mas en los encoders durante dropouts largos. (Si al volver a ver un ArUco el
# filtro empieza a rechazar correcciones buenas, sube estos K de nuevo.)
Q_TRANS_K, Q_TRANS_B = 0.02, 0.005   # antes 0.03 (y 0.10 con encoders sin calibrar)
Q_ROT_K,   Q_ROT_B   = 0.035, 0.01   # antes 0.05 (y 0.15 con encoders sin calibrar)

# ruido de medicion
SIGMA_R_FLOOR = 0.025         # piso del sigma de rango (m) ~ error real de cerca
SIGMA_BEARING = math.radians(2.5)
SIGMA_HEADING_BASE  = math.radians(8.0)   # heading desde gamma: sigma de cerca
SIGMA_HEADING_SLOPE = math.radians(10.0)  # + rad de sigma por metro (lejos pesa menos)
BEARING_SIGN  = -1.0          # yaw_deg(+)=marcador a la derecha -> bearing(-)
GATE_CHI2_1D  = 6.63          # gate heading (1 GL, 99%)
GATE_CHI2_2D  = 9.21          # gate rango-bearing (2 GL, 99%)
MAX_USE_RANGE = 4.0
HEADING_MAX_RANGE = 3.0       # arriba de esto ni se usa el gamma para heading
INIT_MAX_RANGE    = 1.5       # init confiable solo con marcadores asi de cerca

# re-localizacion (recuperarse si el filtro se pierde)
RELOC_STD_M  = 1.0            # si la incertidumbre de posicion supera esto -> relocaliza
RELOC_STREAK = 10             # ciclos seguidos con marcador visible pero todo rechazado

# visualizacion
_WX_MIN, _WX_MAX = -0.3, 4.1
_WY_MIN, _WY_MAX = -0.3, 5.2
MAP_SIZE, MAP_PADDING = 700, 50
_WX_RANGE = max(_WX_MAX - _WX_MIN, 1.0)
_WY_RANGE = max(_WY_MAX - _WY_MIN, 1.0)
COL_BG, COL_GRID = (30, 30, 30), (55, 55, 55)
COL_ARUCO_UNK, COL_ARUCO_VIS = (100, 100, 100), (50, 220, 50)
COL_ODO, COL_KF, COL_LINE, COL_ELLIPSE = (255, 180, 50), (50, 180, 255), (100, 180, 100), (180, 140, 255)

# logos: posicion DESCONOCIDA. NO corrigen el EKF. Solo nos interesa SI uno cae
# sobre un espacio de puerta FIJO (problema inverso al de los aruco): proyectamos
# la deteccion a mundo con la pose del EKF y, si aterriza encima de un espacio
# conocido, le sumamos un voto a esa puerta. Si cae en otra pared/rack -> se ignora.
LOGO_MIN_RANGE = 0.10
LOGO_MAX_RANGE = 4.0
LOGO_MIN_CONF  = 0.80    # confianza minima de YOLO para considerar la deteccion

# ── pared y=0: SOLO 3 puertas. La x de cada una es la posicion del robot (KF x) medida
# con el robot parado ENFRENTE de cada puerta. Cada caja sobresale hacia AFUERA (y<0),
# como los cuadros azules. d4 = mayor x (izquierda del robot, que lee la pared mirando -y),
# d2 = menor x. Si una puerta quedo corrida, edita su numero en DOOR_CENTERS.
BOX_W     = 0.36
BOX_DEPTH = 0.25                    # cuanto sobresalen hacia afuera (m, hacia y<0)
DOOR_CENTERS = (2.03, 2.57, 3.18)   # x de robot enfrente de puerta 1, 2, 3 (orden creciente)

WALL_BOXES = [(c - BOX_W / 2, c + BOX_W / 2) for c in DOOR_CENTERS]
_cx = sorted(DOOR_CENTERS)
DOOR_SPACES = {'d2': _cx[0], 'd3': _cx[1], 'd4': _cx[2]}

DOOR_SNAP_RADIUS = 0.25  # dist 2D (m) al espacio fijo mas cercano; fuera de esto = mal lugar, se ignora
COL_DOOR         = (80, 220, 255)
COL_BOX          = (230, 130, 40)  # azul (BGR) para las cajas de la pared
DOOR_MIN_VOTE    = 30     # frames acumulados (a 20Hz ~1s) antes de fijar una puerta
DOOR_MARGIN      = 2.0    # la marca ganadora debe tener >= 2x los votos de la 2da (dominancia clara)
DOOR_MIN_VOTE    = 30     # frames acumulados (a 20Hz ~1s) antes de fijar una puerta
DOOR_MARGIN      = 2.0    # la marca ganadora debe tener >= 2x los votos de la 2da (dominancia clara)

# no votar hasta que el EKF este CONVERGIDO y estable: la posicion del logo sale de
# range*sin(heading), asi que un heading malo de arranque lo manda al espacio vecino.
EKF_VOTE_STD       = 0.20  # std de posicion (m) por debajo del cual el filtro es confiable
EKF_MIN_CONV_CYCLES = 20   # ciclos seguidos por debajo del umbral antes de empezar a votar

# QR: igual que los logos (posicion DESCONOCIDA, se mapean con la pose del EKF).
# El qr_detector manda distance_m y yaw_deg; aqui proyectamos a mundo y promediamos.
COL_QR, COL_QR_VIS = (120, 255, 200), (160, 255, 220)
QR_EMA       = 0.3
QR_MIN_RANGE = 0.10
QR_MAX_RANGE = 3.0
QR_MIN_SIGHT = 4

# estantes (racks con pallets). Solo decorativos: NO entran al EKF, son referencia visual.
# Los dos verticales salen EXACTO de los arucos: 7 (arriba) y 6 (abajo) comparten x=1.2430;
# 8 (arriba) y 5 (abajo) comparten x=2.3795; el rack va entre y=2.505 y y=3.565.
# El horizontal de abajo es APROXIMADO (medido a ojo del plano); ajusta los 4 numeros a gusto.
SHELF_W = 0.18   # grosor en x de los racks verticales (m)
SHELVES = [
    # (x_min, y_min, x_max, y_max) en mundo
    (1.2430 - SHELF_W / 2, 2.505, 1.2430 + SHELF_W / 2, 3.565),  # arucos 7 / 6
    (2.3795 - SHELF_W / 2, 2.505, 2.3795 + SHELF_W / 2, 3.565),  # arucos 8 / 5
    (1.16, 1.25, 2.46, 1.55),                                    # horizontal de abajo (aprox)
]
COL_SHELF_FILL = (45, 55, 80)    # relleno tenue
COL_SHELF      = (90, 120, 170)  # borde, cafe-naranja para no chocar con aruco/door/qr


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ══════════════════════════════════════════════════════════════════
#  EKF range-bearing con offset de camara
# ══════════════════════════════════════════════════════════════════

class EKF:
    def __init__(self):
        self.x = np.zeros(3)
        self.P = np.diag([1.0, 1.0, 1.0])
        self.initialized = False

    def init_pose(self, x, y, th, pos_sigma=0.10, th_sigma=math.radians(15)):
        self.x = np.array([x, y, th], dtype=float)
        self.P = np.diag([pos_sigma ** 2, pos_sigma ** 2, th_sigma ** 2])
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
        trans, rot = abs(v) * dt, abs(w) * dt
        Qd = np.diag([(Q_TRANS_K * trans + Q_TRANS_B) ** 2,
                      (Q_TRANS_K * trans + Q_TRANS_B) ** 2,
                      (Q_ROT_K * rot + Q_ROT_B) ** 2])
        self.P = Fx @ self.P @ Fx.T + Qd

    def update_landmark(self, mx, my, z_r, z_b, sigma_r):
        if not self.initialized:
            return None
        x, y, th = self.x
        L = CAM_FORWARD
        c, s = math.cos(th), math.sin(th)
        cx = x + L * c          # posicion de la camara, no del centro del robot
        cy = y + L * s
        dx, dy = mx - cx, my - cy
        q = dx * dx + dy * dy
        r = math.sqrt(q)
        if r < 1e-3:
            return None
        z_hat = np.array([r, wrap_angle(math.atan2(dy, dx) - th)])
        H = np.array([
            [-dx / r, -dy / r,  L * (dx * s - dy * c) / r],
            [ dy / q, -dx / q, -L * (dx * c + dy * s) / q - 1.0],
        ])
        R = np.diag([sigma_r ** 2, SIGMA_BEARING ** 2])
        innov = np.array([z_r - z_hat[0], wrap_angle(z_b - z_hat[1])])
        S = H @ self.P @ H.T + R
        Sinv = np.linalg.inv(S)
        if float(innov @ Sinv @ innov) > GATE_CHI2_2D:
            return False
        K = self.P @ H.T @ Sinv
        self.x = self.x + K @ innov
        self.x[2] = wrap_angle(self.x[2])
        self.P = (np.eye(3) - K @ H) @ self.P
        return True

    def update_heading(self, z_th, sigma_th):
        # mide theta directo desde la orientacion del marcador (m_yaw + gamma).
        # solo afecta theta (H=[0,0,1]) -> hace el heading observable con 1 marcador
        # sin reconstruir posicion con gamma (por eso no regresa el bug del orbitar).
        if not self.initialized:
            return None
        H = np.array([[0.0, 0.0, 1.0]])
        R = np.array([[sigma_th ** 2]])
        innov = np.array([wrap_angle(z_th - self.x[2])])
        S = H @ self.P @ H.T + R
        Sinv = np.linalg.inv(S)
        if float(innov @ Sinv @ innov) > GATE_CHI2_1D:
            return False
        K = self.P @ H.T @ Sinv
        self.x = self.x + (K @ innov)
        self.x[2] = wrap_angle(self.x[2])
        self.P = (np.eye(3) - K @ H) @ self.P
        return True

    @property
    def state(self):
        return float(self.x[0]), float(self.x[1]), float(self.x[2])


# ══════════════════════════════════════════════════════════════════
#  Visualizacion
# ══════════════════════════════════════════════════════════════════

def _scale():
    return min((MAP_SIZE - 2 * MAP_PADDING) / _WX_RANGE,
               (MAP_SIZE - 2 * MAP_PADDING) / _WY_RANGE)


def world_to_map(wx, wy):
    s = _scale()
    return (int(MAP_PADDING + (wx - _WX_MIN) * s),
            int(MAP_SIZE - MAP_PADDING - (wy - _WY_MIN) * s))


def draw_cov_ellipse(canvas, x, y, P2, color):
    try:
        vals, vecs = np.linalg.eigh(P2)
    except np.linalg.LinAlgError:
        return
    vals = np.clip(vals, 1e-6, None)
    ang = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    axes = (int(2 * math.sqrt(vals[0]) * _scale()),
            int(2 * math.sqrt(vals[1]) * _scale()))
    cv2.ellipse(canvas, world_to_map(x, y), axes, -ang, 0, 360, color, 1, cv2.LINE_AA)


def draw_map(ekf, odo, trail_kf, trail_odo, visible, meas_r, initialized, door_map=None, qrs=None):
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
    cv2.rectangle(canvas, world_to_map(0.0, 0.0), world_to_map(3.6645, 4.863), (80, 80, 80), 2)

    # estantes (racks con pallets): relleno tenue + borde. Se dibujan temprano para que
    # trails, marcadores y lineas queden ENCIMA.
    for (xa, ya, xb, yb) in SHELVES:
        cv2.rectangle(canvas, world_to_map(xa, ya), world_to_map(xb, yb), COL_SHELF_FILL, -1)
        cv2.rectangle(canvas, world_to_map(xa, ya), world_to_map(xb, yb), COL_SHELF, 2)

    # cajas/espacios sobre la pared y=0 (sobresalen hacia afuera, y<0), como los cuadros azules.
    for (x0, x1) in WALL_BOXES:
        cv2.rectangle(canvas, world_to_map(x0, 0.0), world_to_map(x1, -BOX_DEPTH), COL_BOX, 2)

    # logos: rombo dentro de la caja con logo (B2/B3/B4). Id corto junto al rombo;
    # las marcas asignadas van en una leyenda aparte, legible.
    for dn, dxw in DOOR_SPACES.items():
        px, py = world_to_map(dxw, -BOX_DEPTH / 2)
        cv2.drawMarker(canvas, (px, py), COL_DOOR, cv2.MARKER_DIAMOND, 16, 2)
        cv2.putText(canvas, dn, (px - 9, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_DOOR, 1)
    # leyenda de asignaciones (esquina superior izquierda)
    cv2.putText(canvas, 'Puertas:', (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_DOOR, 1, cv2.LINE_AA)
    for i, dn in enumerate(('d4', 'd3', 'd2')):
        marca = door_map.get(dn) if door_map else None
        txt = f'{dn}: {marca}' if marca else f'{dn}: ---'
        cv2.putText(canvas, txt, (18, 48 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_DOOR, 1, cv2.LINE_AA)

    for pt in trail_odo:
        cv2.circle(canvas, world_to_map(pt[0], pt[1]), 2, COL_ODO, -1)
    for pt in trail_kf:
        cv2.circle(canvas, world_to_map(pt[0], pt[1]), 2, COL_KF, -1)

    for mid, (mx, my, mth) in ARUCO_MAP.items():
        px, py = world_to_map(mx, my)
        color = COL_ARUCO_VIS if mid in visible else COL_ARUCO_UNK
        cv2.rectangle(canvas, (px - 8, py - 8), (px + 8, py + 8), color, -1)
        cv2.arrowedLine(canvas, (px, py),
                        (int(px + 16 * math.cos(mth)), int(py - 16 * math.sin(mth))),
                        color, 2, tipLength=0.4)
        cv2.putText(canvas, str(mid), (px + 10, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    if not initialized:
        cv2.putText(canvas, "Esperando primer Aruco para inicializar...",
                    (MAP_PADDING, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        return canvas

    kx, ky, kth = ekf.state
    rxk, ryk = world_to_map(kx, ky)
    for mid in visible:
        if mid in ARUCO_MAP:
            mpx, mpy = world_to_map(ARUCO_MAP[mid][0], ARUCO_MAP[mid][1])
            cv2.line(canvas, (rxk, ryk), (mpx, mpy), COL_LINE, 1, cv2.LINE_AA)
            if mid in meas_r:
                cv2.putText(canvas, f"{meas_r[mid]:.2f}m",
                            ((rxk + mpx) // 2 + 3, (ryk + mpy) // 2 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 255, 180), 1)
    draw_cov_ellipse(canvas, kx, ky, ekf.P[:2, :2], COL_ELLIPSE)

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
    cv2.putText(canvas, f"KF:  x={kx:.2f} y={ky:.2f} th={math.degrees(kth):.1f}",
                (10, MAP_SIZE - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_KF, 1)
    cv2.putText(canvas, f"Odo: x={ox:.2f} y={oy:.2f} th={math.degrees(oth):.1f}",
                (10, MAP_SIZE - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_ODO, 1)

    # QR descubiertos (estrella): solo se dibuja MIENTRAS se ve. Asi no quedan fantasmas
    # regados por el mapa cuando el robot deja de detectarlos.
    if qrs:
        for qid, e in qrs.items():
            if not e['visible'] or e['n'] < QR_MIN_SIGHT:
                continue
            qx, qy = world_to_map(e['x'], e['y'])
            cv2.drawMarker(canvas, (qx, qy), COL_QR_VIS, cv2.MARKER_STAR, 16, 2)
            cv2.putText(canvas, str(qid), (qx + 8, qy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_QR_VIS, 1)
            cv2.line(canvas, (rxk, ryk), (qx, qy), COL_QR_VIS, 1, cv2.LINE_AA)
    return canvas


# ══════════════════════════════════════════════════════════════════
#  Nodo
# ══════════════════════════════════════════════════════════════════

class PoseKalmanNode(Node):
    def __init__(self):
        super().__init__('pose_kalman')

        self.declare_parameter('show_view', True)
        self.show_view = bool(self.get_parameter('show_view').value)
        self.declare_parameter('debug', False)
        self.debug = bool(self.get_parameter('debug').value)
        self._dbg_n = 0
        self.map_pub = self.create_publisher(CompressedImage, '/pose_map/compressed', 10)

        self.pose_pub = self.create_publisher(Odometry, '/odom', 10)
        self.door_pub = self.create_publisher(String, '/logo/door_map', 10)
        self.qr_pub = self.create_publisher(String, '/qr/map', 10)
        self.create_subscription(String, '/aruco/detections', self.aruco_cb, 10)
        self.create_subscription(String, '/logo_detections', self.logo_cb, 10)
        self.create_subscription(String, '/qr/detections', self.qr_cb, 10)
        self.create_subscription(Float32, '/VelocityEncR', self.encR_cb, qos.qos_profile_sensor_data)
        self.create_subscription(Float32, '/VelocityEncL', self.encL_cb, qos.qos_profile_sensor_data)

        self.wr = 0.0
        self.wl = 0.0
        self.last_dets = []
        self.last_logos = []
        self.door_votes = {dn: Counter() for dn in DOOR_SPACES}  # puerta -> conteo de marcas
        self.door_map = {}   # puerta -> marca confirmada
        self._conv_cycles = 0  # ciclos seguidos con el EKF convergido
        self.last_logo_time = self.get_clock().now()
        self.last_qrs = []
        self.qr_map = {}     # id_qr -> {'x','y','n','dist','visible'}

        self.ekf = EKF()
        self.odo = np.zeros(3)
        self.cur_v = 0.0
        self.cur_w = 0.0
        self.reject_streak = 0
        self._dr_dist = 0.0      # dead-reckoning acumulado (para calibrar escala)
        self._dr_rot = 0.0

        self.trail_kf, self.trail_odo = [], []
        self.MAX_TRAIL = 400
        self.visible, self.meas_r = [], {}

        self.last_time = self.get_clock().now()
        self.create_timer(0.05, self.loop)
        self.get_logger().info('PoseKalman REAL (range-bearing) | esperando primer Aruco')

    def encR_cb(self, msg): self.wr = msg.data
    def encL_cb(self, msg): self.wl = msg.data

    def aruco_cb(self, msg):
        try:
            self.last_dets = json.loads(msg.data)
        except json.JSONDecodeError:
            self.last_dets = []

    def logo_cb(self, msg):
        try:
            self.last_logos = json.loads(msg.data).get('detections', [])
            self.last_logo_time = self.get_clock().now()
        except json.JSONDecodeError:
            self.last_logos = []

    def qr_cb(self, msg):
        try:
            self.last_qrs = json.loads(msg.data).get('qrs', [])
        except json.JSONDecodeError:
            self.last_qrs = []

    def update_logos(self):
        # solo vota cuando el EKF lleva varios ciclos convergido (std bajo). Si aun no,
        # republica el mapa ya bloqueado pero no acumula votos (evita fijar con pose mala).
        age = (
            self.get_clock().now() -
            self.last_logo_time
        ).nanoseconds * 1e-9

        if age > 0.5:
            return

        converged = False
        if self.ekf.initialized:
            std_pos = math.sqrt(self.ekf.P[0, 0] + self.ekf.P[1, 1])
            self._conv_cycles = self._conv_cycles + 1 if std_pos < EKF_VOTE_STD else 0
            converged = self._conv_cycles >= EKF_MIN_CONV_CYCLES

        if converged:
            x, y, th = self.ekf.state
            camx = x + CAM_FORWARD * math.cos(th)
            camy = y + CAM_FORWARD * math.sin(th)
            for d in self.last_logos:
                name, rng, yaw = d.get('class_name'), d.get('distance_m'), d.get('yaw_deg')
                conf = d.get('confidence', 1.0)
                if name is None or rng is None or yaw is None:
                    continue
                if conf < LOGO_MIN_CONF:                       # filtro 1: confianza
                    continue
                if not (LOGO_MIN_RANGE < rng <= LOGO_MAX_RANGE):
                    continue
                wd = th + BEARING_SIGN * math.radians(yaw)     # rumbo robot->logo en mundo
                lx = camx + rng * math.cos(wd)
                ly = camy + rng * math.sin(wd)
                # filtro 2: debe aterrizar ENCIMA de un espacio fijo conocido (dist 2D).
                dn, dd = self._nearest_door(lx, ly)
                if dn is not None and dd < DOOR_SNAP_RADIUS:
                    self.door_votes[dn][name] += 1

        self._update_door_map()

    def _nearest_door(self, lx, ly):
        best, bestd = None, 1e9
        for dn, dxw in DOOR_SPACES.items():
            dd = math.hypot(lx - dxw, ly - 0.0)   # espacios en y=0
            if dd < bestd:
                best, bestd = dn, dd
        return best, bestd

    def _update_door_map(self):
        # asigna marca a puerta por mayoria de votos y la BLOQUEA: una puerta ya asignada
        # nunca se reescribe, y una marca ya tomada no la puede robar otra puerta. Ademas
        # exige DOMINANCIA: la marca top debe llevarle >= DOOR_MARGIN x a la 2da disponible,
        # asi una puerta que ve dos marcas (proyeccion en la frontera) no se fija a la ligera.
        taken = set(self.door_map.values())
        changed = False
        order = sorted(self.door_votes,
                       key=lambda d: -(self.door_votes[d].most_common(1)[0][1]
                                       if self.door_votes[d] else 0))
        for dn in order:
            if dn in self.door_map:                 # ya bloqueada -> no se toca
                continue
            avail = [(n, v) for n, v in self.door_votes[dn].most_common() if n not in taken]
            if not avail:
                continue
            name, votes = avail[0]
            runner = avail[1][1] if len(avail) > 1 else 0
            if votes >= DOOR_MIN_VOTE and votes >= DOOR_MARGIN * runner:
                self.door_map[dn] = name
                taken.add(name)
                changed = True
        if changed:
            self.get_logger().info(f'door_map -> {self.door_map} (bloqueado)')
        if self.door_map:
            self.door_pub.publish(String(data=json.dumps(self.door_map)))

    def update_qrs(self):
        # proyecta cada QR visto a mundo con la pose actual del EKF y promedia (EMA).
        # NO toca el filtro: el QR es de posicion desconocida, lo ubicamos con la pose
        # ya estimada (problema inverso al del aruco). Solo lecturas frescas.
        for e in self.qr_map.values():
            e['visible'] = False
        if not self.ekf.initialized:
            return
        x, y, th = self.ekf.state
        camx = x + CAM_FORWARD * math.cos(th)
        camy = y + CAM_FORWARD * math.sin(th)
        out = {}
        for d in self.last_qrs:
            if not d.get('fresh', True):           # ignora el seguimiento a ciegas
                continue
            qid, rng, yaw = d.get('id'), d.get('distance_m'), d.get('yaw_deg')
            if qid is None or rng is None or yaw is None:
                continue
            if not (QR_MIN_RANGE < rng <= QR_MAX_RANGE):
                continue
            wd = th + BEARING_SIGN * math.radians(yaw)   # rumbo robot->QR en mundo
            qx = camx + rng * math.cos(wd)
            qy = camy + rng * math.sin(wd)
            e = self.qr_map.get(qid)
            if e is None:
                self.qr_map[qid] = {'x': qx, 'y': qy, 'n': 1, 'dist': rng, 'visible': True}
            else:
                e['x'] = (1 - QR_EMA) * e['x'] + QR_EMA * qx
                e['y'] = (1 - QR_EMA) * e['y'] + QR_EMA * qy
                e['n'] += 1
                e['dist'] = rng
                e['visible'] = True
        for k, v in self.qr_map.items():
            if v['n'] >= QR_MIN_SIGHT:
                out[k] = {'x': round(v['x'], 3), 'y': round(v['y'], 3), 'n': v['n']}
        self.qr_pub.publish(String(data=json.dumps(out)))

    def predict_step(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now
        if dt <= 0 or dt > 0.5:
            return
        vr = WHEEL_RADIUS * self.wr
        vl = WHEEL_RADIUS * self.wl
        v = (vr + vl) / 2.0
        w = (vr - vl) / WHEEL_BASE
        self.cur_v, self.cur_w = v, w
        self._dr_dist += v * dt
        self._dr_rot += w * dt

        if self.debug:
            self._dbg_n += 1
            if self._dbg_n % 10 == 0:
                self.get_logger().info(
                    f'wr={self.wr:+.2f} wl={self.wl:+.2f} -> v={v:+.3f} w={math.degrees(w):+.1f}deg/s '
                    f'| acum: dist={self._dr_dist:+.3f}m rot={math.degrees(self._dr_rot):+.1f}deg')

        self.ekf.predict(v, w, dt)
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

    def valid_dets(self):
        out = []
        for d in self.last_dets:
            mid, rng, yaw = d.get('id'), d.get('distance_m'), d.get('yaw_deg')
            if mid in ARUCO_MAP and rng is not None and yaw is not None \
               and 0.05 < rng <= MAX_USE_RANGE:
                out.append(d)
        return out

    def try_init(self, dets):
        # pose completa del/los marcador(es), sin suponer heading.
        # preferir CERCANOS (gamma y rango confiables). si solo hay lejanos,
        # inicializa igual pero con mucha incertidumbre para que se corrija al
        # moverse, en vez de comprometerse a un gamma lejano (que sale mal).
        ds = sorted(dets, key=lambda d: d['distance_m'])
        close = [d for d in ds if d['distance_m'] <= INIT_MAX_RANGE]
        use = close if close else ds[:1]
        confident = bool(close)
        xs, ys, ss, cc = [], [], [], []
        for d in use:
            mx, my, m_yaw = ARUCO_MAP[d['id']]
            rng = d['distance_m']
            gamma = d.get('gamma_deg')
            if gamma is None:
                continue
            th = wrap_angle(m_yaw + math.radians(gamma))
            beta = BEARING_SIGN * math.radians(d['yaw_deg'])
            world_dir = th + beta                 # rumbo camara->marcador
            cam_x = mx - rng * math.cos(world_dir)
            cam_y = my - rng * math.sin(world_dir)
            xs.append(cam_x - CAM_FORWARD * math.cos(th))
            ys.append(cam_y - CAM_FORWARD * math.sin(th))
            ss.append(math.sin(th)); cc.append(math.cos(th))
        if not xs:
            return
        x0 = float(np.median(xs))
        y0 = float(np.median(ys))
        th0 = math.atan2(float(np.mean(ss)), float(np.mean(cc)))
        pos_sigma = 0.10 if confident else 0.30
        th_sigma = math.radians(15) if confident else math.radians(35)
        self.ekf.init_pose(x0, y0, th0, pos_sigma, th_sigma)
        self.odo = np.array([x0, y0, th0], dtype=float)
        self.get_logger().info(
            f'INICIALIZADO desde {[d["id"] for d in use]} '
            f'({"cercano" if confident else "LEJANO, baja confianza"}) | '
            f'x={x0:.3f} y={y0:.3f} th={math.degrees(th0):.1f}deg')

    def correct_step(self, dets):
        self.visible, self.meas_r = [], {}
        accepted = 0
        for d in dets:
            mid = d['id']
            rng = d['distance_m']
            self.visible.append(mid)
            self.meas_r[mid] = rng
            z_b = BEARING_SIGN * math.radians(d['yaw_deg'])
            sigma_r = d.get('range_sigma_m')
            if sigma_r is None:
                sigma_r = SIGMA_R_FLOOR
            sigma_r = max(float(sigma_r), SIGMA_R_FLOOR)   # piso
            mx, my, m_yaw = ARUCO_MAP[mid]
            if self.ekf.update_landmark(mx, my, rng, z_b, sigma_r):
                accepted += 1
            # heading desde la orientacion del marcador, solo si es confiable
            gamma = d.get('gamma_deg')
            if gamma is not None and rng <= HEADING_MAX_RANGE:
                z_th = wrap_angle(m_yaw + math.radians(gamma))
                sigma_th = SIGMA_HEADING_BASE + SIGMA_HEADING_SLOPE * rng
                self.ekf.update_heading(z_th, sigma_th)

        # divergencia -> re-localizar desde los marcadores visibles
        if dets:
            self.reject_streak = self.reject_streak + 1 if accepted == 0 else 0
            std_pos = math.sqrt(self.ekf.P[0, 0] + self.ekf.P[1, 1])
            if self.reject_streak >= RELOC_STREAK or std_pos > RELOC_STD_M:
                self.get_logger().warn(
                    f'divergencia (rechazos={self.reject_streak}, std={std_pos:.2f}m) '
                    f'-> re-localizando desde {[d["id"] for d in dets]}')
                self.try_init(dets)
                self.reject_streak = 0
                # el filtro estaba perdido: los votos acumulados no son de fiar.
                # reinicia conteo y convergencia (lo ya BLOQUEADO en door_map se respeta).
                self.door_votes = {dn: Counter() for dn in DOOR_SPACES}
                self._conv_cycles = 0

    def publish_pose(self):
        if not self.ekf.initialized:
            return
        kx, ky, kth = self.ekf.state
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_footprint'
        msg.pose.pose.position.x = kx
        msg.pose.pose.position.y = ky
        msg.pose.pose.orientation.z = math.sin(kth / 2)
        msg.pose.pose.orientation.w = math.cos(kth / 2)
        msg.twist.twist.linear.x = self.cur_v
        msg.twist.twist.angular.z = self.cur_w
        P = self.ekf.P
        cov = [0.0] * 36
        cov[0], cov[1], cov[5] = P[0, 0], P[0, 1], P[0, 2]
        cov[6], cov[7], cov[11] = P[1, 0], P[1, 1], P[1, 2]
        cov[30], cov[31], cov[35] = P[2, 0], P[2, 1], P[2, 2]
        msg.pose.covariance = cov
        self.pose_pub.publish(msg)

    def loop(self):
        self.predict_step()
        dets = self.valid_dets()

        if not self.ekf.initialized:
            if dets:
                self.try_init(dets)
        else:
            if dets:
                self.correct_step(dets)
            else:
                self.visible, self.meas_r = [], {}

        self.publish_pose()

        self.update_logos()
        self.update_qrs()

        if self.ekf.initialized:
            kx, ky, _ = self.ekf.state
            self.trail_kf.append((kx, ky))
            self.trail_odo.append((self.odo[0], self.odo[1]))
            self.trail_kf = self.trail_kf[-self.MAX_TRAIL:]
            self.trail_odo = self.trail_odo[-self.MAX_TRAIL:]

        if self.show_view:
            img = draw_map(self.ekf, tuple(self.odo), self.trail_kf, self.trail_odo,
                           self.visible, self.meas_r, self.ekf.initialized,
                           self.door_map, self.qr_map)
            cv2.imshow('PoseKalman - Mapa', img)
            if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                raise KeyboardInterrupt
            ret, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret:
                cmsg = CompressedImage()
                cmsg.header.stamp = self.get_clock().now().to_msg()
                cmsg.format = 'jpeg'
                cmsg.data = buf.tobytes()
                self.map_pub.publish(cmsg)


def main(args=None):
    rclpy.init(args=args)
    node = PoseKalmanNode()
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