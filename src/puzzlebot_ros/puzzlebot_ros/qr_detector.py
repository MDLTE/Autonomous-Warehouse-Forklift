#!/usr/bin/env python3
"""
qr_detector.py — Detector de QR standalone para el Puzzlebot REAL

La parte de VISION del nodo de alineacion, sacada a su propio nodo (igual que
aruco_detector). Solo percepcion: se suscribe a la camara, decodifica el QR
(WeChat + QRCodeDetector + enderezado de ROI), saca la pose por solvePnP
(IPPE_SQUARE) y publica /qr/detections (String/JSON). NO mueve el robot.

Lo consumen dos nodos:
  - qr_alignment_node: usa el QR 'locked' para alinear e insertar.
  - poseKalman: proyecta cada QR a mundo con la pose del EKF y lo ubica en el mapa.

Tamano auto (9cm vs 4.5cm): arranca asumiendo grande; con lecturas limpias
estables decide por distancia (el chico se ve a la mitad -> reporta ~2x). El
tamano cambia la escala de la pose, asi que se resuelve aqui, no en el control.

Mensaje (String/JSON):
  {"locked_id": "Emezon",
   "qrs": [ {id, dist, bearing_deg, psi_deg, e_lat, gamma_deg, tx, tz, nx, nz,
             distance_m, yaw_deg, range_sigma_m, size, size_m, fresh, locked}, ... ]}
distance_m y yaw_deg salen con la MISMA convencion que aruco_detector
(yaw_deg = atan2(tx,tz)) para que poseKalman los trate igual que un logo.
"""

import json
import math
import os
import numpy as np
import cv2
import rclpy
from rclpy import qos
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String

# calibracion real (la misma del nodo de alineacion)
CAMERA_MATRIX = np.array([
    [1305.61,    0.0,   640.84],
    [   0.0,  1305.89,  357.29],
    [   0.0,     0.0,     1.0 ],
], dtype=np.float64)
DIST_COEFFS = np.array([[0.13807, -0.36743, 0.01148, -0.00193, 0.0]], dtype=np.float64)

BIG_SIZE   = 0.09          # QR grande (m)
SMALL_SIZE = 0.045         # QR chico (m)
SIZE_SPLIT = 0.95          # dist (asumiendo grande) > esto -> es chico
CAMERA_PITCH_DEG = 0.0     # camara nivelada en el robot real
PIXEL_NOISE_PX = 1.0       # para el sigma de rango (R del EKF)
MIN_MARKER_PX = 20.0       # arista minima en px para confiar en la pose

WHITELIST = {'Emezon', 'Wolmar', 'Popsi'}

class QRDetector(Node):
    def __init__(self):
        super().__init__('qr_detector')

        self.declare_parameter('image_topic', '/video_source/compressed')
        self.declare_parameter('detections_topic', '/qr/detections')
        self.declare_parameter('show_view', True)
        self.declare_parameter('wechat_dir', os.path.expanduser('~/wechat_models'))

        self.image_topic = self.get_parameter('image_topic').value
        self.det_topic   = self.get_parameter('detections_topic').value
        self.show_view   = bool(self.get_parameter('show_view').value)

        self.K = CAMERA_MATRIX
        self.D = DIST_COEFFS
        p = math.radians(CAMERA_PITCH_DEG)
        self.R_level = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)],
                                 [0, math.sin(p), math.cos(p)]], dtype=np.float64)

        # tamano (escala de la pose) — se autodetecta
        self.qr_size = BIG_SIZE
        self.size_locked = False
        self.size_dist_buf = []
        self.use_subpix = False
        self._build_obj(self.qr_size)

        # lock por id + continuidad (igual que el nodo viejo)
        self.last_qr_center = None
        self.lock_age = 999
        self.lock_max = 15
        self.lock_radius_px = 140
        self.locked_id = ''
        self.prev_gamma = None
        self.lost_reset = 45        # frames sin QR -> reevalua tamano en el siguiente

        self._gray = None

        # decoders
        self.qr = cv2.QRCodeDetector()
        self.wechat = None
        mdir = self.get_parameter('wechat_dir').get_parameter_value().string_value
        files = [os.path.join(mdir, f) for f in
                 ('detect.prototxt', 'detect.caffemodel', 'sr.prototxt', 'sr.caffemodel')]
        if hasattr(cv2, 'wechat_qrcode') and all(os.path.isfile(f) for f in files):
            try:
                self.wechat = cv2.wechat_qrcode.WeChatQRCode(*files)
                self.get_logger().info(f'WeChat QRCode activo ({mdir})')
            except Exception as e:
                self.get_logger().warn(f'WeChat no cargo ({e}); uso QRCodeDetector')
        else:
            self.get_logger().info('WeChat no disponible; uso QRCodeDetector + enderezado')

        self.create_subscription(CompressedImage, self.image_topic,
                                 self.image_cb, qos.qos_profile_sensor_data)
        self.create_subscription(Bool, '/qr_detector/reset', self._reset_cb, 10)
        self.pub = self.create_publisher(String, self.det_topic, 10)

        if self.show_view:
            cv2.namedWindow('qr_detector', cv2.WINDOW_NORMAL)
        self.get_logger().info('qr_detector listo')

    # ---------- utilidades ----------
    def _build_obj(self, size):
        h = size / 2.0
        self.qr_size = size
        self.obj_pts = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]],
                                dtype=np.float32)

    def _reset_cb(self, msg):
        if msg.data:
            self._reset_size()
            self.last_qr_center = None
            self.lock_age = 999
            self.locked_id = ''
            self.prev_gamma = None

    def _reset_size(self):
        self._build_obj(BIG_SIZE)
        self.size_locked = False
        self.use_subpix = False
        self.size_dist_buf = []

    @staticmethod
    def _wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _ang_diff(self, a, b):
        return abs(self._wrap(a - b))

    def _gamma_rad(self, rvec):
        R, _ = cv2.Rodrigues(rvec)
        return self._wrap(math.atan2(R[0, 2], R[2, 2]) + math.pi)

    # ---------- decodificacion ----------
    def _decode_roi(self, gray, pts):
        # endereza el QR a un cuadro frontal y lo amplia, para recuperar lecturas
        # chicas/lejanas/oblicuas
        try:
            p = np.asarray(pts, np.float32).reshape(-1, 2)[:4]
            S, pad = 320, 40
            dst = np.array([[pad, pad], [S - pad, pad], [S - pad, S - pad], [pad, S - pad]], np.float32)
            warp = cv2.warpPerspective(gray, cv2.getPerspectiveTransform(p, dst), (S, S),
                                       flags=cv2.INTER_CUBIC, borderValue=255)
            txt, _, _ = self.qr.detectAndDecode(warp)
            if txt.strip():
                return txt.strip()
            xs, ys = p[:, 0], p[:, 1]
            bw, bh = float(xs.max() - xs.min()), float(ys.max() - ys.min())
            m = int(0.35 * max(bw, bh)) + 4
            h, w = gray.shape[:2]
            x0 = int(np.clip(xs.min() - m, 0, w - 1)); x1 = int(np.clip(xs.max() + m, 0, w))
            y0 = int(np.clip(ys.min() - m, 0, h - 1)); y1 = int(np.clip(ys.max() + m, 0, h))
            if x1 - x0 < 8 or y1 - y0 < 8:
                return ''
            sc = float(np.clip(300.0 / max(bw, bh, 1), 1.0, 6.0))
            up = cv2.resize(gray[y0:y1, x0:x1], None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
            txt, _, _ = self.qr.detectAndDecode(up)
            return txt.strip()
        except Exception:
            return ''

    def _gather(self, gray):
        # candidatos crudos (texto, pts4, centro). texto puede venir vacio.
        cands = []
        if self.wechat is not None:
            try:
                texts, points = self.wechat.detectAndDecode(gray)
                for t, pp in zip(texts, points):
                    pp = np.asarray(pp, dtype=np.float64).reshape(-1, 2)
                    if pp.shape[0] >= 4:
                        cands.append((t.strip(), pp[:4], pp[:4].mean(axis=0)))
            except Exception as e:
                self.get_logger().warn(f'wechat: {e}', throttle_duration_sec=2.0)
        if not any(t in WHITELIST for t, _, _ in cands):
            try:
                retval, infos, points, _ = self.qr.detectAndDecodeMulti(gray)
                if retval and points is not None:
                    for i in range(len(points)):
                        pts_i = np.asarray(points[i], dtype=np.float64).reshape(-1, 2)
                        if pts_i.shape[0] < 4:
                            continue
                        text = infos[i] if (infos is not None and i < len(infos)) else ''
                        cands.append((text.strip(), pts_i[:4], pts_i[:4].mean(axis=0)))
            except Exception as e:
                self.get_logger().warn(f'detect: {e}', throttle_duration_sec=1.0)
        return cands

    def _decoded_whitelist(self, gray, cands):
        # texto valido por decodificacion directa; si no, reintenta enderezando ROI.
        # devuelve [(id, pts, center), ...] (uno por QR leido este frame)
        out, seen = [], []
        for text, pts_i, center in cands:
            if text in WHITELIST:
                out.append((text, pts_i, center)); seen.append(center)
        order = cands
        if self.last_qr_center is not None:
            order = sorted(cands, key=lambda c: float(np.linalg.norm(c[2] - self.last_qr_center)))
        for _t, pts_i, center in order:
            if any(np.linalg.norm(center - c) < 5 for c in seen):
                continue
            txt = self._decode_roi(gray, pts_i)
            if txt in WHITELIST:
                out.append((txt, pts_i, center)); seen.append(center)
        return out

    # ---------- pose ----------
    def compute_pose(self, img_pts, update_gamma):
        img_pts = np.asarray(img_pts, np.float32).reshape(4, 2)
        if self.use_subpix and self._gray is not None:
            try:
                ref = img_pts.reshape(-1, 1, 2).copy()
                cv2.cornerSubPix(self._gray, ref, (5, 5), (-1, -1),
                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
                img_pts = ref.reshape(4, 2)
            except Exception:
                pass
        edge = (np.linalg.norm(img_pts[2] - img_pts[1]) +
                np.linalg.norm(img_pts[0] - img_pts[3])) / 2.0
        if edge < MIN_MARKER_PX:
            return None
        try:
            n, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
                self.obj_pts, img_pts, self.K, self.D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except Exception:
            return None
        if n == 0:
            return None
        if n >= 2 and self.prev_gamma is not None:
            g = [self._gamma_rad(rvecs[i]) for i in range(n)]
            best_i = min(range(n), key=lambda i: self._ang_diff(g[i], self.prev_gamma))
        else:
            best_i = int(np.argmin(np.array(reproj).ravel()))
        rvec, tvec = rvecs[best_i], tvecs[best_i]
        R, _ = cv2.Rodrigues(rvec)
        gamma = self._gamma_rad(rvec)
        if update_gamma:
            self.prev_gamma = gamma

        t = (self.R_level @ tvec.reshape(3)).ravel()
        tx, tz = float(t[0]), float(t[2])
        dist = math.hypot(tx, tz)
        bearing = math.atan2(tx, tz)
        nrm = R[:, 2].astype(float)
        if (nrm[0] * (-tx) + nrm[2] * (-tz)) < 0:
            nrm = -nrm
        nx, nz = nrm[0], nrm[2]
        ln = math.hypot(nx, nz) + 1e-9
        nx, nz = nx / ln, nz / ln
        psi = math.atan2(-nx, -nz)
        e_lat = (-tx * nz + tz * nx)
        fx = self.K[0, 0]
        range_sigma = (dist ** 2) / (fx * self.qr_size) * PIXEL_NOISE_PX
        return dict(dist=dist, bearing=bearing, gamma=gamma, psi=psi, e_lat=e_lat,
                    tx=tx, tz=tz, nx=nx, nz=nz, range_sigma=range_sigma,
                    rvec=rvec, tvec=tvec, pts=img_pts)

    def _maybe_lock_size(self, dist, fresh):
        # arranca grande; con lecturas limpias estables decide por distancia
        if self.size_locked or not fresh:
            return
        self.size_dist_buf.append(dist)
        if len(self.size_dist_buf) >= 5:
            d = float(np.median(self.size_dist_buf))
            chico = d > SIZE_SPLIT
            self._build_obj(SMALL_SIZE if chico else BIG_SIZE)
            self.use_subpix = chico
            self.size_locked = True
            self.size_dist_buf = []
            self.get_logger().info(
                f'QR {"chico (4.5cm)" if chico else "grande (9.0cm)"} por distancia, dist~{d:.2f}')

    def _entry(self, qid, pose, fresh, locked):
        size_tag = 'small' if self.qr_size < BIG_SIZE else 'big'
        f = lambda v: round(float(v), 4)
        return {
            'id': qid,
            'dist': f(pose['dist']),
            'bearing_deg': round(math.degrees(pose['bearing']), 2),
            'psi_deg': round(math.degrees(pose['psi']), 2),
            'e_lat': f(pose['e_lat']),
            'gamma_deg': round(math.degrees(pose['gamma']), 2),
            'tx': f(pose['tx']),
            'tz': f(pose['tz']),
            'nx': f(pose['nx']),
            'nz': f(pose['nz']),
            'distance_m': f(pose['dist']),                        # alias para poseKalman
            'yaw_deg': round(math.degrees(pose['bearing']), 2),   # convencion aruco
            'range_sigma_m': f(pose['range_sigma']),
            'size': size_tag,
            'size_m': float(self.qr_size),
            'corners': pose['pts'].tolist(),
            'img_w': int(self.K[0,2]*2),
            'img_h': int(self.K[1,2]*2),
            'fresh': bool(fresh),
            'locked': bool(locked),
        }

    # ---------- loop ----------
    def image_cb(self, msg: CompressedImage):
        try:
            frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().warn(f'decode: {e}')
            return
        if frame is None:
            return
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        self._gray = gray

        cands = self._gather(gray)
        decoded = self._decoded_whitelist(gray, cands)

        qrs = []
        locked_pose = None

        if decoded:
            self.lock_age = 0
            # el 'locked' es el mas cercano al lock anterior; si no hay, el primero
            if self.last_qr_center is not None:
                decoded.sort(key=lambda c: float(np.linalg.norm(c[2] - self.last_qr_center)))
            lk_id, lk_pts, lk_center = decoded[0]
            self.last_qr_center = lk_center
            self.locked_id = lk_id

            for qid, pts_i, center in decoded:
                is_locked = (qid == lk_id and np.allclose(center, lk_center))
                pose = self.compute_pose(pts_i, update_gamma=is_locked)
                if pose is None:
                    continue
                if is_locked:
                    self._maybe_lock_size(pose['dist'], fresh=True)
                    locked_pose = pose
                qrs.append(self._entry(qid, pose, fresh=True, locked=is_locked))
        else:
            # continuidad: si hay lock reciente y un candidato cerca, lo seguimos a
            # ciegas (sin id nuevo) para que el control no pierda el QR de cerca
            self.lock_age += 1
            if (self.last_qr_center is not None and self.lock_age < self.lock_max
                    and cands and self.locked_id):
                best, bestd, bestc = None, 1e9, None
                for _t, pts_i, center in cands:
                    dd = float(np.linalg.norm(center - self.last_qr_center))
                    if dd < bestd:
                        best, bestd, bestc = pts_i, dd, center
                if best is not None and bestd < self.lock_radius_px:
                    self.last_qr_center = bestc
                    pose = self.compute_pose(best, update_gamma=True)
                    if pose is not None:
                        locked_pose = pose
                        qrs.append(self._entry(self.locked_id, pose, fresh=False, locked=True))
            if self.lock_age >= self.lost_reset:
                self._reset_size()
                self.locked_id = ''
                self.last_qr_center = None
                self.prev_gamma = None

        out = String()
        out.data = json.dumps({'locked_id': self.locked_id, 'qrs': qrs})
        self.pub.publish(out)

        if self.show_view:
            self._show(frame, qrs, locked_pose)

    def _show(self, frame, qrs, locked_pose):
        h, w = frame.shape[:2]
        cx = int(round(self.K[0, 2]))
        cv2.line(frame, (cx, 0), (cx, h), (255, 255, 255), 1)
        if locked_pose is not None:
            cv2.polylines(frame, [locked_pose['pts'].astype(int)], True, (0, 255, 0), 2)
            try:
                cv2.drawFrameAxes(frame, self.K, self.D, locked_pose['rvec'],
                                  locked_pose['tvec'], self.qr_size * 0.5, 2)
            except cv2.error:
                pass
        y = 24
        for q in qrs:
            tag = '*' if q['locked'] else ' '
            cv2.putText(frame, f"{tag}{q['id']} {q['dist']:.2f}m b={q['bearing_deg']:+.0f}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            y += 26
        cv2.imshow('qr_detector', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = QRDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.show_view:
            cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()