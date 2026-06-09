#!/usr/bin/env python3
"""
aruco_detector_sim.py
Detector ArUco para la camara simulada del PuzzleBot (mundo e80).

Distancia y angulo con solvePnP (IPPE_SQUARE). En sim la camara es perfecta,
asi que la pose sale exacta si K y marker_size son correctos.

CALIBRADO PARA NUESTRO MUNDO (slam_pkg/aruco_models):
  - La cara del modelo mide 15 cm, PERO la textura tiene 1 celda de quiet
    zone blanca por lado. El marcador 4x4_50 son 6x6 celdas y el PNG son
    8x8, asi que el patron negro real es 6/8 * 0.15 = 0.1125 m.
    Por eso marker_size = 0.1125 (no 0.15). Usar 0.15 inflaba todo x1.333.
  - K confirmada por /camera/camera_info: fx=fy=277.19, 320x240, dist 0.
  - Marcadores a z=0.20, camara a z=0.11 -> 9 cm de diferencia de altura.

distance_m que publicamos = rango HORIZONTAL en el piso (range_xz), que es
lo que poseKalmanSim usa para triangular (rx = mx + distance*cos(a)). Quita
la componente vertical, asi que la diferencia de altura no estorba.

Publica /aruco/detections (std_msgs/String, JSON list).
"""

import json
import math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge

# fallback por si camera_info no llega (320x240, fov 60 deg)
FALLBACK_K = np.array([
    [277.19135641132203, 0.0,                160.5],
    [0.0,                277.19135641132203, 120.5],
    [0.0,                0.0,                 1.0],
], dtype=np.float64)
FALLBACK_D = np.zeros((1, 5), dtype=np.float64)

ARUCO_DICT = cv2.aruco.DICT_4X4_50
VALID_IDS  = set(range(0, 11))   # IDs 0-10 del mapa e80


class ArucoDetectorSim(Node):

    def __init__(self):
        super().__init__('aruco_detector_sim')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('detections_topic', '/aruco/detections')
        self.declare_parameter('marker_size', 0.1125)   # patron negro real (6/8 * 0.15)
        self.declare_parameter('use_camera_info', True)
        self.declare_parameter('equalize', False)
        self.declare_parameter('show_view', True)
        self.declare_parameter('min_marker_px', 8.0)    # mas chico -> distancia basura
        self.declare_parameter('max_range_m', 0.0)      # 0 = sin tope
        self.declare_parameter('pixel_noise_px', 0.7)   # ruido de corner para sigma
        self.declare_parameter('gt_distance', 0.0)      # >0 enciende diagnostico

        self.image_topic     = self.get_parameter('image_topic').value
        self.info_topic      = self.get_parameter('camera_info_topic').value
        self.det_topic       = self.get_parameter('detections_topic').value
        self.marker_size     = float(self.get_parameter('marker_size').value)
        self.use_camera_info = bool(self.get_parameter('use_camera_info').value)
        self.equalize        = bool(self.get_parameter('equalize').value)
        self.show_view       = bool(self.get_parameter('show_view').value)
        self.min_marker_px   = float(self.get_parameter('min_marker_px').value)
        self.max_range_m     = float(self.get_parameter('max_range_m').value)
        self.pixel_noise_px  = float(self.get_parameter('pixel_noise_px').value)
        self.gt_distance     = float(self.get_parameter('gt_distance').value)

        self.K = FALLBACK_K.copy()
        self.D = FALLBACK_D.copy()
        self.have_info = not self.use_camera_info
        self.warned_res = False
        self.bridge = CvBridge()

        self._build_obj_pts()
        self._setup_detector()

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.sub = self.create_subscription(
            Image, self.image_topic, self.image_cb, sensor_qos)
        if self.use_camera_info:
            self.sub_info = self.create_subscription(
                CameraInfo, self.info_topic, self.info_cb, sensor_qos)

        self.pub = self.create_publisher(String, self.det_topic, 10)

        if self.show_view:
            cv2.namedWindow('ArUco Sim', cv2.WINDOW_NORMAL)

        self.get_logger().info(
            f'aruco_detector_sim listo | marker={self.marker_size*100:.2f}cm | '
            f'use_camera_info={self.use_camera_info}')
        if self.gt_distance > 0:
            self.get_logger().warn(
                f'>> DIAGNOSTICO ON: deja UN marcador a {self.gt_distance:.3f} m '
                'y revisa el log')

    def _build_obj_pts(self):
        h = self.marker_size / 2.0
        self.obj_pts = np.array([
            [-h,  h, 0],
            [ h,  h, 0],
            [ h, -h, 0],
            [-h, -h, 0],
        ], dtype=np.float32)

    def _setup_detector(self):
        try:
            d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
            p = cv2.aruco.DetectorParameters()
            p.minMarkerPerimeterRate = 0.01
            p.maxMarkerPerimeterRate = 4.0
            try:
                p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                p.cornerRefinementWinSize = 5
                p.cornerRefinementMaxIterations = 30
                p.cornerRefinementMinAccuracy = 0.01
            except Exception:
                pass
            self.detector = cv2.aruco.ArucoDetector(d, p)
            self.use_new_api = True
        except AttributeError:
            self.aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.aruco_params.minMarkerPerimeterRate = 0.01
            self.aruco_params.maxMarkerPerimeterRate = 4.0
            try:
                self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                self.aruco_params.cornerRefinementWinSize = 5
                self.aruco_params.cornerRefinementMaxIterations = 30
                self.aruco_params.cornerRefinementMinAccuracy = 0.01
            except Exception:
                pass
            self.detector = None
            self.use_new_api = False

    def info_cb(self, msg: CameraInfo):
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        if K[0, 0] <= 1.0:
            return
        D = (np.array(msg.d, dtype=np.float64).reshape(1, -1)
             if len(msg.d) else FALLBACK_D.copy())
        self.K = K
        self.D = D
        self.have_info = True
        self.get_logger().info(
            f'camera_info OK | fx={K[0,0]:.2f} cx={K[0,2]:.2f} '
            f'cy={K[1,2]:.2f} | {msg.width}x{msg.height}')
        try:
            self.destroy_subscription(self.sub_info)
        except Exception:
            pass

    def image_cb(self, msg: Image):
        if not self.have_info:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return

        if not self.warned_res:
            h_img, w_img = frame.shape[:2]
            if abs(w_img - 2 * self.K[0, 2]) > 4 or abs(h_img - 2 * self.K[1, 2]) > 4:
                self.get_logger().warn(
                    f'OJO: imagen {w_img}x{h_img} no concuerda con la K.')
            self.warned_res = True

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.equalize:
            gray = cv2.equalizeHist(gray)

        if self.use_new_api:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params)

        detections = []
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if int(marker_id) not in VALID_IDS:
                    continue
                det = self._solve_one(corners[i], int(marker_id))
                if det is not None:
                    detections.append(det)

        out = String()
        out.data = json.dumps([{k: v for k, v in d.items() if not k.startswith('_')}
                               for d in detections])
        self.pub.publish(out)

        if self.show_view:
            for d in detections:
                self._draw(frame, d)
            cv2.imshow('ArUco Sim', frame)
            cv2.waitKey(1)

    def _solve_one(self, marker_corners, marker_id):
        img_pts = marker_corners.reshape((4, 2)).astype(np.float32)
        size_px = self._size_px(img_pts)

        if size_px < self.min_marker_px:
            return None

        try:
            n, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
                self.obj_pts, img_pts, self.K, self.D,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except Exception:
            return None
        if n == 0:
            return None

        errs  = np.array(reproj).ravel()
        order = np.argsort(errs)
        rvec  = rvecs[order[0]]
        tvec  = tvecs[order[0]]
        reproj_err = float(errs[order[0]])

        tx = float(tvec[0]); ty = float(tvec[1]); tz = float(tvec[2])

        range_xz = math.sqrt(tx * tx + tz * tz)             # horizontal -> EKF
        range_3d = math.sqrt(tx * tx + ty * ty + tz * tz)   # euclidiano (ref MCR2)

        if self.max_range_m > 0 and range_xz > self.max_range_m:
            return None

        yaw_deg = float(np.degrees(np.arctan2(tx, tz)))
        gamma_deg = self._gamma_of(rvec)
        gamma_alt_deg = self._gamma_of(rvecs[order[1]]) if n > 1 else -gamma_deg

        # incertidumbre del rango: Z = fx*marker/s -> dZ = (Z^2/(fx*marker))*ds
        fx = self.K[0, 0]
        range_sigma = (range_xz ** 2) / (fx * self.marker_size) * self.pixel_noise_px

        if self.gt_distance > 0:
            ratio = range_xz / self.gt_distance
            implied = self.marker_size / ratio if ratio > 1e-6 else float('nan')
            expected_px = fx * self.marker_size / tz if tz > 1e-6 else 0.0
            self.get_logger().info(
                f'ID{marker_id} | medido={range_xz:.3f}m gt={self.gt_distance:.3f}m '
                f'ratio={ratio:.3f} -> marker_size real ~= {implied*100:.2f}cm | '
                f'size_px={size_px:.1f} (esperado {expected_px:.1f}) | reproj={reproj_err:.3f}')

        return {
            'id':            marker_id,
            'distance_m':    round(range_xz, 4),   # HORIZONTAL: lo que usa poseKalmanSim
            'range_3d_m':    round(range_3d, 4),   # euclidiano (= ArucoPose.py de MCR2)
            'range_sigma_m': round(range_sigma, 4),# sigma para R del EKF (crece con dist)
            'yaw_deg':       round(yaw_deg, 2),
            'gamma_deg':     round(gamma_deg, 2),
            'gamma_alt_deg': round(gamma_alt_deg, 2),
            'aruco_x_cam':   round(tx, 3),
            'aruco_y_cam':   round(ty, 3),
            'aruco_z_cam':   round(tz, 3),         # distancia axial (debug)
            'size_px':       round(size_px, 2),
            'reproj_err':    round(reproj_err, 4),
            '_rvec': rvec, '_tvec': tvec, '_pts': img_pts,
        }

    def _gamma_of(self, rvec):
        R, _ = cv2.Rodrigues(rvec)
        wrapped = (np.arctan2(R[0, 2], R[2, 2]) + math.pi) % (2 * math.pi) - math.pi
        return float(np.degrees(wrapped))

    def _size_px(self, img_pts):
        right = np.linalg.norm(img_pts[2] - img_pts[1])
        left  = np.linalg.norm(img_pts[0] - img_pts[3])
        return float((right + left) / 2.0)

    def _draw(self, frame, d):
        pts = d['_pts'].astype(int)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        cx = int(np.mean(pts[:, 0])); cy = int(np.mean(pts[:, 1]))
        cv2.drawFrameAxes(frame, self.K, self.D,
                          d['_rvec'], d['_tvec'], self.marker_size * 0.5, 2)
        cv2.putText(frame,
            f"ID{d['id']} {d['distance_m']:.2f}m +-{d['range_sigma_m']:.2f} yaw={d['yaw_deg']:.0f}",
            (max(cx - 90, 5), max(cy - 10, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorSim()
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