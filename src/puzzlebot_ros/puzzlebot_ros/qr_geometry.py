#!/usr/bin/env python3
"""
QR Geometry Diagnostic Node.

Detecta el QR "Emezon" y calcula toda la geometría necesaria
para planear la maniobra de alineación e inserción.

Muestra en pantalla:
  - Ángulo del QR respecto al robot (perpendicularity)
  - Distancia cámara → QR (cm)
  - Offset lateral real (cm) del centro del QR respecto al centro de la cámara
  - Posición del centro del pallet estimada
  - La trayectoria que el robot necesitaría seguir

GEOMETRÍA DEL ROBOT:
  Eje rotación ← 6cm → Cámara ← 7cm → Punta horquillas
  
  Cámara: FOV horizontal ≈ 70° (Logitech C920 a 640px)
  Calibración: 83px = 25cm → K = 2075

TECLAS:
  ESPACIO = capturar snapshot de datos
  ESC = cerrar
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math


class QRGeometryNode(Node):

    def __init__(self):
        super().__init__('qr_geometry_node')

        # ── Calibración ──────────────────────────────────────────────
        # QR size-distance: K = size_px * distance_cm
        # Medido: 83px a 25cm, 66px a 30cm → K promedio ≈ 2055
        self.K = 2055.0

        # Geometría del robot (cm)
        self.CAM_FROM_AXIS = 6.0      # Cámara adelante del eje de rotación
        self.FORK_FROM_AXIS = 13.0    # Horquillas adelante del eje de rotación
        self.FORK_FROM_CAM = 7.0      # Horquillas adelante de la cámara

        # FOV horizontal de la cámara (Logitech C920 a 640px)
        self.HFOV_DEG = 70.3
        self.IMG_W = 640  # Se actualiza con el frame real

        # ── ROS ──────────────────────────────────────────────────────
        img_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.image_sub = self.create_subscription(
            Image, '/video_source/raw', self.image_callback, img_qos
        )

        # ── OpenCV ───────────────────────────────────────────────────
        self.bridge = CvBridge()
        self.qr_detector = cv2.QRCodeDetector()
        self.last_valid_data = None

        self.get_logger().info('QR Geometry Node — esperando QR...')
        self.get_logger().info('ESPACIO=snapshot  ESC=cerrar')

    def compute_angle_and_width(self, pts):
        """Calcula ángulo del QR y su ancho en px."""
        edges = []
        for i in range(4):
            dx = pts[(i+1) % 4][0] - pts[i][0]
            dy = pts[(i+1) % 4][1] - pts[i][1]
            length = np.sqrt(dx*dx + dy*dy)
            angle = np.degrees(np.arctan2(dy, dx))
            edges.append((length, angle))

        def h_close(a):
            x = a % 180
            return min(x, 180 - x)

        horiz = sorted(edges, key=lambda e: h_close(e[1]))[:2]
        width = np.mean([e[0] for e in horiz])
        ang = horiz[0][1]
        if ang > 90: ang -= 180
        elif ang < -90: ang += 180
        return ang, width

    def px_to_lateral_cm(self, px_offset, distance_cm):
        """Convierte offset en pixeles a centímetros usando FOV y distancia."""
        # Ángulo lateral = arctan(px_offset / focal_length_px)
        # focal_length_px = (IMG_W / 2) / tan(HFOV/2)
        focal_px = (self.IMG_W / 2.0) / math.tan(math.radians(self.HFOV_DEG / 2.0))
        angle_rad = math.atan2(px_offset, focal_px)
        lateral_cm = distance_cm * math.tan(angle_rad)
        return lateral_cm, math.degrees(angle_rad)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(str(e))
            return

        img_h, img_w = frame.shape[:2]
        frame = cv2.flip(frame, -1)
        self.IMG_W = img_w
        img_cx = img_w / 2.0

        # ── Detectar QR ──────────────────────────────────────────
        data, points, _ = self.qr_detector.detectAndDecode(frame)
        if points is None or len(points) == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            data, points, _ = self.qr_detector.detectAndDecode(gray)

        # ── Línea central ────────────────────────────────────────
        cv2.line(frame, (int(img_cx), 0), (int(img_cx), img_h), (100, 100, 100), 1)

        if points is None or len(points) == 0:
            cv2.putText(frame, 'QR NO DETECTADO', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow('QR Geometry', frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                rclpy.shutdown()
            return

        pts = points[0]

        # Filtros
        edge_lens = [np.linalg.norm(pts[(i+1) % 4] - pts[i]) for i in range(4)]
        avg_sz = np.mean(edge_lens)
        aspect = max(edge_lens) / max(min(edge_lens), 1.0)
        if avg_sz < 25 or aspect > 1.5:
            cv2.imshow('QR Geometry', frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                rclpy.shutdown()
            return

        # Validar data
        if data == 'Emezon':
            self.last_valid_data = data
        elif data == '' and self.last_valid_data == 'Emezon':
            data = self.last_valid_data
        else:
            cv2.imshow('QR Geometry', frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                rclpy.shutdown()
            return

        # ══════════════════════════════════════════════════════════
        # GEOMETRÍA
        # ══════════════════════════════════════════════════════════

        # 1. Ángulo del QR (rotación del pallet respecto a la cámara)
        qr_angle, qr_width_px = self.compute_angle_and_width(pts)

        # 2. Distancia cámara → QR (cm)
        cam_dist_cm = self.K / qr_width_px

        # 3. Centro del QR en pixeles
        qr_cx = np.mean(pts[:, 0])
        qr_cy = np.mean(pts[:, 1])

        # 4. Offset lateral en pixeles y centímetros
        px_offset = qr_cx - img_cx  # (+) = QR a la derecha
        lateral_cm, lateral_angle_deg = self.px_to_lateral_cm(px_offset, cam_dist_cm)

        # 5. Distancias desde el eje de rotación del robot
        axis_to_qr_cm = cam_dist_cm + self.CAM_FROM_AXIS
        fork_to_qr_cm = cam_dist_cm - self.FORK_FROM_CAM

        # 6. Maniobra necesaria para alineación
        # El robot necesita quedar:
        #   - A lateral_cm = 0 respecto al QR
        #   - Con ángulo = 0° respecto al QR
        #   - A distancia de inserción (fork_to_qr = insert_depth)

        # Descomposición de la maniobra:
        # a) Girar para quedar perpendicular: necesita corregir qr_angle grados
        # b) Desplazarse lateralmente: necesita moverse lateral_cm
        #    Para esto: girar 90°, avanzar lateral_cm, girar -90°
        # c) Avanzar recto hasta insertar

        # Si el robot no está perpendicular, el offset lateral real incluye
        # la componente del ángulo:
        # lateral_real = lateral_cm + cam_dist_cm * sin(qr_angle)
        angle_correction_cm = cam_dist_cm * math.sin(math.radians(qr_angle))
        total_lateral_cm = lateral_cm + angle_correction_cm

        # ══════════════════════════════════════════════════════════
        # DIBUJO
        # ══════════════════════════════════════════════════════════

        # Contorno QR
        perp_ok = abs(qr_angle) < 3.0
        centered_ok = abs(total_lateral_cm) < 2.0
        pi = pts.astype(int)
        color_qr = (0, 255, 0) if (perp_ok and centered_ok) else (0, 255, 255)
        for i in range(4):
            p1 = (int(pi[i][0]), int(pi[i][1]))
            p2 = (int(pi[(i+1) % 4][0]), int(pi[(i+1) % 4][1]))
            cv2.line(frame, p1, p2, color_qr, 2)

        # Centro QR
        cv2.circle(frame, (int(qr_cx), int(qr_cy)), 6, (0, 0, 255), -1)
        cv2.line(frame, (int(qr_cx), int(qr_cy)),
                 (int(img_cx), int(qr_cy)), (0, 0, 255), 1)

        # ── Info en pantalla ─────────────────────────────────────
        y = 22
        dy = 20
        lines = [
            ('=== GEOMETRIA QR ===', (0, 200, 255)),
            (f'Angulo QR: {qr_angle:+.1f} deg {"PERP OK" if perp_ok else ""}',
             (0, 255, 0) if perp_ok else (0, 0, 255)),
            (f'QR width: {qr_width_px:.1f}px', (255, 255, 255)),
            ('', (0,0,0)),
            ('=== DISTANCIAS ===', (0, 200, 255)),
            (f'Camara -> QR: {cam_dist_cm:.1f}cm', (255, 255, 255)),
            (f'Eje rot -> QR: {axis_to_qr_cm:.1f}cm', (255, 255, 255)),
            (f'Horquillas -> QR: {fork_to_qr_cm:.1f}cm', (255, 255, 255)),
            ('', (0,0,0)),
            ('=== LATERAL ===', (0, 200, 255)),
            (f'Offset px: {px_offset:+.0f}px', (255, 255, 255)),
            (f'Angulo lateral: {lateral_angle_deg:+.1f} deg', (255, 255, 255)),
            (f'Offset real: {lateral_cm:+.1f}cm', (255, 255, 255)),
            (f'Correccion angulo: {angle_correction_cm:+.1f}cm', (255, 200, 100)),
            (f'Offset TOTAL: {total_lateral_cm:+.1f}cm {"OK" if centered_ok else ""}',
             (0, 255, 0) if centered_ok else (0, 0, 255)),
            ('', (0,0,0)),
            ('=== MANIOBRA ===', (0, 200, 255)),
            (f'1. Girar {-qr_angle:+.1f} deg (perpendicular)', (255, 255, 255)),
            (f'2. Desplazar {-total_lateral_cm:+.1f}cm lateral', (255, 255, 255)),
            (f'3. Avanzar {fork_to_qr_cm:.1f}cm para insertar', (255, 255, 255)),
        ]

        for txt, color in lines:
            if txt:
                cv2.putText(frame, txt, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
            y += dy

        # ── Dibujar vista top-down en esquina ────────────────────
        self.draw_topdown(frame, img_w, img_h, qr_angle, cam_dist_cm,
                          lateral_cm, total_lateral_cm)

        # ── Log ──────────────────────────────────────────────────
        self.get_logger().info(
            f'ang={qr_angle:+.1f}° dist={cam_dist_cm:.1f}cm '
            f'lat={total_lateral_cm:+.1f}cm forks={fork_to_qr_cm:.1f}cm',
            throttle_duration_sec=0.5
        )

        # ── Teclado ──────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            rclpy.shutdown()
        elif key == ord(' '):
            self.get_logger().info(
                f'\n===== SNAPSHOT =====\n'
                f'  QR angle:     {qr_angle:+.1f}°\n'
                f'  QR width:     {qr_width_px:.1f}px\n'
                f'  Cam dist:     {cam_dist_cm:.1f}cm\n'
                f'  Axis dist:    {axis_to_qr_cm:.1f}cm\n'
                f'  Fork dist:    {fork_to_qr_cm:.1f}cm\n'
                f'  Lateral px:   {px_offset:+.0f}px\n'
                f'  Lateral cm:   {lateral_cm:+.1f}cm\n'
                f'  Angle corr:   {angle_correction_cm:+.1f}cm\n'
                f'  Total lateral: {total_lateral_cm:+.1f}cm\n'
                f'  MANIOBRA:\n'
                f'    1. Girar {-qr_angle:+.1f}°\n'
                f'    2. Lateral {-total_lateral_cm:+.1f}cm\n'
                f'    3. Avanzar {fork_to_qr_cm:.1f}cm\n'
                f'====================\n'
            )

        cv2.imshow('QR Geometry', frame)

    def draw_topdown(self, frame, img_w, img_h, qr_angle, cam_dist_cm,
                     lateral_cm, total_lateral_cm):
        """Dibuja una vista top-down de la situación en la esquina del frame."""
        # Área del mini-mapa
        mw, mh = 200, 200
        mx, my = img_w - mw - 10, 10  # Esquina superior derecha
        scale = 3.0  # px por cm

        # Fondo semitransparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (mx, my), (mx + mw, my + mh), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Centro del mini-mapa = posición del robot (eje de rotación)
        rcx = mx + mw // 2
        rcy = my + mh - 30  # Robot abajo

        # Dibujar robot (triángulo)
        robot_pts = np.array([
            [rcx, rcy - 8],
            [rcx - 6, rcy + 6],
            [rcx + 6, rcy + 6]
        ], np.int32)
        cv2.fillPoly(frame, [robot_pts], (0, 200, 0))

        # Horquillas
        fork_y = rcy - int(self.FORK_FROM_AXIS * scale)
        cv2.line(frame, (rcx - 4, rcy - 8), (rcx - 4, fork_y), (0, 255, 0), 2)
        cv2.line(frame, (rcx + 4, rcy - 8), (rcx + 4, fork_y), (0, 255, 0), 2)

        # Cámara
        cam_y = rcy - int(self.CAM_FROM_AXIS * scale)
        cv2.circle(frame, (rcx, cam_y), 3, (0, 0, 255), -1)

        # QR / Pallet (posición relativa)
        qr_y = rcy - int((cam_dist_cm + self.CAM_FROM_AXIS) * scale)
        qr_x = rcx + int(lateral_cm * scale)

        # Dibujar pallet como rectángulo rotado
        pallet_w = 25  # px visual
        pallet_h = 8
        angle_rad = math.radians(qr_angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        corners = []
        for dx, dy in [(-pallet_w//2, -pallet_h//2), (pallet_w//2, -pallet_h//2),
                        (pallet_w//2, pallet_h//2), (-pallet_w//2, pallet_h//2)]:
            rx = int(qr_x + dx * cos_a - dy * sin_a)
            ry = int(qr_y + dx * sin_a + dy * cos_a)
            corners.append([rx, ry])
        cv2.polylines(frame, [np.array(corners)], True, (0, 255, 255), 2)

        # QR center
        cv2.circle(frame, (qr_x, qr_y), 4, (0, 0, 255), -1)

        # Línea de inserción ideal (desde robot recto al pallet)
        ideal_x = rcx  # Directo al frente
        cv2.line(frame, (rcx, rcy), (ideal_x, qr_y), (0, 100, 255), 1,
                 cv2.LINE_AA)

        # Línea real al QR
        cv2.line(frame, (rcx, cam_y), (qr_x, qr_y), (0, 0, 255), 1,
                 cv2.LINE_AA)

        # Etiqueta
        cv2.putText(frame, 'TOP VIEW', (mx + 5, my + 15),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def main(args=None):
    rclpy.init(args=args)
    node = QRGeometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()