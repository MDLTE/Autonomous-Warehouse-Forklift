#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
from pathlib import Path
from collections import deque, Counter

import cv2
import numpy as np
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String


# intrinsecos reales (mismos que aruco_detector, cam a 1280x720)
FX, FY = 1305.61, 1305.89
CX, CY = 640.84, 357.29
DIST_COEFFS = np.array([[0.13807, -0.36743, 0.01148, -0.00193, 0.0]], dtype=np.float64)
CAMERA_MATRIX = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]], dtype=np.float64)

# altura real impresa del logo (las 3 marcas miden ~10x10.5cm). La distancia sale
# de la ALTURA del box (d = fy*H_real/h_px): robusta al angulo horizontal, la altura
# no se acorta cuando el carro mira de lado. El bearing del centro, des-distorsionado.
LOGO_REAL_H = 0.105

# Renombrado de etiquetas: el modelo .pt define los nombres de clase, no este script.
# Aqui traducimos lo que diga el modelo al nombre que queremos mostrar/publicar.
# Cubrimos varias grafias por si en el modelo viene como "Walmart", "Wallmart", etc.
LABEL_REMAP = {
    "Wallmart": "Wolmar",
    "wallmart": "Wolmar",
    "Walmart": "Wolmar",
    "walmart": "Wolmar",
    "WALMART": "Wolmar",
    "WALLMART": "Wolmar",
}


def remap_label(name):
    return LABEL_REMAP.get(name, name)


class LogoDetectionNode(Node):
    def __init__(self):
        super().__init__("logo_detection_node")

        self.declare_parameter("image_topic", "/video_source/compressed")
        self.declare_parameter("model_path", "bestLogos3.pt")
        self.declare_parameter("confidence", 0.55)
        self.declare_parameter("flip_mode", 0)
        self.declare_parameter("show_window", True)
        self.declare_parameter("default_logo_height_m", LOGO_REAL_H)

        self.image_topic = self.get_parameter("image_topic").value
        self.model_path = self.get_parameter("model_path").value
        self.conf_threshold = float(self.get_parameter("confidence").value)
        self.flip_mode = int(self.get_parameter("flip_mode").value)
        self.show_window = bool(self.get_parameter("show_window").value)
        self.default_logo_height_m = float(self.get_parameter("default_logo_height_m").value)

        self.window_size = 10
        self.min_count_to_confirm = 7
        self.min_avg_conf = 0.60

        self.logo_history = deque(maxlen=self.window_size)
        self.conf_history = deque(maxlen=self.window_size)

        base_dir = Path(__file__).resolve().parent
        model_file = Path(self.model_path)

        if not model_file.is_absolute():
            model_file = base_dir / model_file

        if not model_file.exists():
            raise FileNotFoundError(f"No se encontró el modelo: {model_file}")

        self.get_logger().info(f"Cargando modelo YOLO: {model_file}")
        self.model = YOLO(str(model_file))
        self.get_logger().info("Modelo cargado correctamente.")
        self.get_logger().info(f"Clases del modelo: {self.model.names}")

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.sub_img = self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.image_callback,
            qos
        )

        self.pub_detections = self.create_publisher(
            String,
            "/logo_detections",
            10
        )

        self.get_logger().info(f"Suscrito a: {self.image_topic}")
        self.get_logger().info("Publicando detecciones en: /logo_detections")
        self.get_logger().info("Nodo de solo vision (sin teleop). Q cierra la ventana.")

    def compressed_to_cv2(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("No se pudo decodificar la imagen comprimida.")

        return frame

    def fix_camera_orientation(self, frame):
        if self.flip_mode == 0:
            return frame
        elif self.flip_mode == 1:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif self.flip_mode == 2:
            return cv2.flip(frame, 1)
        elif self.flip_mode == 3:
            return cv2.flip(frame, 0)
        elif self.flip_mode == 4:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.flip_mode == 5:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return frame

    def image_callback(self, msg):
        try:
            frame = self.compressed_to_cv2(msg)
            frame = self.fix_camera_orientation(frame)
        except Exception as e:
            self.get_logger().warn(f"Error procesando imagen: {e}")
            return

        height, width = frame.shape[:2]
        image_center_x = width / 2

        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            verbose=False
        )

        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = remap_label(self.model.names[cls])

                box_center_x = (x1 + x2) / 2
                box_center_y = (y1 + y2) / 2

                offset_from_center = box_center_x - image_center_x
                abs_offset_from_center = abs(offset_from_center)

                # distancia por ALTURA (robusta al angulo horizontal)
                h_px = max(float(y2 - y1), 1.0)
                distance_m = FY * self.default_logo_height_m / h_px
                # bearing: des-distorsiona el centro -> rayo normalizado. +deg = a la derecha
                und = cv2.undistortPoints(
                    np.array([[[box_center_x, box_center_y]]], np.float64),
                    CAMERA_MATRIX, DIST_COEFFS)
                yaw_deg = math.degrees(math.atan(float(und[0, 0, 0])))

                detection = {
                    "class_id": cls,
                    "class_name": class_name,
                    "confidence": round(conf, 3),
                    "bbox": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
                    },
                    "center": {
                        "x": round(box_center_x, 2),
                        "y": round(box_center_y, 2)
                    },
                    "offset_from_center_x": round(offset_from_center, 2),
                    "abs_offset_from_center_x": round(abs_offset_from_center, 2),
                    "distance_m": round(distance_m, 4),
                    "yaw_deg": round(yaw_deg, 2)
                }

                detections.append(detection)

                label = f"{class_name} {conf:.2f} {distance_m:.2f}m"

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

        center_selected_logo = None
        center_selected_confidence = 0.0

        if len(detections) > 0:
            closest_detection = min(
                detections,
                key=lambda d: d["abs_offset_from_center_x"]
            )

            center_selected_logo = closest_detection["class_name"]
            center_selected_confidence = closest_detection["confidence"]

            self.logo_history.append(center_selected_logo)
            self.conf_history.append(center_selected_confidence)
        else:
            self.logo_history.append("none")
            self.conf_history.append(0.0)

        stable_logo, stable_conf, stable_count = self.get_stable_decision()

        output_data = {
            "detected_count": len(detections),
            "detections": detections,

            "center_selected_logo": center_selected_logo,
            "center_selected_confidence": center_selected_confidence,

            "stable_logo": stable_logo,
            "stable_confidence_avg": stable_conf,
            "stable_count": stable_count,
            "window_size": self.window_size,
            "decision_confirmed": stable_logo is not None
        }

        msg_out = String()
        msg_out.data = json.dumps(output_data)
        self.pub_detections.publish(msg_out)

        cv2.line(
            frame,
            (int(image_center_x), 0),
            (int(image_center_x), height),
            (255, 0, 0),
            2
        )

        if stable_logo is not None:
            decision_text = (
                f"STABLE: {stable_logo} "
                f"{stable_count}/{self.window_size} "
                f"avg:{stable_conf:.2f}"
            )
            decision_color = (0, 255, 0)
        else:
            decision_text = "STABLE: waiting..."
            decision_color = (0, 0, 255)

        cv2.putText(
            frame,
            decision_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            decision_color,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Center selected: {center_selected_logo}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        if self.show_window:
            cv2.imshow("Logo Detection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                self.get_logger().info("Cerrando con q...")
                rclpy.shutdown()

    def get_stable_decision(self):
        valid_logos = [
            logo for logo in self.logo_history
            if logo != "none"
        ]

        if len(valid_logos) == 0:
            return None, 0.0, 0

        counts = Counter(valid_logos)
        most_common_logo, count = counts.most_common(1)[0]

        confs_for_logo = [
            conf
            for logo, conf in zip(self.logo_history, self.conf_history)
            if logo == most_common_logo
        ]

        avg_conf = float(np.mean(confs_for_logo)) if len(confs_for_logo) > 0 else 0.0

        if count >= self.min_count_to_confirm and avg_conf >= self.min_avg_conf:
            return most_common_logo, round(avg_conf, 3), count

        return None, round(avg_conf, 3), count

    def shutdown_node(self):
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)

    node = LogoDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_node()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()