#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
door_align.py — alineacion final del Puzzlebot montacargas con la puerta de entrega.

No carga YOLO ni toca la camara. Consume /logo_detections (lo que ya publica
logo_detection.py) y hace servo visual hacia el CENTRO DE LA PUERTA, que esta a
la derecha del logo por un offset lateral fijo (el logo cae a la IZQUIERDA de la
puerta desde el punto de vista del robot). Logos usados: Emezon, Popsi, Wallmart.

Estrategia:
  ALIGN  -> visual: corrige el angulo de llegada desconocido centrando la PUERTA.
  APPROACH -> odometrico: avanza una distancia fija desde el waypoint (40cm) usando
              /odom. El logo a 24.5cm se sale del cuadro al acercarse, asi que la
              distancia NO la gobierna la vision. Mientras el logo siga visible se
              corrige el yaw; cuando se pierde, sigue recto.
              Si no hay /odom, fallback: mide el avance por la baja de distancia visual.

Flujo:
  nav deja al robot en el waypoint (medio alineado, angulo desconocido)
  -> se publica el target (clase del logo del QR) en /align/start
  -> SEARCH gira hasta ver el logo estable
  -> ALIGN centra el yaw de la PUERTA, luego baja horquilla a rack_abajo_entrada
     y espera fork_status=DONE antes de avanzar
  -> APPROACH avanza 40cm (odom) corrigiendo yaw mientras vea el logo
  -> DROP frena, baja horquilla a base, espera fork_status=DONE,
     termina y avisa en /align/status via /task_status

IMPORTANTE: corre logo_detection.py con publish_cmd:=false para que este nodo sea
el unico que manda /cmd_vel.

Sub: /logo_detections (String/JSON), /odom (Odometry), /align/start (String),
     /align/cancel (Bool), /fork_status (String), /fsm_cmd (String)
Pub: /align/cmd_vel (Twist), /align/status (String/JSON), /fork_cmd_name (String),
     /task_status (String)
"""

import json
import math
import time

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class DoorAlignNode(Node):
    def __init__(self):
        super().__init__("door_align_node")
        self.enabled = False
        self.done_published = False

        self.declare_parameter("logo_detections_topic", "/logo_detections")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_vel_topic", "/align/cmd_vel")
        self.declare_parameter("start_topic", "/align/start")
        self.declare_parameter("status_topic", "/align/status")
        self.declare_parameter("forklift_topic", "/fork_cmd_name")
        self.declare_parameter("door_offset_m", 0.245)
        self.declare_parameter("approach_travel_m", 0.40)
        self.declare_parameter("known_classes", ["Emezon", "Popsi", "Wallmart"])
        self.declare_parameter("yaw_tol_deg", 3.5)
        self.declare_parameter("align_hold_cycles", 4)
        self.declare_parameter("search_angular_speed", 0.45)
        self.declare_parameter("search_direction", 1)
        self.declare_parameter("search_sweep_time", 4.0)
        self.declare_parameter("kp_yaw_align", 1.6)
        self.declare_parameter("max_angular_align", 0.7)
        self.declare_parameter("kp_yaw_approach", 1.0)
        self.declare_parameter("max_angular_approach", 0.4)
        self.declare_parameter("approach_speed", 0.12)
        self.declare_parameter("approach_slowdown_m", 0.10)
        self.declare_parameter("ema_alpha", 0.4)
        self.declare_parameter("det_stale_s", 0.4)
        self.declare_parameter("target_lost_s", 1.0)
        self.declare_parameter("min_confidence", 0.55)
        self.declare_parameter("min_safe_distance_m", 0.12)
        self.declare_parameter("max_approach_time_s", 12.0)
        self.declare_parameter("drop_cmd", "down 2500")
        self.declare_parameter("raise_after_drop", False)
        self.declare_parameter("raise_cmd", "up 2500")
        self.declare_parameter("drop_wait_s", 3.0)
        self.declare_parameter("aruco_detections_topic", "/aruco/detections")
        self.declare_parameter("reverse_after_drop", True)
        self.declare_parameter("reverse_target_aruco_id", -1)
        self.declare_parameter("reverse_speed", 0.10)
        self.declare_parameter("aruco_confirm_frames", 3)
        self.declare_parameter("aruco_stale_s", 0.4)
        self.declare_parameter("max_reverse_dist_m", 0.8)
        self.declare_parameter("max_reverse_time_s", 8.0)
        self.declare_parameter("qr_to_class_json", "{}")

        g = self.get_parameter
        self.det_topic = g("logo_detections_topic").value
        self.odom_topic = g("odom_topic").value
        self.cmd_topic = g("cmd_vel_topic").value
        self.door_offset_m = float(g("door_offset_m").value)
        self.approach_travel_m = float(g("approach_travel_m").value)
        self.known_classes = list(g("known_classes").value)
        self.yaw_tol = math.radians(float(g("yaw_tol_deg").value))
        self.align_hold_cycles = int(g("align_hold_cycles").value)
        self.search_w = float(g("search_angular_speed").value)
        self.search_dir = int(g("search_direction").value)
        self.search_sweep_time = float(g("search_sweep_time").value)
        self.kp_yaw_align = float(g("kp_yaw_align").value)
        self.max_w_align = float(g("max_angular_align").value)
        self.kp_yaw_app = float(g("kp_yaw_approach").value)
        self.max_w_app = float(g("max_angular_approach").value)
        self.approach_speed = float(g("approach_speed").value)
        self.approach_slowdown_m = float(g("approach_slowdown_m").value)
        self.alpha = float(g("ema_alpha").value)
        self.det_stale_s = float(g("det_stale_s").value)
        self.target_lost_s = float(g("target_lost_s").value)
        self.min_conf = float(g("min_confidence").value)
        self.min_safe_dist = float(g("min_safe_distance_m").value)
        self.max_approach_time = float(g("max_approach_time_s").value)
        self.drop_cmd = g("drop_cmd").value
        self.raise_after_drop = bool(g("raise_after_drop").value)
        self.raise_cmd = g("raise_cmd").value
        self.drop_wait_s = float(g("drop_wait_s").value)
        self.aruco_topic = g("aruco_detections_topic").value
        self.reverse_after_drop = bool(g("reverse_after_drop").value)
        self.reverse_target_id = int(g("reverse_target_aruco_id").value)
        self.reverse_speed = float(g("reverse_speed").value)
        self.aruco_confirm_frames = int(g("aruco_confirm_frames").value)
        self.aruco_stale_s = float(g("aruco_stale_s").value)
        self.max_reverse_dist = float(g("max_reverse_dist_m").value)
        self.max_reverse_time = float(g("max_reverse_time_s").value)

        try:
            self.qr_to_class = json.loads(g("qr_to_class_json").value)
        except json.JSONDecodeError:
            self.qr_to_class = {}

        self.state = "IDLE"
        self.target_class = None
        self.last_det_time = 0.0
        self.last_seen_time = 0.0
        self.latest_detections = []
        self.door_yaw_ema = None
        self.distance_ema = None
        self.aligned_streak = 0
        self.search_flip_time = 0.0
        self.search_sign = self.search_dir
        self.drop_done_time = None
        self.odom_xy = None
        self.have_odom = False
        self.approach_start_xy = None
        self.approach_ref_dist = None
        self.approach_start_time = None
        self.aruco_ids = set()
        self.last_aruco_time = 0.0
        self.aruco_streak = 0
        self.reverse_start_xy = None
        self.reverse_start_time = None

        self.sub_det = self.create_subscription(String, self.det_topic, self.on_detections, 10)
        self.sub_aruco = self.create_subscription(String, self.aruco_topic, self.on_aruco, 10)
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.on_odom, 10)
        self.sub_start = self.create_subscription(String, g("start_topic").value, self.on_start, 10)
        self.sub_cancel = self.create_subscription(Bool, "/align/cancel", self.on_cancel, 10)
        self.sub_fsm = self.create_subscription(String, '/fsm_cmd', self.cb_fsm_cmd, 10)

        self.pub_cmd = self.create_publisher(Twist, self.cmd_topic, 10)
        self.pub_status = self.create_publisher(String, g("status_topic").value, 10)
        self.pub_fork = self.create_publisher(String, g("forklift_topic").value, 10)
        self._task_status_pub = self.create_publisher(String, '/task_status', 10)
        self.logo_detected_pub = self.create_publisher(Bool, '/logo_detected', 10)
        self.done_pub = self.create_publisher(Bool, '/door_align/done', 10)

        # fork_status subscriber para esperar DONE antes de continuar
        self._fork_status_done = False
        self._waiting_fork     = False
        self.create_subscription(String, '/fork_status', self._cb_fork_status, 10)

        # fork_status subscriber para esperar DONE antes de continuar
        self._fork_status_done = False
        self._waiting_fork     = False
        self.create_subscription(String, '/fork_status', self._cb_fork_status, 10)

        self.timer = self.create_timer(0.05, self.control_loop)  # 20 Hz
        self.get_logger().info(f"Clases conocidas: {self.known_classes}")
        self.get_logger().info(f"offset_puerta={self.door_offset_m:.3f}m  avance={self.approach_travel_m:.3f}m")

    def on_detections(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.latest_detections = data.get("detections", [])
        self.last_det_time = time.time()

    def on_aruco(self, msg):
        try:
            arr = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        ids = set()
        for d in arr:
            try:
                ids.add(int(d.get("id")))
            except (TypeError, ValueError):
                pass
        self.aruco_ids = ids
        self.last_aruco_time = time.time()

    def aruco_target_visible(self):
        if (time.time() - self.last_aruco_time) > self.aruco_stale_s:
            return False
        if not self.aruco_ids:
            return False
        if self.reverse_target_id < 0:
            return True
        return self.reverse_target_id in self.aruco_ids

    def on_odom(self, msg):
        self.odom_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.have_odom = True

    def on_start(self, msg):
        raw = msg.data.strip()
        if not raw:
            return
        target = self.qr_to_class.get(raw, raw)
        if self.known_classes and target not in self.known_classes:
            self.get_logger().warn(f"target '{target}' no esta en known_classes {self.known_classes}")
        self.target_class = target
        self.reset_for_new_goal()
        self.state = "SEARCH"
        self.search_flip_time = time.time()
        self.search_sign = self.search_dir
        self.get_logger().info(f"Nuevo objetivo: '{target}' -> SEARCH")

    def on_cancel(self, msg):
        if msg.data:
            self.abort("cancelado por /align/cancel")

    def _cb_fork_status(self, msg: String):
        if msg.data.startswith('DONE'):
            self._fork_status_done = True

    def cb_fsm_cmd(self, msg):
        cmd = msg.data.strip()
        if cmd.startswith("T7:"):
            empresa = cmd.split(":", 1)[1].strip()
            empresa_map = {
                'Walmar':  'Wallmart', 'walmar': 'Wallmart', 'WALMAR': 'Wallmart',
                'Emezon':  'Emezon',   'Popsi':  'Popsi',
            }
            empresa = empresa_map.get(empresa, empresa)
            self.get_logger().info(f"[FSM_CMD] T7 → empresa={empresa}")
            self.enabled = True
            fake = String(); fake.data = empresa
            self.on_start(fake)
        elif cmd == "STOP":
            self.enabled = False
            self.abort("STOP por FSM")

    def _pub_task_status(self, status: str, data: str = ""):
        msg = String()
        msg.data = f"ALIGN:{status}:{data}"
        self.pub_status.publish(msg)
        task_msg = String()
        task_msg.data = f"ALIGN:{status}:{data}"
        self._task_status_pub.publish(task_msg)

    def reset_for_new_goal(self):
        self.door_yaw_ema = None
        self.distance_ema = None
        self.aligned_streak = 0
        self.drop_done_time = None
        self.approach_start_xy = None
        self.approach_ref_dist = None
        self.approach_start_time = None
        self.done_published = False
        self._fork_status_done = False
        self._waiting_fork = False
        self._fork_status_done = False
        self._waiting_fork = False

    def abort(self, reason):
        self.get_logger().warn(f"ABORT: {reason}")
        self.stop_robot()
        self.state = "IDLE"
        self.target_class = None
        self.reset_for_new_goal()
        self._pub_task_status("FAILED", reason)

    def pick_target_detection(self):
        cands = [
            d for d in self.latest_detections
            if d.get("class_name") == self.target_class
            and float(d.get("confidence", 0.0)) >= self.min_conf
        ]
        if not cands:
            return None
        return min(cands, key=lambda d: d.get("abs_offset_from_center_x", 1e9))

    def door_geometry(self, det):
        dist = float(det["distance_m"])
        yaw_logo = math.radians(float(det["yaw_deg"]))
        x_logo = dist * math.tan(yaw_logo)
        x_door = x_logo + self.door_offset_m
        yaw_door = math.atan2(x_door, dist)
        return yaw_door, dist

    def update_ema(self, yaw_door, dist):
        if self.door_yaw_ema is None:
            self.door_yaw_ema = yaw_door
            self.distance_ema = dist
        else:
            a = self.alpha
            self.door_yaw_ema = a * yaw_door + (1 - a) * self.door_yaw_ema
            self.distance_ema = a * dist + (1 - a) * self.distance_ema

    def traveled(self):
        if self.have_odom and self.approach_start_xy is not None and self.odom_xy is not None:
            dx = self.odom_xy[0] - self.approach_start_xy[0]
            dy = self.odom_xy[1] - self.approach_start_xy[1]
            return math.hypot(dx, dy)
        if self.approach_ref_dist is not None and self.distance_ema is not None:
            return self.approach_ref_dist - self.distance_ema
        return 0.0

    def send(self, vx, wz):
        t = Twist()
        t.linear.x = float(vx)
        t.angular.z = float(wz)
        self.pub_cmd.publish(t)

    def stop_robot(self):
        self.send(0.0, 0.0)

    def fork(self, cmd):
        m = String()
        m.data = cmd
        self.pub_fork.publish(m)

    def publish_status(self, done=False, extra=None):
        out = {
            "state": self.state,
            "target": self.target_class,
            "done": done,
            "door_yaw_deg": round(math.degrees(self.door_yaw_ema), 2) if self.door_yaw_ema is not None else None,
            "distance_m": round(self.distance_ema, 3) if self.distance_ema is not None else None,
            "traveled_m": round(self.traveled(), 3) if self.state == "APPROACH" else None,
            "reverse_m": round(self.reverse_traveled(), 3) if self.state == "REVERSE" else None,
            "aruco_streak": self.aruco_streak if self.state == "REVERSE" else None,
            "odom": self.have_odom,
            "waiting_fork": self._waiting_fork,
        }
        if extra:
            out.update(extra)
        m = String()
        m.data = json.dumps(out)
        self.pub_status.publish(m)

    def control_loop(self):
        if not self.enabled:
            self.stop_robot()
            return
        now = time.time()
        det_fresh = (now - self.last_det_time) < self.det_stale_s

        if self.state in ("IDLE", "DONE"):
            self.stop_robot()
            return

        if not det_fresh and self.state in ("SEARCH", "ALIGN"):
            self.stop_robot()
            self.publish_status()
            return

        det = self.pick_target_detection() if det_fresh else None
        self.logo_detected_pub.publish(Bool(data=(det is not None)))
        if det is not None:
            yaw_door, dist = self.door_geometry(det)
            self.update_ema(yaw_door, dist)
            self.last_seen_time = now

        lost = (now - self.last_seen_time) > self.target_lost_s

        if self.state == "SEARCH":
            self.do_search(det, now)
        elif self.state == "ALIGN":
            self.do_align(det, lost)
        elif self.state == "APPROACH":
            self.do_approach(det, now)
        elif self.state == "DROP":
            self.do_drop(now)
        elif self.state == "REVERSE":
            self.do_reverse(now)

        self.publish_status()

    def do_search(self, det, now):
        if det is not None and self.door_yaw_ema is not None:
            self.aligned_streak = 0
            self.state = "ALIGN"
            self.get_logger().info(f"'{self.target_class}' visto -> ALIGN")
            self.stop_robot()
            return
        if now - self.search_flip_time > self.search_sweep_time:
            self.search_sign *= -1
            self.search_flip_time = now
        self.send(0.0, self.search_sign * self.search_w)

    def do_align(self, det, lost):
        if lost:
            self.state = "SEARCH"
            self.search_flip_time = time.time()
            self.get_logger().warn("target perdido en ALIGN -> SEARCH")
            self.stop_robot()
            return
        err = self.door_yaw_ema
        wz = clamp(-self.kp_yaw_align * err, -self.max_w_align, self.max_w_align)
        self.send(0.0, wz)
        if abs(err) <= self.yaw_tol:
            self.aligned_streak += 1
        else:
            self.aligned_streak = 0
        if self.aligned_streak >= self.align_hold_cycles:
            if not self._waiting_fork:
                # bajar horquilla a posición de entrada antes de avanzar
                self.fork("rack_abajo_entrada")
                self._fork_status_done = False
                self._waiting_fork = True
                self.get_logger().info("Alineado — bajando horquilla: rack_abajo_entrada")
                return
            if not self._fork_status_done:
                # esperando DONE de fork_status
                self.stop_robot()
                return
            # horquilla lista — arrancar avance
            self._waiting_fork = False
            self._fork_status_done = False
            self.approach_start_xy = self.odom_xy if self.have_odom else None
            self.approach_ref_dist = self.distance_ema
            self.approach_start_time = time.time()
            self.state = "APPROACH"
            mode = "odom" if self.have_odom else "visual (sin odom)"
            self.get_logger().info(f"horquilla lista -> APPROACH ({mode}), avance {self.approach_travel_m:.2f}m")

    def do_approach(self, det, now):
        trav = self.traveled()
        remaining = self.approach_travel_m - trav
        too_close = (det is not None and self.distance_ema is not None
                     and self.distance_ema <= self.min_safe_dist)
        timeout = (self.approach_start_time is not None
                   and now - self.approach_start_time > self.max_approach_time)
        if remaining <= 0.0 or too_close or timeout:
            self.stop_robot()
            self.state = "DROP"
            self.drop_done_time = None
            reason = "avance completo" if remaining <= 0.0 else ("guarda de distancia" if too_close else "timeout")
            self.get_logger().info(f"fin de avance ({reason}, recorrido {trav:.3f}m) -> DROP")
            return
        if remaining < self.approach_slowdown_m:
            vx = self.approach_speed * clamp(remaining / self.approach_slowdown_m, 0.25, 1.0)
        else:
            vx = self.approach_speed
        if det is not None and self.door_yaw_ema is not None:
            wz = clamp(-self.kp_yaw_app * self.door_yaw_ema, -self.max_w_app, self.max_w_app)
        else:
            wz = 0.0
        self.send(vx, wz)

    def do_drop(self, now):
        self.stop_robot()
        if self.drop_done_time is None:
            self.drop_done_time = now
            self.get_logger().info("DROP: esperando drop_wait_s antes de bajar a base")
            return
        if now - self.drop_done_time >= self.drop_wait_s:
            # soltar pallet: bajar a base y esperar DONE
            if not self._waiting_fork:
                self.fork("base")
                self._fork_status_done = False
                self._waiting_fork = True
                self.get_logger().info("Pallet depositado — bajando a base")
                return
            if not self._fork_status_done:
                self.stop_robot()
                return
            self._waiting_fork = False
            self._fork_status_done = False
            if self.raise_after_drop:
                self.fork(self.raise_cmd)
                self.get_logger().info(f"levantando horquilla: '{self.raise_cmd}'")
            if self.reverse_after_drop:
                self.reverse_start_xy = self.odom_xy if self.have_odom else None
                self.reverse_start_time = now
                self.aruco_streak = 0
                self.state = "REVERSE"
                tgt = "cualquier aruco" if self.reverse_target_id < 0 else f"aruco {self.reverse_target_id}"
                self.get_logger().info(f"carga dejada -> REVERSE hasta ver {tgt}")
            else:
                self.state = "DONE"
                if not self.done_published:
                    self.done_pub.publish(Bool(data=True))
                    self.done_published = True
                self.publish_status(done=True, extra={"result": "ok"})
                self._pub_task_status("DONE")
                self.get_logger().info("Entrega completada -> DONE")

    def reverse_traveled(self):
        if (self.have_odom and self.reverse_start_xy is not None and self.odom_xy is not None):
            dx = self.odom_xy[0] - self.reverse_start_xy[0]
            dy = self.odom_xy[1] - self.reverse_start_xy[1]
            return math.hypot(dx, dy)
        return 0.0

    def do_reverse(self, now):
        if self.aruco_target_visible():
            self.aruco_streak += 1
        else:
            self.aruco_streak = 0
        if self.aruco_streak >= self.aruco_confirm_frames:
            self.stop_robot()
            self.state = "DONE"
            self.publish_status(done=True, extra={"result": "ok", "aruco_ids": sorted(self.aruco_ids)})
            self._pub_task_status("DONE")
            self.get_logger().info(f"aruco visible (ids {sorted(self.aruco_ids)}) -> DONE")
            return
        dist_cap = self.reverse_traveled() >= self.max_reverse_dist
        time_cap = (self.reverse_start_time is not None
                    and now - self.reverse_start_time > self.max_reverse_time)
        if dist_cap or time_cap:
            self.stop_robot()
            self.state = "DONE"
            reason = "limite de distancia" if dist_cap else "timeout"
            self.publish_status(done=True, extra={"result": "reverse_no_aruco", "reason": reason})
            self._pub_task_status("DONE")
            self.get_logger().warn(f"retroceso terminado sin ver aruco ({reason}) -> DONE")
            return
        self.send(-abs(self.reverse_speed), 0.0)

    def shutdown_node(self):
        self.stop_robot()


def main(args=None):
    rclpy.init(args=args)
    node = DoorAlignNode()
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