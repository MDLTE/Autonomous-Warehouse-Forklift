#!/usr/bin/env python3
"""
Alineacion con pallet por QR, con ODOMETRIA LOCAL por encoders.
Mide la pose del QR desde lejos, calcula el punto G sobre la normal a la
distancia de standoff, y maniobra hasta ahi (giro-avance-giro) integrando
VelocityEncR/L localmente (no usa /odom global, que frente al estante no se
ubica). Al final ajuste visual. Carro de FRENTE. Solo ids de la whitelist.

ACTIVACION: solo funciona cuando nav_node publica SWEEPING:<label> en /nav_status.
Cualquier otro estado (IDLE, FOLLOWING, HANDOFF, LOGO_GOING...) lo desactiva.
"""

import json
import math
import numpy as np
from collections import deque
import rclpy
from rclpy import qos
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Twist

READY_DIST   = 0.27        # distancia alineada final (justo arriba del limite de vision ~25cm)
STANDOFF     = 0.50        # G de la maniobra: deja pista para alinear de frente
INSERT_PUSH  = 0.22        # avance a ciegas (DR) para meter horquillas
BACKOUT_DIST = 0.40        # reversa RECTA para sacar el pallet antes de girar -> AJUSTAR
RETURN_TOL   = 0.05        # tolerancia de posicion al volver al inicio
CAM_AHEAD = 0.066

# encoders ya calibrados
WHEEL_RADIUS = 0.0391
WHEEL_BASE   = 0.180

START_ENABLED = False      # arranca desactivado; espera SWEEPING:<label> de nav_node
TURN_SIGN = -1.0          # mapeo bearing->giro (centrado visual)
GIRO_SIGN = 1.0           # si los giros por odometria salen al reves, ponlo -1.0
PERP_SIGN = 1.0           # si en AJUSTE 'perp' se aleja de 0, ponlo -1.0

TIMER_STATES = {'GIRO1', 'AVANCE', 'GIRO2', 'INSERTAR', 'RETROCESO', 'RETORNO',
                'FORK_ENTRADA', 'FORK_AGARRE', 'FORK_ARRIBA'}

# labels de waypoints que deben activar la alineacion
SWEEP_LABELS = {
    'FOURCONVEYORS',
    'CENTERRIGHTSHELF-RIGHTSIDE',
    'CENTERSHELVES-CENTER',
    'CENTERLEFTSHELF-LEFTSIDE',
    'CENTERBOTTOMSHELF-TOPSIDE',
    'CENTERBOTTOMSHELF-BOTTOMSIDE',
}

FORK_CMDS = {
    1: {'entrada': 'loading_entrada',    'agarre': 'loading_agarre',    'arriba': 'rack_arriba_entrada'},
    2: {'entrada': 'rack_abajo_entrada', 'agarre': 'rack_abajo_agarre', 'arriba': 'rack_arriba_entrada'},
}


class CenterQR(Node):
    def __init__(self):
        super().__init__('center_qr')

        self.declare_parameter('test_mode', 1)
        self.test_mode = self.get_parameter('test_mode').value
        if self.test_mode not in FORK_CMDS:
            self.get_logger().warn(f'test_mode={self.test_mode} invalido; usando 1')
            self.test_mode = 1

        self.cmd_pub = self.create_publisher(Twist, '/qr/cmd_vel', 10)
        self.det_pub = self.create_publisher(Bool, '/qr_detected', 10)
        self.cen_pub = self.create_publisher(Bool, '/qr_centered', 10)
        self.done_pub = self.create_publisher(Bool, '/qr_align/done', 10)
        self.create_subscription(String, '/qr/detections', self.qr_cb, 10)
        self.qr_reset_pub = self.create_publisher(Bool, '/qr_detector/reset', 10)
        self.create_subscription(Bool, '/center_qr/enable', self._enable_cb, 10)

        self.create_subscription(Float32, '/VelocityEncR', self._encR, qos.qos_profile_sensor_data)
        self.create_subscription(Float32, '/VelocityEncL', self._encL, qos.qos_profile_sensor_data)
        self.fork_pub = self.create_publisher(Bool, '/forklift/insert', 10)
        self.create_subscription(Bool, '/forklift/done', self._fork_done_cb, 10)
        self.create_subscription(Bool, '/center_qr/stop', self._stop_cb, 10)
        self.create_subscription(String, '/fsm_cmd', self._cb_fsm_cmd, 10)
        self._task_status_pub = self.create_publisher(String, '/task_status', 10)
        self._empresa_reportada = False

        self.fork_cmd_pub = self.create_publisher(String, '/fork_cmd_name', 10)
        self.create_subscription(String, '/fork_status', self._cb_fork_status, 10)

        self.enabled = START_ENABLED
        self.state = 'SCAN'

        # odometria local (dead reckoning por encoders)
        self.wr = self.wl = 0.0
        self.have_enc = False
        self.dr_x = self.dr_y = self.dr_yaw = 0.0
        self.dr_last = self.get_clock().now()

        # deteccion / centrado
        self.bearing_obs   = math.radians(7)
        self.bearing_tol   = math.radians(5)
        self.dist_tol      = 0.04
        self.psi_tol       = math.radians(7)     # perpendicular para LISTO
        self.lat_tol       = 0.04
        self.k_bearing     = 1.2
        self.k_v           = 0.5
        self.v_max         = 0.12
        self.w_max         = 0.5
        # lazo cerrado del ajuste final (cerca y de frente -> pose fiable)
        self.k_psi      = 0.5
        self.k_e        = 0.7
        self.perp_cap   = 0.30
        self.v_rev      = 0.04                    # reversa chica solo si se paso de cerca
        # estos 5 los fija _apply_size_profile() segun el tamano del QR detectado
        self.remaneuver_perp = math.radians(25)
        self.gross_need = 1
        self.gross_count = 0
        self.psi_buf  = deque(maxlen=1)
        self.elat_buf = deque(maxlen=1)
        self.use_subpix = False
        self.listo_need = 5
        self.listo_count = 0

        # maniobra
        self.observe_frames = 15
        self.obs_buf = []
        self.Gx = self.Gy = self.Ghead = 0.0
        self.giro_speed  = 0.4
        self.drive_speed = 0.12
        self.yaw_tol     = math.radians(4)
        self.pos_tol     = 0.04
        self.retries     = 0
        self.max_retries = 2

        # punto inicial (para regresar) + insercion
        self.home_x = self.home_y = 0.0
        self.home_set = False
        self.ins_x0 = None
        self.ins_signaled = False
        self.ins_t0 = self.get_clock().now()
        self.insert_wait = 4.0
        self.fork_done = False
        self.reverse_speed = 0.10
        self.retro_x0 = None
        self.retro_yaw0 = 0.0
        self.estop = False
        self.done_published = False

        # estados FORK_*
        self.fork_cmd_sent  = False
        self.fork_cmd_t0    = self.get_clock().now()
        self.fork_cmd_wait  = 3.0   # timeout de seguridad si no llega DONE
        self._fork_status_done = False

        # scan / reacquire
        self.scan_omega  = 0.35
        self.scan_period = 4.0
        self.scanning    = False
        self.scan_dir    = 1.0
        self.scan_first  = True
        self.scan_t0     = self.get_clock().now()
        self.reacq_omega = 0.25
        self.qr_lost     = 999
        self.relost_scan = 60

        # id del QR seguido + su bearing (lo reporta el detector; sirve para reacquire)
        self.locked_id = ''
        self.last_lock_brg = None

        # tamano/perfil de control: lo manda el detector en cada mensaje ('big'/'small').
        # arrancamos en grande (perfil apretado) hasta que llegue la primera lectura.
        self.size_locked = False
        self.applied_size = 'big'
        self._apply_size_profile(True)
        self._fresh = False
        self.last_dets = []
        # creep: acercarse despacio hasta tener lectura estable (QR chico)
        self.creep_speed = 0.07
        self.creep_need  = 5
        self.creep_ready = 0.62
        self.creep_min   = 0.48
        self.creep_bear  = math.radians(12)
        self.creep_fresh = 0

        self.dbg = dict(state='SCAN', dist=0.0, bearing=0.0, psi=0.0, e_lat=0.0,
                        v=0.0, w=0.0, gerr=0.0)

        self.create_timer(0.02, self.ctrl_timer)   # 50 Hz: odometria + maniobra
        self.get_logger().info('CenterQR (odometria local) iniciado — esperando SWEEPING')

    # ---------- callbacks ----------
    def _encR(self, m): self.wr = m.data; self.have_enc = True
    def _encL(self, m): self.wl = m.data; self.have_enc = True
    def _fork_done_cb(self, m): self.fork_done = bool(m.data)


    def qr_cb(self, msg: String):
        # llega del qr_detector; arma el pose del QR 'locked' y corre la FSM de vision
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        qrs = data.get('qrs', [])
        if data.get('locked_id'):
            new_id = data['locked_id']
            if new_id and new_id != self.locked_id:
                self.locked_id = new_id
                self._pub_status('EMPRESA', self.locked_id)
                self.get_logger().info(f'[QR] Empresa detectada: {self.locked_id}')
            elif new_id:
                self.locked_id = new_id

        e = next((q for q in qrs if q.get('locked')), None)
        pose = None
        if e is not None:
            self._fresh = bool(e.get('fresh'))
            self.last_lock_brg = math.radians(e['bearing_deg'])
            sz = e.get('size', 'big')
            if sz != self.applied_size:          # ajusta tolerancias segun tamano del QR
                self._apply_size_profile(sz == 'big')
                self.applied_size = sz
                self.size_locked = True
            pose = dict(dist=e['dist'], bearing=math.radians(e['bearing_deg']),
                        psi=math.radians(e['psi_deg']), e_lat=e['e_lat'],
                        gamma=math.radians(e['gamma_deg']),
                        tx=e['tx'], tz=e['tz'], nx=e['nx'], nz=e['nz'])
        else:
            self._fresh = False

        self.det_pub.publish(Bool(data=pose is not None))
        if pose is not None:
            self.dbg.update(dist=pose['dist'], bearing=pose['bearing'],
                            psi=pose['psi'], e_lat=pose['e_lat'])

        if self.estop:
            self._stop()
            self.dbg.update(state='PARO')
        elif not self.enabled:
            self.state = 'DESACTIVADO'
            self._stop()
        else:
            s = self.state
            if s in TIMER_STATES:
                pass                          # lo maneja el timer (sin camara)
            elif s == 'SCAN':
                if pose is not None:
                    if not self.home_set:     # guarda el punto inicial para regresar
                        self.home_x, self.home_y = self.dr_x, self.dr_y
                        self.home_set = True
                    self.scanning = False
                    self.creep_fresh = 0
                    self.qr_lost = 0
                    self.state = 'APROXIMAR'
                else:
                    self.qr_lost += 1
                    if self.last_lock_brg is not None and self.qr_lost < self.relost_scan:
                        self._reacquire()
                    else:
                        self._scan()
            elif s == 'APROXIMAR':
                self.st_aproximar(pose)
            elif s == 'OBSERVAR':
                self.st_observe(pose)
            elif s == 'REVERIFICAR':
                self.st_reverificar(pose)
            elif s == 'AJUSTE':
                self.st_ajuste(pose)
            elif s in ('LISTO', 'FIN'):
                self._stop()

                if s == 'FIN' and not self.done_published:
                    self.done_pub.publish(Bool(data=True))
                    self.done_published = True
                    self._pub_status('DONE')
                    self.get_logger().info('[center_qr] Maniobra completa -> /qr_align/done')

        self.dbg.update(state='PARO' if self.estop else self.state)
        self.get_logger().info(
            f"[DBG] {self.dbg['state']} qr={pose is not None} id={self.locked_id} "
            f"dist={self.dbg['dist']:.3f} bearing={math.degrees(self.dbg['bearing']):+.1f} "
            f"perp={math.degrees(self.dbg['psi']):+.1f} elat={self.dbg['e_lat']*100:+.1f}cm "
            f"dr=({self.dr_x:+.2f},{self.dr_y:+.2f},{math.degrees(self.dr_yaw):+.0f}) "
            f"gerr={self.dbg['gerr']:+.3f}",
            throttle_duration_sec=0.4)
    def _cb_fsm_cmd(self, msg):
        cmd = msg.data.strip()
        if cmd == 'T3':
            self.get_logger().info('[FSM_CMD] T3 → activando center_qr')
            self._empresa_reportada = False
            self.enabled = True
            self.state = 'SCAN'   # ← resetear estado explícitamente
            self.scanning = False
            self.qr_lost = 999
        elif cmd == 'STOP':
            self._enable_cb(Bool(data=False))

    def _pub_status(self, status, data=''):
        msg = String()
        msg.data = f'QR:{status}:{data}'
        self._task_status_pub.publish(msg)
        self.get_logger().info(f'[task_status] QR:{status}:{data}')
    def _stop_cb(self, m):
        self.estop = bool(m.data)
        if self.estop:
            self._stop()
            self.get_logger().warn('PARO DE EMERGENCIA activado')

    def _enable_cb(self, msg: Bool):
        self.enabled = msg.data
        if not self.enabled:
            self._reset()
            self._stop()

    def _reset(self):
        self.state = 'SCAN'
        self.scanning = False
        self.qr_lost = 999
        self.last_lock_brg = None
        self.locked_id = ''
        self.obs_buf = []
        self.retries = 0
        self.listo_count = 0
        self.home_set = False
        self.ins_x0 = None
        self.ins_signaled = False
        self.fork_done = False
        self.retro_x0 = None
        self.creep_fresh = 0
        self.gross_count = 0
        self.retries = 0
        self.psi_buf.clear(); self.elat_buf.clear()
        self.applied_size = 'big'
        self._apply_size_profile(True)              # vuelve al perfil grande por defecto
        self.size_locked = False
        self.qr_reset_pub.publish(Bool(data=True))  # que el detector reevalue el tamano
        self.done_published = False
        self._empresa_reportada = False
        self.fork_cmd_sent = False
        self._fork_status_done = False

    # ---------- utilidades ----------
    def _apply_size_profile(self, big):
        # GRANDE = logica original (apretada, sin filtro, re-maniobra de 1 frame)
        # CHICO  = logica nueva (sub-pixel + mediana + tolerancias y anti-rebote)
        if big:
            self.psi_tol = math.radians(5)
            self.lat_tol = 0.03
            self.remaneuver_perp = math.radians(25)
            self.gross_need = 1
            self.use_subpix = False
            n = 1
        else:
            self.psi_tol = math.radians(7)
            self.lat_tol = 0.04
            self.remaneuver_perp = math.radians(30)
            self.gross_need = 5
            self.use_subpix = True
            n = 5
        self.psi_buf = deque(maxlen=n)
        self.elat_buf = deque(maxlen=n)
        self.gross_count = 0

    @staticmethod
    def _wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _stop(self):
        self.cmd_pub.publish(Twist())

    def _send(self, v, w):
        c = Twist()
        c.linear.x = float(v)
        c.angular.z = float(w)
        self.cmd_pub.publish(c)
        self.dbg.update(v=v, w=w)

    def _pub_fork_cmd(self, key):
        cmd = FORK_CMDS[self.test_mode][key]
        m = String()
        m.data = cmd
        self.fork_cmd_pub.publish(m)
        self.get_logger().info(f'[FORK] /fork_cmd_name <- "{cmd}"')

    def _cb_fork_status(self, msg: String):
        if msg.data.startswith('DONE'):
            self._fork_status_done = True
            self.get_logger().info('[FORK] fork_status DONE recibido')

    # ---------- odometria local + maniobra (50 Hz) ----------
    def ctrl_timer(self):
        now = self.get_clock().now()
        dt = (now - self.dr_last).nanoseconds * 1e-9
        self.dr_last = now
        if 0 < dt < 0.5 and self.have_enc:
            v = WHEEL_RADIUS * (self.wr + self.wl) / 2.0
            w = WHEEL_RADIUS * (self.wr - self.wl) / WHEEL_BASE
            self.dr_x += v * math.cos(self.dr_yaw) * dt
            self.dr_y += v * math.sin(self.dr_yaw) * dt
            self.dr_yaw = self._wrap(self.dr_yaw + w * dt)

        if self.estop:
            self._stop()
            return
        if not self.enabled:
            return
        s = self.state
        if s == 'GIRO1':
            self.st_giro1()
        elif s == 'AVANCE':
            self.st_avance()
        elif s == 'GIRO2':
            self.st_giro2()
        elif s == 'FORK_ENTRADA':
            self.st_fork_entrada()
        elif s == 'INSERTAR':
            self.st_insertar()
        elif s == 'FORK_AGARRE':
            self.st_fork_agarre()
        elif s == 'RETROCESO':
            self.st_retroceso()
        elif s == 'FORK_ARRIBA':
            self.st_fork_arriba()
        elif s == 'RETORNO':
            self.st_retorno()

    def st_giro1(self):
        tgt = math.atan2(self.Gy - self.dr_y, self.Gx - self.dr_x)
        err = self._wrap(tgt - self.dr_yaw)
        self.dbg.update(gerr=err)
        if abs(err) < self.yaw_tol:
            self.state = 'AVANCE'
            self._stop()
            return
        self._send(0.0, GIRO_SIGN * self.giro_speed * (1.0 if err > 0 else -1.0))

    def st_avance(self):
        dx, dy = self.Gx - self.dr_x, self.Gy - self.dr_y
        d = math.hypot(dx, dy)
        self.dbg.update(gerr=d)
        if d < self.pos_tol:
            self.state = 'GIRO2'
            self._stop()
            return
        herr = self._wrap(math.atan2(dy, dx) - self.dr_yaw)
        v = min(self.drive_speed, self.k_v * d + 0.04)
        self._send(v, GIRO_SIGN * float(np.clip(1.5 * herr, -0.4, 0.4)))

    def st_giro2(self):
        err = self._wrap(self.Ghead - self.dr_yaw)
        self.dbg.update(gerr=err)
        if abs(err) < self.yaw_tol:
            self.state = 'REVERIFICAR'
            self.qr_lost = 999
            self._stop()
            return
        self._send(0.0, GIRO_SIGN * self.giro_speed * (1.0 if err > 0 else -1.0))

    def st_fork_entrada(self):
        if not self.fork_cmd_sent:
            self._fork_status_done = False          # limpiar DONE de estados anteriores
            self._pub_fork_cmd('entrada')
            self.fork_cmd_t0 = self.get_clock().now()
            self.fork_cmd_sent = True
            return                                  # no evaluar condicion en el mismo tick
        waited = (self.get_clock().now() - self.fork_cmd_t0).nanoseconds * 1e-9
        if self._fork_status_done or waited >= self.fork_cmd_wait:
            self.fork_cmd_sent = False
            self.state = 'INSERTAR'

    def st_insertar(self):
        # ya alineado a READY_DIST: empuja a ciegas (DR) para meter horquillas,
        # avisa al mecanismo (FPGA) y espera a que termine
        if self.ins_x0 is None:
            self.ins_x0, self.ins_y0 = self.dr_x, self.dr_y
        pushed = math.hypot(self.dr_x - self.ins_x0, self.dr_y - self.ins_y0)
        self.dbg.update(gerr=INSERT_PUSH - pushed)
        if pushed < INSERT_PUSH:
            self._send(self.drive_speed * 0.5, 0.0)     # avance lento, recto, a ciegas
            return
        self._stop()
        if not self.ins_signaled:                        # horquillas en posicion -> activa
            self.fork_pub.publish(Bool(data=True))
            self.ins_t0 = self.get_clock().now()
            self.ins_signaled = True
        waited = (self.get_clock().now() - self.ins_t0).nanoseconds * 1e-9
        if self.fork_done or waited > self.insert_wait:
            self.state = 'FORK_AGARRE'

    def st_fork_agarre(self):
        if not self.fork_cmd_sent:
            self._fork_status_done = False
            self._pub_fork_cmd('agarre')
            self.fork_cmd_t0 = self.get_clock().now()
            self.fork_cmd_sent = True
            return
        waited = (self.get_clock().now() - self.fork_cmd_t0).nanoseconds * 1e-9
        if self._fork_status_done or waited >= self.fork_cmd_wait:
            self.fork_cmd_sent = False
            self.state = 'RETROCESO'

    def st_retroceso(self):
        # saca el pallet en reversa RECTA antes de girar (no pegarle a pallets de al lado)
        if self.retro_x0 is None:
            self.retro_x0, self.retro_y0 = self.dr_x, self.dr_y
            self.retro_yaw0 = self.dr_yaw
        moved = math.hypot(self.dr_x - self.retro_x0, self.dr_y - self.retro_y0)
        self.dbg.update(gerr=BACKOUT_DIST - moved)
        if moved < BACKOUT_DIST:
            hold = self._wrap(self.retro_yaw0 - self.dr_yaw)
            self._send(-self.reverse_speed, GIRO_SIGN * float(np.clip(1.0 * hold, -0.2, 0.2)))
            return
        self._stop()
        self.state = 'FORK_ARRIBA'

    def st_fork_arriba(self):
        if not self.fork_cmd_sent:
            self._fork_status_done = False
            self._pub_fork_cmd('arriba')
            self.fork_cmd_t0 = self.get_clock().now()
            self.fork_cmd_sent = True
            return
        waited = (self.get_clock().now() - self.fork_cmd_t0).nanoseconds * 1e-9
        if self._fork_status_done or waited >= self.fork_cmd_wait:
            self.fork_cmd_sent = False
            self.state = 'RETORNO'

    def st_retorno(self):
        # regresa al punto inicial por DR; el rumbo final no importa
        dx, dy = self.home_x - self.dr_x, self.home_y - self.dr_y
        d = math.hypot(dx, dy)
        self.dbg.update(gerr=d)
        if d < RETURN_TOL:
            self.state = 'FIN'
            self._stop()
            return
        herr = self._wrap(math.atan2(dy, dx) - self.dr_yaw)
        if abs(herr) > math.radians(8):
            self._send(0.0, GIRO_SIGN * self.giro_speed * (1.0 if herr > 0 else -1.0))
        else:
            v = min(self.drive_speed, self.k_v * d + 0.04)
            self._send(v, GIRO_SIGN * float(np.clip(1.5 * herr, -0.4, 0.4)))

    def goal_in_robot(self, pose):
        qx_r = CAM_AHEAD + pose['tz']
        qy_r = -pose['tx']
        nf, nl = pose['nz'], -pose['nx']
        ln = math.hypot(nf, nl) + 1e-9
        nf, nl = nf / ln, nl / ln
        gx_r = qx_r + nf * STANDOFF
        gy_r = qy_r + nl * STANDOFF
        gh_r = math.atan2(-nl, -nf)
        return gx_r, gy_r, gh_r

    def lock_goal(self):
        arr = np.array(self.obs_buf)
        gx_r, gy_r, gh_r = np.median(arr[:, 0]), np.median(arr[:, 1]), np.median(arr[:, 2])
        c, s = math.cos(self.dr_yaw), math.sin(self.dr_yaw)
        self.Gx = self.dr_x + gx_r * c - gy_r * s
        self.Gy = self.dr_y + gx_r * s + gy_r * c
        self.Ghead = self._wrap(self.dr_yaw + gh_r)

    # ---------- scan / reacquire ----------
    def _scan(self):
        now = self.get_clock().now()
        if not self.scanning:
            self.scanning, self.scan_dir, self.scan_first, self.scan_t0 = True, 1.0, True, now
        if (now - self.scan_t0).nanoseconds * 1e-9 > self.scan_period * (0.5 if self.scan_first else 1.0):
            self.scan_dir *= -1.0
            self.scan_first = False
            self.scan_t0 = now
        self._send(0.0, self.scan_dir * self.scan_omega)

    def _reacquire(self):
        d = 1.0 if (self.last_lock_brg is not None and self.last_lock_brg > 0) else -1.0
        self._send(0.0, TURN_SIGN * self.reacq_omega * d)

    # ---------- estados de vision ----------
    def st_aproximar(self, pose):
        if pose is None:
            self.creep_fresh = 0
            self.qr_lost += 1
            if self.last_lock_brg is not None and self.qr_lost < self.relost_scan:
                self._reacquire()
            else:
                self.state = 'SCAN'
            return
        self.qr_lost = 0
        self.creep_fresh = self.creep_fresh + 1 if self._fresh else 0
        stable = self.creep_fresh >= self.creep_need
        if pose['dist'] <= self.creep_min or (stable and pose['dist'] <= self.creep_ready):
            self._stop()
            self.obs_buf = []
            self.state = 'OBSERVAR'
            return
        w = TURN_SIGN * float(np.clip(self.k_bearing * pose['bearing'], -self.w_max, self.w_max))
        v = self.creep_speed if abs(pose['bearing']) < self.creep_bear else 0.0
        self._send(v, w)

    def st_observe(self, pose):
        if pose is None:
            self.obs_buf = []
            self.state = 'SCAN'
            return
        if abs(pose['bearing']) > self.bearing_obs:
            self.obs_buf = []
            self._send(0.0, TURN_SIGN * float(np.clip(self.k_bearing * pose['bearing'],
                                                      -self.w_max, self.w_max)))
            return
        self.obs_buf.append(self.goal_in_robot(pose))
        self._stop()
        if len(self.obs_buf) >= self.observe_frames:
            if not self.have_enc:
                self.get_logger().warn('sin encoders, no puedo maniobrar', throttle_duration_sec=2.0)
                self.obs_buf = []
                return
            self.lock_goal()
            self.obs_buf = []
            self.state = 'GIRO1'

    def st_reverificar(self, pose):
        if pose is not None and abs(pose['bearing']) < math.radians(15):
            self.psi_buf.clear(); self.elat_buf.clear()
            self.gross_count = 0
            self.state = 'AJUSTE'
            return
        self.qr_lost += 1
        if self.last_lock_brg is not None and self.qr_lost < self.relost_scan:
            self._reacquire()
        else:
            self._scan()

    def st_ajuste(self, pose):
        if pose is None:
            self.qr_lost += 1
            if self.last_lock_brg is not None and self.qr_lost < self.relost_scan:
                self._reacquire()
            else:
                self.state = 'REVERIFICAR'
            return
        self.qr_lost = 0
        bearing, dist = pose['bearing'], pose['dist']
        self.psi_buf.append(pose['psi'])
        self.elat_buf.append(pose['e_lat'])
        psi = float(np.median(self.psi_buf))
        e_lat = float(np.median(self.elat_buf))
        e_dist = dist - READY_DIST
        centered = abs(bearing) < self.bearing_tol
        at_dist  = abs(e_dist) < self.dist_tol
        square   = abs(psi) < self.psi_tol
        on_axis  = abs(e_lat) < self.lat_tol

        if abs(psi) > self.remaneuver_perp:
            self.gross_count += 1
        else:
            self.gross_count = 0
        if self.gross_count >= self.gross_need and self.retries < self.max_retries:
            self.retries += 1
            self.gross_count = 0
            self.obs_buf = []
            self.psi_buf.clear(); self.elat_buf.clear()
            self.listo_count = 0
            self.state = 'OBSERVAR'
            self._stop()
            return

        if centered and at_dist and square and on_axis:
            self.listo_count += 1
            self._stop()
            if self.listo_count >= self.listo_need:
                self.cen_pub.publish(Bool(data=True))
                self.state = 'FORK_ENTRADA'
            return
        self.listo_count = 0
        self.cen_pub.publish(Bool(data=False))

        perp = float(np.clip(PERP_SIGN * (self.k_psi * psi + self.k_e * e_lat),
                             -self.perp_cap, self.perp_cap))
        w = TURN_SIGN * float(np.clip(self.k_bearing * bearing + perp, -self.w_max, self.w_max))
        slow = float(np.clip(1.0 - abs(bearing) / math.radians(25), 0.2, 1.0))
        v = float(np.clip(self.k_v * e_dist, -self.v_rev, self.v_max)) * slow
        self._send(v, w)


def main(args=None):
    rclpy.init(args=args)
    node = CenterQR()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node.cmd_pub.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()