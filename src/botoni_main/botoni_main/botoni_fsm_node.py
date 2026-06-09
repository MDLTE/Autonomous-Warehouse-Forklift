#!/usr/bin/env python3
"""
botoni_fsm_node.py  v4
======================
FSM con arquitectura de 2 topics: /fsm_cmd y /task_status

TOPICS PUB:
  /fsm_state        String  — nombre del estado
  /fsm_mission      String  — JSON completo para UI
  /empresa_detected String  — empresa leída del QR
  /fsm_cmd          String  — comando a nodos: "T1"|"T2"|"T6:E"|"T7:E"|"STOP"

TOPICS SUB:
  /task_status  String  — "NAV:DONE:" | "NAV:QR_FOUND:" | "QR:EMPRESA:X" |
                          "QR:DONE:" | "ALIGN:DONE:" | "ALIGN:FAILED:"
  /ui_cmd       String JSON
  /ui_mode      String
  /ui_goto_step String
  /voice_cmd    String
"""

import rclpy, json, time, threading
from enum import Enum, auto
from rclpy.node import Node
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist


class State(Enum):
    STALL        = auto()
    IDLE         = auto()
    PAUSED       = auto()
    ERROR        = auto()
    NAV_GET      = auto()   # T1 (Rack) | T2 (Dock)
    ALIGN_GET    = auto()   # T3/T4 — QR → empresa + fork get
    NAV_LOGO     = auto()   # T6
    ALIGN_LOGO   = auto()   # T7/T8 — logo align + fork logo
    MISSION_DONE = auto()


class BotoniMaster(Node):

    def __init__(self):
        super().__init__('botoni_master')
        self.get_logger().info("=" * 55)
        self.get_logger().info("  BOTONI MASTER v4")
        self.get_logger().info("=" * 55)

        # ── Variables de misión ───────────────────────────────
        self._state   = State.STALL
        self._pallet  = 0
        self._get     = None   # "Rack" | "Dock"
        self._logo    = None   # "Emezon" | "Walmar" | "Popsi"
        self._empresa = None   # leída del QR
        self._mode    = "FLOW"
        self._log     = "Listo. Selecciona GET y empresa, luego START."

        # ── Pause snapshot ────────────────────────────────────
        self._prev_state   = None
        self._prev_pallet  = 0
        self._prev_empresa = None

        # ── Control ───────────────────────────────────────────
        self._running    = False
        self._lock       = threading.Lock()
        self._fsm_thread = None

        # ── Eventos ───────────────────────────────────────────
        self._task_ok    = threading.Event()
        self._task_fail  = threading.Event()

        # ── Publishers ────────────────────────────────────────
        self.pub_state   = self.create_publisher(String, '/fsm_state',        10)
        self.pub_mission = self.create_publisher(String, '/fsm_mission',      10)
        self.pub_empresa = self.create_publisher(String, '/empresa_detected', 10)
        self.pub_vel     = self.create_publisher(Twist,  '/cmd_vel',          10)
        self.pub_cmd     = self.create_publisher(String, '/fsm_cmd',          10)

        # ── Subscribers ───────────────────────────────────────
        # ── Mux cmd_vel ───────────────────────────────────────────
        self.create_subscription(Twist, '/nav/cmd_vel',   self._mux_nav,   10)
        self.create_subscription(Twist, '/qr/cmd_vel',    self._mux_qr,    10)
        self.create_subscription(Twist, '/align/cmd_vel', self._mux_align, 10)
        self.create_subscription(String,  '/task_status',  self._cb_task_status, 10)
        self.create_subscription(String,  '/ui_cmd',       self._cb_ui_cmd,      10)
        self.create_subscription(String,  '/ui_mode',      self._cb_ui_mode,     10)
        self.create_subscription(String,  '/ui_goto_step', self._cb_goto_step,   10)
        self.create_subscription(String,  '/voice_cmd',    self._cb_voice,       10)

        self.create_timer(0.2, self._publish_state)
        self._publish_state()
        self.get_logger().info("[STALL] Listo.")

    # ─────────────────────────────────────────
    #  CALLBACKS
    # ─────────────────────────────────────────
    def _mux_nav(self, msg):
        if self._state in (State.NAV_GET, State.NAV_LOGO) and self._mode != 'MANUAL':
            self.pub_vel.publish(msg)

    def _mux_qr(self, msg):
        if self._state == State.ALIGN_GET and self._mode != 'MANUAL':
            self.pub_vel.publish(msg)

    def _mux_align(self, msg):
        if self._state == State.ALIGN_LOGO and self._mode != 'MANUAL':
            self.pub_vel.publish(msg)

    def _cb_task_status(self, msg):
        raw = msg.data.strip()
        self.get_logger().info(f"[STATUS] {raw}")
        parts = raw.split(":", 2)
        if len(parts) < 2:
            return
        nodo, status = parts[0].upper(), parts[1].upper()
        data = parts[2] if len(parts) > 2 else ""

        # NAV done/found
        if nodo == "NAV":
            if status in ("DONE", "QR_FOUND", "LOGO_REACHED"):
                self._task_ok.set()
            elif status == "FAILED":
                self._task_fail.set()

        # QR — puede mandar empresa antes del done
        elif nodo == "QR":
            if status == "EMPRESA" and data:
                self._empresa = data.strip()
                self.get_logger().info(f"[QR] Empresa detectada: {self._empresa}")
                pub = String(); pub.data = self._empresa
                self.pub_empresa.publish(pub)
                self._publish_state()
            elif status == "DONE":
                self._task_ok.set()
            elif status == "FAILED":
                self._task_fail.set()

        # ALIGN done
        elif nodo == "ALIGN":
            if status == "DONE":
                self._task_ok.set()
            elif status == "FAILED":
                self._task_fail.set()

    def _cb_ui_mode(self, msg):
        self._mode = msg.data.upper().strip()
        self.get_logger().info(f"[MODE] → {self._mode}")

    def _cb_goto_step(self, msg):
        if self._state not in (State.PAUSED, State.STALL, State.IDLE):
            self.get_logger().warn("[GOTO] Solo desde PAUSED/STALL/IDLE")
            return
        try:
            target = State[msg.data.upper()]
            self.get_logger().info(f"[GOTO] Saltando a {target.name}")
            self._set_state(target)
            self._start_fsm_thread()
        except KeyError:
            self.get_logger().error(f"[GOTO] Estado desconocido: {msg.data}")

    def _cb_ui_cmd(self, msg):
        try:
            data    = json.loads(msg.data)
            action  = data.get("action", "")
            payload = data.get("payload", {})
        except Exception:
            return

        if action == "START":
            self._handle_start()
        elif action == "STOP":
            self._handle_stop()
        elif action == "PAUSE":
            self._handle_pause()
        elif action == "SELECT_MISSION":
            m = payload.get("mission", "")
            self._get = "Rack" if m == "rack_truck" else "Dock"
            self.get_logger().info(f"[CONFIG] GET={self._get}")
            self._maybe_idle()
        elif action == "SELECT_COMPANY":
            co = payload.get("company", "")
            self._logo = co  # guardamos nombre completo: "Emezon"|"Walmar"|"Popsi"
            self.get_logger().info(f"[CONFIG] LOGO={self._logo}")
            self._maybe_idle()
        elif action == "SET_MODE":
            self._mode = payload.get("mode", "FLOW").upper()
            self.get_logger().info(f"[MODE] {self._mode}")

    def _cb_voice(self, msg):
        cmd = msg.data.upper().strip()
        if cmd in ("START", "INICIO", "GO"):   self._handle_start()
        elif cmd in ("STOP", "ALTO", "PARA"):  self._handle_stop()
        elif cmd in ("PAUSE", "PAUSA"):        self._handle_pause()
        elif cmd in ("RACK", "ESTANTE"):
            self._get = "Rack"; self._maybe_idle()
        elif cmd in ("DOCK", "CARGA", "LOADING"):
            self._get = "Dock"; self._maybe_idle()

    # ─────────────────────────────────────────
    #  CONTROL
    # ─────────────────────────────────────────

    def _handle_stop(self):
        self.get_logger().info("[CMD] STOP")
        with self._lock:
            self._running = False
        self.pub_vel.publish(Twist())
        self._send_cmd("STOP")
        self._pallet  = 0
        self._get     = None
        self._logo    = None
        self._empresa = None
        self._prev_state = None
        self._task_ok.clear()
        self._task_fail.clear()
        self._log = "STOP — variables vaciadas."
        self._set_state(State.STALL)
        self.get_logger().info("[STALL] Reset completo.")

    def _handle_pause(self):
        if self._state in (State.STALL, State.IDLE, State.PAUSED, State.ERROR):
            return
        self.get_logger().info("[CMD] PAUSE")
        with self._lock:
            self._running      = False
            self._prev_state   = self._state
            self._prev_pallet  = self._pallet
            self._prev_empresa = self._empresa
        self.pub_vel.publish(Twist())
        self._send_cmd("STOP")
        self._set_state(State.PAUSED)
        self._log = f"Pausado en {self._prev_state.name} | P={self._prev_pallet}"
        self.get_logger().info(f"[PAUSED] {self._prev_state.name}")

    def _handle_start(self):
        if self._state == State.PAUSED:
            self.get_logger().info("[CMD] RESUME")
            self._pallet  = self._prev_pallet
            self._empresa = self._prev_empresa
            self._set_state(self._prev_state)
            self._log = f"Reanudando: {self._state.name}"
            self._start_fsm_thread()

        elif self._state in (State.STALL, State.IDLE):
            if not self._get:
                self._log = "⚠ Falta GET (Rack o Dock)"; return
            self.get_logger().info(f"[CMD] START | GET={self._get} LOGO={self._logo}")
            self._pallet  = 0
            self._empresa = None
            self._set_state(State.NAV_GET)
            self._log = f"Iniciando: GET={self._get}"
            self._start_fsm_thread()
        else:
            self.get_logger().warn(f"[START] Ignorado en {self._state.name}")

    def _maybe_idle(self):
        if self._state == State.STALL and self._get:
            self._set_state(State.IDLE)
            self._log = f"Listo: GET={self._get} — presiona START"

    # ─────────────────────────────────────────
    #  FSM THREAD
    # ─────────────────────────────────────────

    def _start_fsm_thread(self):
        with self._lock:
            if self._fsm_thread and self._fsm_thread.is_alive():
                self._running = False
                self._fsm_thread.join(timeout=2.0)
            self._running = True
            self._fsm_thread = threading.Thread(target=self._fsm_loop, daemon=True)
            self._fsm_thread.start()

    def _fsm_loop(self):
        self.get_logger().info("[FSM] Thread iniciado.")
        try:
            while rclpy.ok() and self._running:
                s = self._state
                if   s == State.NAV_GET:     self._run_nav_get()
                elif s == State.ALIGN_GET:   self._run_align_get()
                elif s == State.NAV_LOGO:    self._run_nav_logo()
                elif s == State.ALIGN_LOGO:  self._run_align_logo()
                elif s == State.MISSION_DONE:self._run_mission_done(); break
                elif s in (State.STALL, State.IDLE, State.PAUSED, State.ERROR): break
                else: break
                if not self._running: break
        except Exception as e:
            self.get_logger().error(f"[FSM] {e}")
            self._set_state(State.ERROR)
            self._log = f"ERROR: {e}"
        self.get_logger().info("[FSM] Thread terminado.")

    # ─────────────────────────────────────────
    #  ESTADOS
    # ─────────────────────────────────────────

    def _run_nav_get(self):
        cmd = "T1" if self._get == "Rack" else "T2"
        self.get_logger().info(f"[NAV_GET] → {cmd}")
        self._log = f"NAV GET — {cmd} ({self._get})"
        self._fire_cmd(cmd)
        if not self._await_task(): return
        self.get_logger().info("[NAV_GET] ✓")
        if not self._step_or_pause(State.ALIGN_GET): return

    def _run_align_get(self):
        # center_qr se activa automáticamente por nav_status SWEEPING
        # Solo esperamos: empresa via QR:EMPRESA:X, luego QR:DONE
        self.get_logger().info("[ALIGN_GET] Esperando QR + empresa")
        self._log = "ALIGN GET — esperando QR y empresa"
        self._fire_cmd("T3")  # activa center_qr si necesita trigger explícito
        if not self._await_task(): return

        # Si no llegó empresa por status, pausar para selección manual
        if not self._empresa:
            self.get_logger().info("[ALIGN_GET] Sin empresa — pausando")
            self._log = "⚠ Selecciona empresa en UI y presiona START"
            with self._lock:
                self._prev_state   = State.NAV_LOGO
                self._prev_pallet  = self._pallet
                self._prev_empresa = self._empresa
                self._running      = False
            self.pub_vel.publish(Twist())
            self._set_state(State.PAUSED)
            return

        self._pallet = 1
        self.get_logger().info(f"[ALIGN_GET] ✓ empresa={self._empresa} P=1")
        if not self._step_or_pause(State.NAV_LOGO): return

    def _run_nav_logo(self):
        empresa = self._empresa or self._logo or "?"
        self.get_logger().info(f"[NAV_LOGO] → T6:{empresa}")
        self._log = f"NAV LOGO — T6 | empresa={empresa}"
        self._fire_cmd(f"T6:{empresa}")
        if not self._await_task(): return
        self.get_logger().info("[NAV_LOGO] ✓")
        if not self._step_or_pause(State.ALIGN_LOGO): return

    def _run_align_logo(self):
        empresa = self._empresa or self._logo or "?"
        self.get_logger().info(f"[ALIGN_LOGO] → T7:{empresa}")
        self._log = f"ALIGN LOGO — T7 | empresa={empresa}"
        self._fire_cmd(f"T7:{empresa}")
        if not self._await_task(): return
        self.get_logger().info("[ALIGN_LOGO] ✓")
        self._set_state(State.MISSION_DONE)

    def _run_mission_done(self):
        self.get_logger().info("=" * 55)
        self.get_logger().info("  ✓ MISIÓN COMPLETADA")
        self.get_logger().info("=" * 55)
        self._log = "✓ Misión completada."
        self._set_state(State.MISSION_DONE)
        time.sleep(2.0)
        self._pallet  = 0
        self._empresa = None
        self._set_state(State.STALL)
        self._log = "Misión completada. Lista para nueva misión."

    # ─────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────

    def _step_or_pause(self, next_state: State) -> bool:
        if self._mode == "MANUAL":
            self.get_logger().info(f"[MANUAL] Pausando antes de {next_state.name}")
            with self._lock:
                self._prev_state   = next_state
                self._prev_pallet  = self._pallet
                self._prev_empresa = self._empresa
                self._running      = False
            self.pub_vel.publish(Twist())
            self._set_state(State.PAUSED)
            self._log = f"MANUAL — listo para {next_state.name}. Presiona START."
            return False
        else:
            self._set_state(next_state)
            return True

    def _fire_cmd(self, cmd: str):
        """Publica en /fsm_cmd y limpia eventos."""
        self._task_ok.clear()
        self._task_fail.clear()
        self._send_cmd(cmd)
        self.get_logger().info(f"  ↳ /fsm_cmd → {cmd}")

    def _send_cmd(self, cmd: str):
        msg = String(); msg.data = cmd
        self.pub_cmd.publish(msg)

    def _await_task(self, timeout=300.0) -> bool:
        while self._running:
            if self._task_ok.wait(timeout=0.2): return True
            if self._task_fail.is_set():
                self.get_logger().error("  [WAIT] Task falló.")
                self._set_state(State.ERROR)
                self._log = "ERROR: Task falló."
                return False
            timeout -= 0.2
            if timeout <= 0:
                self.get_logger().warn("  [WAIT] Timeout — continuando.")
                return True
        return False

    def _set_state(self, s: State):
        self._state = s
        self._publish_state()

    def _publish_state(self):
        s = String(); s.data = self._state.name
        self.pub_state.publish(s)
        m = String()
        m.data = json.dumps({
            "state":   self._state.name,
            "pallet":  self._pallet,
            "get":     self._get,
            "logo":    self._logo,
            "empresa": self._empresa,
            "mode":    self._mode,
            "log":     self._log,
        })
        self.pub_mission.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = BotoniMaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._handle_stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()