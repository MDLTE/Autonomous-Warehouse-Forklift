#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Puente de voz -> PuzzleBot.

Se suscribe a /voice_cmd (std_msgs/String) — el topico donde la UI publica
la palabra reconocida — y la traduce a comandos de ROS2:

    forward / back   -> /cmd_vel        (Twist, linear.x)
    left / right     -> /cmd_vel        (Twist, angular.z)
    turn             -> /cmd_vel        (giro de 180 grados en el sitio)
    pick / drop      -> /forklift_cmd   (String, "up 2500" / "down 2500")
    stop             -> detiene las ruedas
    start / home     -> placeholders de mision (por implementar)

MONTACARGAS — por que estos comandos:
  En el resto del sistema el montacargas se comanda por dos interfaces:
    (1) /forklift_cmd  (std_msgs/String)  con "up <N>" / "down <N>"  -> lo usa
        door_align para soltar/levantar (N = objetivo que entiende el FPGA,
        p.ej. cuentas de encoder; door_align usa 2500).
    (2) /forklift/insert (std_msgs/Bool) + /forklift/done (Bool)  -> lo usa
        center_qr para disparar la SECUENCIA COMPLETA de insercion+levantado.
  Para un comando de voz simple usamos la interfaz directa (1):
    pick  = subir horquilla  -> "up 2500"
    drop  = bajar horquilla  -> "down 2500"
  Si prefieres que "pick" lance la secuencia autonoma completa del FPGA, mira
  el bloque comentado de la interfaz Bool mas abajo (USAR_INSERT_BOOL).

VERIFICA con quien programo el FPGA:
  - Que significa el "2500" (posicion absoluta, cuentas, duracion) y si
    up/down mueven hacia donde esperas; la convencion podria estar invertida.

Correr:
    source /opt/ros/<distro>/setup.bash
    source install/setup.bash
    python3 puzzlebot_voice_bridge.py
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String


# --- Parametros del PuzzleBot (ajusta a tu robot) ---
VELOCIDAD_LINEAL = 0.15     # m/s   para forward / back
VELOCIDAD_ANGULAR = 0.8     # rad/s para left / right / turn
# DURACION_PULSO: cuanto dura un comando de movimiento antes de parar solo.
#   > 0  -> por pulsos (seguro): "forward" avanza ~N s y se detiene.
#   <= 0 -> latcheado: sigue moviendose hasta que llegue "stop".
DURACION_PULSO = 1.5

# --- Giro "turn" (lazo abierto, por tiempo) ---
GIRO_GRADOS = 180.0   # cuantos grados gira "turn"
GIRO_FACTOR = 1.0     # calibracion: subelo si gira de menos, bajalo si gira de mas

# --- Montacargas (interfaz directa que consume el FPGA via door_align) ---
TOPICO_FORK = "/forklift_cmd"     # std_msgs/String, formato "<dir> <N>"
FORK_UP_CMD = "up 2500"           # pick: levanta la horquilla
FORK_DOWN_CMD = "down 2500"       # drop: baja la horquilla

# Si quieres que "pick" dispare la secuencia AUTONOMA del FPGA (la que usa
# center_qr) en vez de un simple "up", pon esto en True. Publica un Bool en
# /forklift/insert y opcionalmente lee /forklift/done.
USAR_INSERT_BOOL = False
TOPICO_FORK_INSERT = "/forklift/insert"   # std_msgs/Bool
TOPICO_FORK_DONE = "/forklift/done"       # std_msgs/Bool

# Topico donde la UI publica la palabra reconocida
TOPICO_VOZ = "/voice_cmd"


class PuzzlebotVoiceBridge(Node):

    def __init__(self):
        super().__init__("puzzlebot_voice_bridge")

        self.lin = VELOCIDAD_LINEAL
        self.ang = VELOCIDAD_ANGULAR
        self.pulso = DURACION_PULSO

        # Publicadores
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_mision = self.create_publisher(String, "/fsm_mission", 10)
        # Montacargas: interfaz directa por String ("up 2500" / "down 2500")
        self.pub_fork = self.create_publisher(String, TOPICO_FORK, 10)

        # (Opcional) interfaz Bool de secuencia completa de insercion
        self.pub_fork_insert = None
        self.fork_done = False
        if USAR_INSERT_BOOL:
            self.pub_fork_insert = self.create_publisher(
                Bool, TOPICO_FORK_INSERT, 10)
            self.create_subscription(
                Bool, TOPICO_FORK_DONE, self._on_fork_done, 10)

        # Suscripcion al topico que ya publica la UI
        self.create_subscription(String, TOPICO_VOZ, self._on_voice, 10)

        # Estado de movimiento
        self._target = Twist()
        self._deadline = None          # instante (s) en que parar; None = parado

        # Publica /cmd_vel a 20 Hz (mantiene contento al watchdog del robot)
        self.create_timer(0.05, self._tick)

        self.get_logger().info(
            f"Bridge de voz escuchando {TOPICO_VOZ}  "
            f"(lin={self.lin} m/s, ang={self.ang} rad/s, pulso={self.pulso} s)"
        )
        self.get_logger().info(
            f"Montacargas -> {TOPICO_FORK}  "
            f"(pick='{FORK_UP_CMD}', drop='{FORK_DOWN_CMD}')"
            + (f" | insert Bool en {TOPICO_FORK_INSERT}"
               if USAR_INSERT_BOOL else "")
        )

    # -- Entrada: palabra recibida desde /voice_cmd --
    def _on_voice(self, msg: String):
        self.handle(msg.data.strip().lower())

    def _on_fork_done(self, msg: Bool):
        self.fork_done = bool(msg.data)
        if self.fork_done:
            self.get_logger().info("montacargas: secuencia terminada (/forklift/done)")

    # -- Publicacion periodica de velocidad --
    def _tick(self):
        comando = self._target

        if self._deadline is not None:
            ahora = self.get_clock().now().nanoseconds / 1e9
            if ahora >= self._deadline:
                self._target = Twist()
                self._deadline = None
                comando = self._target

        self.pub_cmd.publish(comando)

    # -- Helpers de movimiento --
    def _mover(self, lx=0.0, az=0.0, duracion=None):
        """Mueve con (lx, az). Si 'duracion' se pasa, esa duracion manda
        (p.ej. el giro de 180 grados); si no, usa el pulso por defecto."""
        tw = Twist()
        tw.linear.x = float(lx)
        tw.angular.z = float(az)
        self._target = tw

        ahora = self.get_clock().now().nanoseconds / 1e9
        if duracion is not None and duracion > 0:
            self._deadline = ahora + duracion
        elif self.pulso > 0:
            self._deadline = ahora + self.pulso
        else:
            self._deadline = None   # latcheado

    def _girar_grados(self, grados: float):
        """Giro en el sitio en lazo abierto. + = izquierda (antihorario)."""
        if self.ang <= 0:
            self.get_logger().warn("VELOCIDAD_ANGULAR <= 0, no se puede girar")
            return
        radianes = math.radians(abs(grados))
        tiempo = (radianes / self.ang) * GIRO_FACTOR
        sentido = self.ang if grados >= 0 else -self.ang
        self._mover(az=sentido, duracion=tiempo)
        self.get_logger().info(f"giro de {grados:.0f} grados -> {tiempo:.2f} s")

    def _detener_ruedas(self):
        self._target = Twist()
        self._deadline = None
        self.pub_cmd.publish(Twist())

    # -- Publicacion de mision / montacargas --
    def _publicar_mision(self, texto: str):
        msg = String()
        msg.data = texto
        self.pub_mision.publish(msg)

    def _publicar_fork(self, texto: str):
        msg = String()
        msg.data = texto
        self.pub_fork.publish(msg)

    def _insertar_pallet(self):
        """Dispara la secuencia completa de insercion del FPGA (interfaz Bool)."""
        if self.pub_fork_insert is None:
            self.get_logger().warn(
                "USAR_INSERT_BOOL=False; no hay publisher de /forklift/insert")
            return
        self.fork_done = False
        self.pub_fork_insert.publish(Bool(data=True))
        self.get_logger().info(f"montacargas: insert -> {TOPICO_FORK_INSERT}")

    # -- Traduccion palabra -> accion --
    def handle(self, palabra: str):
        # -- Movimiento: ya funcional --------------------------------------
        if palabra == "forward":
            self._mover(lx=self.lin)
        elif palabra == "back":
            self._mover(lx=-self.lin)
        elif palabra == "left":
            self._mover(az=self.ang)        # angular.z + = izquierda
        elif palabra == "right":
            self._mover(az=-self.ang)
        elif palabra == "turn":
            # Giro de 180 grados en el sitio (lazo abierto, por tiempo)
            self._girar_grados(GIRO_GRADOS)

        # -- Montacargas: ya cableado --------------------------------------
        elif palabra == "pick":
            # Levantar pallet = subir la horquilla
            if USAR_INSERT_BOOL:
                self._insertar_pallet()     # secuencia completa del FPGA
            else:
                self._publicar_fork(FORK_UP_CMD)   # "up 2500"
            self.get_logger().info("pick -> levantar horquilla")
        elif palabra == "drop":
            # Soltar pallet = bajar la horquilla
            self._publicar_fork(FORK_DOWN_CMD)     # "down 2500"
            self.get_logger().info("drop -> bajar horquilla")

        # -- stop: detiene las ruedas (seguro) -----------------------------
        elif palabra == "stop":
            self._detener_ruedas()
            # TODO (opcional): tambien avisar a la mision para que se detenga
            # ej.: self._publicar_mision("stop")
            self.get_logger().info("stop -> ruedas detenidas")

        # -- Placeholders de mision (dependen de TU capa de mision) --------
        elif palabra == "start":
            # TODO: iniciar / reanudar la mision.
            # Nota: nav_node_simple no escucha "start"; usa /search_trigger
            # con valores T1 / T2 / D4 / D3 / D2. Mapea segun tu flujo.
            # ej.: self._publicar_mision("start")
            self.get_logger().info("[placeholder] start (sin implementar)")
        elif palabra == "home":
            # TODO: volver a base. nav_node_simple real no tiene trigger HOME;
            # define el destino de regreso en tu capa de mision.
            # ej.: self._publicar_mision("home")
            self.get_logger().info("[placeholder] home (sin implementar)")

        else:
            self.get_logger().warn(f"palabra sin accion: {palabra}")
            return

        self.get_logger().info(f"comando recibido: {palabra}")


def main(args=None):
    rclpy.init(args=args)
    nodo = PuzzlebotVoiceBridge()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()