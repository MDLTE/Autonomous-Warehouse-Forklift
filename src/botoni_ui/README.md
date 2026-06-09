# botoni_ui — Interfaz web de Botoni (package ROS2)

Package de ROS2 que sirve el dashboard de Botoni y levanta los puentes
(rosbridge + web_video_server) con un solo `ros2 launch`.

```
Jetson (10.42.0.1)
└── camera_publisher → /video_source/compressed

Esta compu (Marcelo) — corre el package
├── botoni_web_server :5173  ← sirve la página
├── rosbridge         :9090  ← topics
└── web_video_server  :8080  ← cámara

Maje / Pato
└── navegador → http://IP_DE_MARCELO:5173
```

═══════════════════════════════════════════════════════════════════

## INSTALACIÓN (una sola vez)

### 1. Dependencias del sistema
```bash
sudo apt install -y ros-humble-rosbridge-suite ros-humble-web-video-server
```

### 2. Copiar el package al workspace
```bash
cp -r botoni_ui ~/ros2_ws/src/
```

### 3. Compilar
```bash
cd ~/ros2_ws
colcon build --packages-select botoni_ui
source install/setup.bash
```

═══════════════════════════════════════════════════════════════════

## CORRER

### 1. La cámara debe estar publicando (en la Jetson)
```bash
# en la Jetson
ros2 topic list | grep video
#   → /video_source/compressed
```

### 2. Levantar la interfaz (en la compu de Marcelo)
```bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch botoni_ui botoni_ui.launch.py
```

Verás algo como:
```
[botoni_web_server]: Interfaz Botoni sirviendo en http://0.0.0.0:5173
```

### 3. Conectarse
Desde cualquier navegador en la misma red:
```
http://IP_DE_MARCELO:5173
```

Sacar la IP:
```bash
ip addr show wlan0     # busca inet 192.168.X.X
```

═══════════════════════════════════════════════════════════════════

## IMPORTANTE — ¿dónde vive la cámara?

La interfaz pide la cámara a la **misma IP** por la que entraste a la web.

- Si corres el package en la **compu de Marcelo** y la cámara está en la
  **Jetson**, hay 2 opciones:

  **A)** Correr `web_video_server` en la Jetson (no en este launch), y que
       la interfaz apunte a la Jetson. Para eso, edita en `web/index.html`
       la constante `HOST` para forzar la IP de la Jetson:
       ```js
       const HOST = "10.42.0.1";
       ```

  **B)** Más simple: corre TODO el launch en la Jetson. Pesa más pero
       todo queda en una sola IP y la interfaz se conecta sola.

- Si corres el package en la **misma máquina** donde están la cámara y los
  topics, no cambias nada (HOST se auto-detecta).

═══════════════════════════════════════════════════════════════════

## Topics que usa la interfaz

PUBLICA:
- `/cmd_vel` (geometry_msgs/Twist) — joystick
- `/fork_cmd` (std_msgs/Float32) — altura objetivo de forks
- `/target_select` (std_msgs/String) — empresa/destino elegido
- `/fsm_go` (std_msgs/Bool), `/fsm_start` (std_msgs/String) — control FSM

SUSCRIBE:
- `/fsm_state` (std_msgs/String) — estado actual
- `/fork_height` (std_msgs/Float32) — altura real de forks
- `/fork_done` (std_msgs/Bool) — fork llegó a destino
- `/qr/detections` (std_msgs/String) — lecturas de QR
- `/video_source` + type=ros_compressed — cámara

Los topics que aún no existan simplemente no muestran datos; la interfaz
no truena.

═══════════════════════════════════════════════════════════════════

## Estructura del package

```
botoni_ui/
├── package.xml
├── setup.py / setup.cfg
├── resource/botoni_ui
├── botoni_ui/
│   ├── __init__.py
│   └── web_server.py        nodo que sirve la web
├── launch/
│   └── botoni_ui.launch.py  levanta web + rosbridge + video
└── web/
    ├── index.html           el dashboard
    └── roslib.min.js        librería ROS para el navegador
```

## Troubleshooting

- **No carga la página** → ¿corrió el launch sin error? ¿puerto 5173 libre?
- **Página carga pero "sin conexión"** → rosbridge no levantó o IP mal. Revisa el log del launch.
- **Cámara negra** → web_video_server no ve `/video_source/compressed`, o la cámara está en otra IP (ver sección "¿dónde vive la cámara?").
- **Firewall** → `sudo ufw allow 5173 && sudo ufw allow 9090 && sudo ufw allow 8080`
