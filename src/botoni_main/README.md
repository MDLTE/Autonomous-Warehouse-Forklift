# botoni_main — Nodo maestro FSM

Paquete ROS2 que corre en la **PC de Marcelo**.

## Estructura

```
botoni_main/
├── botoni_main/
│   ├── botoni_fsm_node.py   ← FSM principal (ejecutable)
│   ├── useFSM.js            ← Hook React para el dashboard
│   └── FSMPanel.jsx         ← Componente visual del dashboard
├── launch/
│   └── botoni_main.launch.py
├── setup.py
└── package.xml
```

## Instalación

```bash
# En ~/ros2_ws/src/ de la PC de Marcelo:
cp -r botoni_main ~/ros2_ws/src/

cd ~/ros2_ws
colcon build --packages-select botoni_main
source install/setup.bash
```

## Uso

```bash
# Opción 1: launch completo (FSM + rosbridge + web_video_server + dashboard)
ros2 launch botoni_main botoni_main.launch.py

# Opción 2: solo el nodo FSM (si el resto ya corre)
ros2 run botoni_main botoni_fsm
```

## Prerequisitos antes de correr el launch

| Máquina | Qué debe estar corriendo |
|---------|--------------------------|
| **Jetson** | `base.launch.py` (cámara, lidar, hackerboard) |
| **Jetson** | `goto_altura_v8.py` (suscribe /fork_cmd, publica /fork_done) |
| **Marcelo** | `slam_node`, `nav_node_pb.py` (suscribe /nav_goal, publica /nav_status) |
| **Marcelo** | `aruco_detector.py`, `poseKalman.py`, `alignment_node.py` |
| **Diego**   | `voice_node.py` (:9091), `logo_detector.py` |

## Topics que el FSM usa

### Publica
| Topic | Tipo | Descripción |
|-------|------|-------------|
| `/fsm_state` | String | Estado actual (nombre del enum) |
| `/fsm_mission` | String | JSON completo para el dashboard |
| `/cmd_vel` | Twist | Cero en STOP/PAUSE, dead reckoning en move |
| `/fork_cmd` | String | Nombre de la altura destino |
| `/nav_goal` | String | JSON `{waypoint: "waypoint_G_Loading_A"}` |
| `/alignment_start` | String | JSON `{role, type, target_company}` |

### Suscribe
| Topic | Tipo | Descripción |
|-------|------|-------------|
| `/fork_done` | Bool | Confirmación de altura alcanzada |
| `/fork_height` | Float32 | Altura actual en mm |
| `/nav_status` | String | `"arrived"` o `"failed"` |
| `/alignment_done` | String | `"success"` o `"failed"` |
| `/aruco/detections` | String | JSON con ArUcos detectados |
| `/qr/detections` | String | JSON con QRs detectados |
| `/voice_cmd` | String | Comandos de voz |
| `/ui_cmd` | String | JSON desde el dashboard |

## Comandos de voz soportados

| Voz | Acción |
|-----|--------|
| START / GO / INICIO | Inicia o reanuda misión |
| STOP / ALTO / PARA | Para todo, fork a punto_bajo |
| PAUSE / PAUSA | Pausa, recuerda estado |
| LOADING / CARGA | Selecciona misión LOADING→TRUCK |
| RACK / ESTANTE | Selecciona misión RACK→TRUCK |

## Comandos UI (/ui_cmd JSON)

```json
{"action": "START"}
{"action": "STOP"}
{"action": "PAUSE"}
{"action": "SELECT_MISSION", "payload": {"mission": "loading_truck"}}
{"action": "SELECT_MISSION", "payload": {"mission": "rack_truck"}}
{"action": "SELECT_COMPANY", "payload": {"company": "EMEZON"}}
```

## Integración del FSMPanel en el dashboard

```jsx
// En App.jsx, agregar:
import FSMPanel from './FSMPanel';

// Dentro del render (donde ya tienes el `ros` del useRos hook):
<FSMPanel ros={ros} />
```

Mover `useFSM.js` y `FSMPanel.jsx` a `src/` del proyecto Vite.

## Flujo STOP / PAUSE / START

```
STOP:
  1. Publica Twist(0,0) → motores paran inmediatamente
  2. Publica /fork_cmd = "punto_bajo"
  3. Espera /fork_done (máx 10s)
  4. Limpia contexto de misión
  5. Estado → IDLE

PAUSE:
  1. Guarda estado actual como snapshot
  2. Publica Twist(0,0)
  3. Estado → PAUSED
  (el fork queda donde está)

START (desde PAUSED):
  1. Restaura estado del snapshot
  2. Reinicia FSM thread desde ese estado
  (continúa como si no hubiera pasado nada)

START (desde IDLE):
  1. Verifica que haya misión seleccionada
  2. Estado → SEARCH_LOADING o SEARCH_RACK
  3. Inicia FSM thread
```
