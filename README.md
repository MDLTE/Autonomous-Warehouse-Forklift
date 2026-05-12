# PuzzleBot — EKF Localization & Trajectory Control

Autonomous navigation system for a differential-drive mobile robot (PuzzleBot) using an **Extended Kalman Filter (EKF)** for localization and a **displaced-point trajectory controller** for path following. Built on **ROS 2 Humble**.

> **Course:** TE3003B — Integration of Robotics and Intelligent Systems  
> **Institution:** ITESM Campus Monterrey  
> **Semester:** Spring 2026

---

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    PuzzleBot Architecture                │
│                                                         │
│  Encoders ──► Dead Reckoning ──┐                        │
│                                ├──► EKF ──► Pose (x,y,θ)│
│  Camera ──► ArUco Detection ──┘         │               │
│                                         ▼               │
│                              Trajectory Controller      │
│                             (Displaced Point)           │
│                                         │               │
│                                         ▼               │
│                                    /cmd_vel             │
│                                         │               │
│                                         ▼               │
│                                  Motor Drivers          │
└─────────────────────────────────────────────────────────┘
```

## Key Features

- **EKF Localization** — Fuses encoder odometry (prediction) with ArUco marker detections (correction) for real-time pose estimation `(x, y, θ)`
- **Dead Reckoning** — Integrates wheel encoder velocities using the differential-drive kinematic model as the EKF prediction step
- **ArUco Marker Correction** — Uses known marker positions in the environment to correct accumulated drift via the EKF update step
- **Displaced-Point Controller** — Resolves the differential-drive singularity by controlling a virtual point `h` meters ahead of the robot center, producing an invertible input matrix
- **Waypoint Navigation** — Sequentially navigates through `(x, y, θ)` waypoints with configurable tolerances
- **ROS 2 Architecture** — Modular nodes communicating via standard topics (`/cmd_vel`, `/odom`, `/VelocityEncR`, `/VelocityEncL`)

---

## Repository Structure

```
puzzlebot-ekf/
├── README.md
├── docs/
│   ├── theory.md              # Mathematical foundations (EKF, kinematics, controller)
│   ├── tuning_guide.md        # How to tune Q, R, controller gains
│   └── architecture.md        # ROS2 node graph and topic map
├── src/
│   └── puzzlebot_ros/
│       ├── package.xml
│       ├── setup.py
│       └── puzzlebot_ros/
│           ├── __init__.py
│           ├── goto_point.py  # Waypoint controller with Dead Reckoning
│           ├── kalman.py      # EKF node (predict + ArUco update)
│           └── my_math.py     # Utility functions (wrap_to_pi, quaternion conversion)
├── simulation/
│   └── ekf_sim.py             # Offline EKF simulation with visualization
├── config/
│   └── aruco_map.yaml         # Known ArUco marker positions
├── scripts/
│   └── test_straight.py       # Diagnostic: encoder balance test
└── media/                     # Screenshots, plots, videos
```

---

## Hardware

| Component | Details |
|---|---|
| **Robot** | PuzzleBot (Manchester Robotics / ITESM) |
| **Drive** | Differential drive, 2 DC motors with encoders |
| **Computer** | Jetson Nano (onboard) + laptop via SSH |
| **Camera** | USB camera for ArUco detection |
| **Wheel radius** | 0.05 m |
| **Wheel base** | 0.18 m |
| **Firmware** | micro-ROS agent bridging MCU ↔ ROS 2 |

---

## Quick Start

### Prerequisites
- Ubuntu 22.04 + ROS 2 Humble
- PuzzleBot with micro-ROS agent running
- Python 3.10, NumPy, OpenCV

### Build
```bash
cd ~/ros2_ws
cp -r <this_repo>/src/puzzlebot_ros src/
colcon build --packages-select puzzlebot_ros
source install/setup.bash
```

### Run (Dead Reckoning only)
```bash
# Terminal 1 — micro-ROS agent
ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0

# Terminal 2 — Waypoint controller
ros2 run puzzlebot_ros goto_point
```

### Run (Full EKF with ArUco)
```bash
# Terminal 1 — micro-ROS agent
ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0

# Terminal 2 — EKF node (publishes /odom)
ros2 run puzzlebot_ros kalman

# Terminal 3 — Waypoint controller (reads /odom)
ros2 run puzzlebot_ros goto_point
```

---

## Theory Summary

### Extended Kalman Filter

**State:** `x = [x, y, θ]ᵀ`

**Prediction (every timestep):**
```
x̂⁻ = f(x̂, u)     where u = [v, ω] from encoders
P⁻  = F · P · Fᵀ + Q
```

**Jacobian F:**
```
F = ┌ 1   0   -v·sin(θ)·dt ┐
    │ 0   1    v·cos(θ)·dt │
    └ 0   0    1            ┘
```

**Update (when ArUco detected):**
```
K = P⁻ · Hᵀ · (H · P⁻ · Hᵀ + R)⁻¹
x̂ = x̂⁻ + K · (z - H · x̂⁻)
P = (I - K · H) · P⁻
```

Where `H = I₃` (ArUco measures `[x, y, θ]` directly).

### Displaced-Point Controller

The standard differential-drive kinematic model has a singular input matrix. By controlling a point displaced `h` meters ahead:

```
[ẋₕ]   [cos(θ)  -h·sin(θ)] [v]
[ẏₕ] = [sin(θ)   h·cos(θ)] [ω]
```

This matrix has `det = h ≠ 0`, making it invertible. The control law becomes:

```
[v]   [cos(θ)  -h·sin(θ)]⁻¹   [k·e₁]
[ω] = [sin(θ)   h·cos(θ)]    · [k·e₂]
```

Where `e₁, e₂` are errors in the displaced-point frame.

---

## Configuration

### Filter Parameters (`kalman.py`)

| Parameter | Description | Default |
|---|---|---|
| `Q` | Process noise covariance (encoder trust) | `diag([0.02², 0.02², 0.01²])` |
| `R` | Measurement noise covariance (ArUco trust) | `diag([0.05², 0.05², 0.02²])` |
| `P₀` | Initial covariance | `diag([0.01, 0.01, 0.01])` |

### Controller Parameters (`goto_point.py`)

| Parameter | Description | Default |
|---|---|---|
| `h` | Displaced point distance (m) | 0.05 |
| `k` | Proportional gain | 1.0 |
| `v_max` | Max linear velocity (m/s) | 0.20 |
| `w_max` | Max angular velocity (rad/s) | 1.5 |
| `D_min` | Position tolerance (m) | 0.07 |
| `THETA_MIN` | Angle tolerance (rad) | 0.05 (~3°) |

### ArUco Map (`config/aruco_map.yaml`)

```yaml
markers:
  - id: 0
    position: [1.0, 0.0, 0.0]  # x, y, θ in world frame
  - id: 1
    position: [0.0, -1.0, 1.5708]
```

---

## ROS 2 Topics

| Topic | Type | Publisher | Subscriber |
|---|---|---|---|
| `/VelocityEncR` | `Float32` | MCU | `goto_point`, `kalman` |
| `/VelocityEncL` | `Float32` | MCU | `goto_point`, `kalman` |
| `/cmd_vel` | `Twist` | `goto_point` | MCU |
| `/odom` | `Odometry` | `kalman` | `goto_point` |
| `/marker_publisher/markers` | Custom | ArUco node | `kalman` |

---

## Team

- **Marcelo** — EKF implementation, trajectory controller, system integration
- **Diego** — ArUco marker calibration and detection

---

## Acknowledgments

- Manchester Robotics — PuzzleBot platform and firmware
- ITESM Robotics Lab — TE3003B course infrastructure
