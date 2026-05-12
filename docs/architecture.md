# ROS 2 Architecture

## Node Graph

```
┌──────────────┐     /VelocityEncR      ┌──────────────┐
│              │ ──────────────────────► │              │
│  micro_ros   │     /VelocityEncL      │  goto_point  │
│    agent     │ ──────────────────────► │              │
│              │                         │  (controller │
│  (MCU ↔ ROS) │     /cmd_vel           │   + Dead     │
│              │ ◄────────────────────── │   Reckoning) │
└──────────────┘                         └──────┬───────┘
       │                                        │
       │  /VelocityEncR                         │ reads /odom
       │  /VelocityEncL                         │ (when EKF active)
       ▼                                        │
┌──────────────┐                         ┌──────┴───────┐
│              │     /odom               │              │
│   kalman     │ ──────────────────────► │  (used by    │
│              │                         │  goto_point) │
│  (EKF node)  │                         └──────────────┘
│              │
└──────┬───────┘
       │ subscribes
       ▼
┌──────────────┐
│   ArUco      │  /marker_publisher/markers
│  Detection   │
│   Node       │
└──────────────┘
```

## Operating Modes

### Mode 1: Dead Reckoning Only

```
Encoders → goto_point (internal DR) → /cmd_vel → robot
```

Set `use_ekf = False` in `goto_point.py`. The node computes Dead Reckoning internally from encoder topics and does not subscribe to `/odom`.

### Mode 2: Full EKF

```
Encoders ──────────────────► kalman (predict) ──► /odom
                                ▲                    │
ArUco Detection ► markers ──────┘  (update)          ▼
                                              goto_point → /cmd_vel
```

Set `use_ekf = True` in `goto_point.py`. The node reads pose from `/odom` published by the `kalman` node instead of computing its own Dead Reckoning.

## Topic Details

### `/VelocityEncR` and `/VelocityEncL` (Float32)

Published by the MCU via micro-ROS at ~50 Hz. Values are angular velocities in rad/s.

### `/cmd_vel` (geometry_msgs/Twist)

Published by `goto_point` at 20 Hz. Only `linear.x` (v) and `angular.z` (ω) are used.

### `/odom` (nav_msgs/Odometry)

Published by `kalman` node. Contains:
- `pose.pose.position.x/y` — estimated position
- `pose.pose.orientation` — quaternion encoding θ
- `pose.covariance` — 6x6 covariance matrix (only 3x3 upper-left is meaningful)

### `/marker_publisher/markers` (Custom)

Published by the ArUco detection node. Contains detected marker IDs and their poses relative to the camera.

## Launch Order

1. `micro_ros_agent` — must be first, bridges MCU to ROS 2
2. `kalman` — needs encoder topics, publishes `/odom`
3. `goto_point` — needs `/odom` (EKF mode) or just encoders (DR mode)

## Timer Rates

| Timer | Node | Rate | Purpose |
|---|---|---|---|
| Dead Reckoning | `goto_point` | 50 Hz (0.02s) | Integrate encoder velocities |
| Control loop | `goto_point` | 20 Hz (0.05s) | Compute and publish /cmd_vel |
| EKF predict | `kalman` | 50 Hz (0.02s) | Propagate state and covariance |
| ArUco update | `kalman` | On detection | Correct state when marker seen |
