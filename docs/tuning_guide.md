# Tuning Guide

## EKF Parameters

### Process Noise `Q`

`Q` represents how much you **distrust** the encoder-based prediction per timestep.

```python
Q = np.diag([σ_x², σ_y², σ_θ²])
```

| Scenario | Values | Why |
|---|---|---|
| Good encoders, flat floor | `diag([0.01², 0.01², 0.005²])` | Low noise, trust the model |
| Rough terrain, wheel slip | `diag([0.05², 0.05², 0.03²])` | High noise, rely more on ArUco |
| **Recommended start** | `diag([0.02², 0.02², 0.01²])` | Balanced |

**Rule of thumb:** If the EKF is too slow to correct after an ArUco detection, increase `Q`. If it's jumpy between detections, decrease `Q`.

### Measurement Noise `R`

`R` represents how much you **distrust** the ArUco measurement.

```python
R = np.diag([σ_x², σ_y², σ_θ²])
```

| Scenario | Values | Why |
|---|---|---|
| Close range, good calibration | `diag([0.02², 0.02², 0.01²])` | Precise measurements |
| Far range, poor lighting | `diag([0.10², 0.10², 0.05²])` | Noisy measurements |
| **Recommended start** | `diag([0.05², 0.05², 0.02²])` | Balanced |

**Rule of thumb:** If ArUco corrections make the estimate **worse** (jump away from truth), increase `R` (trust ArUco less) or check the marker map calibration.

### Common Issues

**EKF correction makes error worse:**
- The ArUco map positions are wrong → recalibrate marker positions
- `R` is too small → the filter over-trusts a noisy measurement
- The measurement `z` is computed incorrectly (e.g., using marker position instead of robot position)

**EKF doesn't correct enough:**
- `Q` is too small → the filter thinks Dead Reckoning is perfect
- `R` is too large → the filter ignores the measurement

**Covariance explodes:**
- Check Jacobian `F` computation — common mistake: using `x̂⁻` instead of `x̂` for θ
- Ensure angle wrapping in the innovation

## Controller Parameters

### Displaced Point Distance `h`

| Value | Behavior |
|---|---|
| 0.01 m | Very aggressive, high ω, may oscillate |
| **0.05 m** | Good balance for PuzzleBot |
| 0.10 m | Smooth but wide turns |
| 0.20 m | Very smooth, may miss tight waypoints |

### Gain `k`

| Value | Behavior |
|---|---|
| 0.5 | Slow convergence, very smooth |
| **1.0** | Good starting point |
| 2.0 | Fast but may overshoot |
| 5.0 | Aggressive, likely oscillation |

### Velocity Limits

```python
v_max = 0.20  # m/s — don't exceed, encoders lose accuracy
w_max = 1.50  # rad/s — PuzzleBot motor limit
```

### Waypoint Tolerances

```python
D_min     = 0.07   # 7 cm position tolerance
THETA_MIN = 0.05   # ~3° angle tolerance
```

If the robot oscillates around a waypoint, increase `D_min`. If it doesn't orient correctly, decrease `THETA_MIN`.

## Diagnostic Checklist

1. **Straight line test:** Robot goes 0.5 m forward. Does it deviate? If yes → wheel imbalance (solved with new motors)
2. **Square test:** Waypoints `(0.5,0), (0.5,0.5), (0,0.5), (0,0)`. Does it close the square? Error at return = accumulated drift
3. **ArUco detection test:** `ros2 topic echo /marker_publisher/markers` — do IDs and positions make sense?
4. **EKF convergence:** Plot error over time. Should decrease after each ArUco detection
