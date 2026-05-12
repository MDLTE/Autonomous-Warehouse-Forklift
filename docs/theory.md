# Mathematical Foundations

## 1. Differential-Drive Kinematics

The PuzzleBot is a two-wheeled differential-drive robot. Given left and right wheel angular velocities `ωL` and `ωR`:

```
v = r/2 · (ωR + ωL)        # linear velocity
ω = r/L · (ωR - ωL)        # angular velocity
```

Where `r = 0.05 m` (wheel radius) and `L = 0.18 m` (wheel base).

The continuous-time kinematic model:

```
ẋ = v · cos(θ)
ẏ = v · sin(θ)
θ̇ = ω
```

Discretized with Euler integration (dt = 0.02s for Dead Reckoning, 0.05s for control):

```
x[k+1] = x[k] + v · cos(θ[k]) · dt
y[k+1] = y[k] + v · sin(θ[k]) · dt
θ[k+1] = θ[k] + ω · dt
```

## 2. Dead Reckoning

Dead Reckoning integrates encoder measurements to estimate pose. The fundamental problem is **error accumulation**: small errors in `θ` cause `cos(θ)` and `sin(θ)` to project motion in the wrong direction, and this error compounds at every timestep.

Sources of error:
- Encoder quantization noise
- Wheel slip
- Unequal wheel diameters
- Floor irregularities

For a straight-line trajectory, a 2° error in θ after 1 meter produces ~3.5 cm lateral deviation. After curves, the error grows much faster because angular errors compound.

## 3. Extended Kalman Filter (EKF)

The EKF addresses drift by fusing Dead Reckoning (prediction) with absolute position measurements from ArUco markers (correction).

### State Vector

```
x = [x, y, θ]ᵀ
```

### Prediction Step

Uses the nonlinear kinematic model with inputs `u = [v, ω]`:

```
x̂⁻[k] = f(x̂[k-1], u[k])

f(x, u) = [ x + v·cos(θ)·dt ]
           [ y + v·sin(θ)·dt ]
           [ θ + ω·dt         ]
```

The Jacobian of `f` with respect to the state:

```
F = ∂f/∂x = ┌ 1   0   -v·sin(θ)·dt ┐
             │ 0   1    v·cos(θ)·dt │
             └ 0   0    1            ┘
```

Covariance propagation:

```
P⁻ = F · P · Fᵀ + Q
```

Where `Q` is the process noise covariance, representing how much we distrust the encoder-based prediction.

### Update Step (ArUco Correction)

When an ArUco marker is detected, we get a direct measurement of `[x, y, θ]` in the world frame. Since the measurement model is linear:

```
z = H · x + noise
H = I₃  (identity matrix)
```

Innovation (difference between measurement and prediction):

```
ỹ = z - H · x̂⁻
```

**Important:** The angle component of the innovation must be wrapped to `[-π, π]`.

Kalman Gain:

```
S = H · P⁻ · Hᵀ + R
K = P⁻ · Hᵀ · S⁻¹
```

State and covariance update:

```
x̂ = x̂⁻ + K · ỹ
P = (I - K · H) · P⁻
```

### Interpretation of the Kalman Gain

- If `R → 0` (perfect sensor): `K → H⁻¹`, the filter trusts the measurement completely
- If `P⁻ → 0` (perfect model): `K → 0`, the filter ignores the measurement
- In practice, `K` balances both sources based on their relative uncertainties

### Why EKF and not KF?

The process model `f(x, u)` contains `cos(θ)` and `sin(θ)` — nonlinear functions of the state. The standard Kalman Filter assumes linear models. The EKF linearizes via the Jacobian `F` at each timestep, making it a local linear approximation that works well when the state estimate is close to the true state.

## 4. Displaced-Point Controller

### The Singularity Problem

The direct kinematic model maps `[v, ω]` to `[ẋ, ẏ]`:

```
B = ┌ cos(θ)   0 ┐
    └ sin(θ)   0 ┘
```

`det(B) = 0` — the matrix is singular. You cannot independently control `x` and `y` because both depend on `θ`. The system is **underactuated** at this operating point.

### The Solution: Displaced Point

Instead of controlling the robot center `(x, y)`, we control a virtual point `(xₕ, yₕ)` located `h` meters ahead:

```
xₕ = x + h·cos(θ)
yₕ = y + h·sin(θ)
```

Taking the time derivative:

```
[ẋₕ]   [cos(θ)  -h·sin(θ)] [v]
[ẏₕ] = [sin(θ)   h·cos(θ)] [ω]
         \_________________/
                 B_h
```

Now `det(B_h) = h·cos²(θ) + h·sin²(θ) = h ≠ 0` (as long as `h > 0`).

### Control Law

Given the desired displaced-point velocity `[ẋₕ_d, ẏₕ_d]`:

```
[v]         [k · e₁]
[ω] = B_h⁻¹ [k · e₂]
```

Where:
- `e₁ = x_goal - xₕ` (error in x)
- `e₂ = y_goal - yₕ` (error in y)
- `k` is the proportional gain

This is a **proportional controller** — no integral or derivative needed. The decoupling provided by `B_h⁻¹` ensures that errors `e₁` and `e₂` decay exponentially. The system is mathematically guaranteed to converge for any `h > 0` and `k > 0`.

### Parameter `h` Trade-offs

- **Small `h` (0.01–0.05 m):** More agile turns, but higher angular velocities and potential oscillation
- **Large `h` (0.1–0.2 m):** Smoother trajectories, but wider turns and slower convergence at waypoints
- **Recommended:** Start at `h = 0.05 m` and increase if oscillation occurs

## 5. ArUco Marker Detection

ArUco markers are fiducial markers with unique binary patterns. Using OpenCV's `cv2.aruco` module:

1. Camera captures frame
2. `detectMarkers()` finds marker corners in the image
3. `estimatePoseSingleMarkers()` computes the marker's pose relative to the camera using the camera's intrinsic parameters
4. Known marker positions in the world frame + camera-to-marker transform → robot pose measurement `z = [x, y, θ]`

The measurement noise `R` depends on:
- Camera resolution and calibration quality
- Distance to marker (further = noisier)
- Viewing angle (oblique = noisier)
- Lighting conditions

## References

- Welch, G. & Bishop, G. — "An Introduction to the Kalman Filter" (UNC-Chapel Hill, TR 95-041)
- Särkkä, S. & Svensson, L. — "Bayesian Filtering and Smoothing" (Cambridge, 2nd ed., 2023)
- Siciliano, B. et al. — "Robotics: Modelling, Planning and Control" (Springer)
