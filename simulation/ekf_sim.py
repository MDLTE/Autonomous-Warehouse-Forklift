"""
ekf_sim.py — Offline EKF Simulation for PuzzleBot
===================================================
Simulates differential-drive robot with:
  - Dead Reckoning (encoder integration only)
  - EKF (encoder prediction + ArUco correction)

Generates comparison plots: trajectory, error over time, covariance ellipses.

Usage:
  python3 simulation/ekf_sim.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────
DT = 0.05  # timestep (s) → 20 Hz

# Filter noise matrices
Q = np.diag([0.02**2, 0.02**2, 0.01**2])     # process noise
R = np.diag([0.03**2, 0.03**2, 0.015**2])     # measurement noise
H = np.eye(3)                                   # observation matrix

# Encoder noise (for simulating the real robot)
ENCODER_NOISE_V = 0.015   # std dev on linear velocity (m/s)
ENCODER_NOISE_W = 0.005   # std dev on angular velocity (rad/s)

# ArUco markers: known positions in world frame
ARUCO_MARKERS = [
    {"id": 0, "pos": np.array([1.5, 0.2, 0.0])},
    {"id": 1, "pos": np.array([0.5, 1.5, np.pi / 2])},
    {"id": 2, "pos": np.array([-0.3, 0.8, np.pi])},
]

DETECTION_RADIUS = 0.8     # meters — max distance to detect a marker
DETECTION_COOLDOWN = 20    # timesteps between detections of same marker


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def normalize(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def get_ellipse(mean, cov, n_std=2.0):
    """Compute 2σ covariance ellipse parameters."""
    vals, vecs = np.linalg.eigh(cov[:2, :2])
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(np.abs(vals))
    return w, h, angle


# ─────────────────────────────────────────────
# EKF CLASS (same as in kalman.py)
# ─────────────────────────────────────────────
class EKF:
    def __init__(self, x0=0, y0=0, theta0=0):
        self.x = np.array([x0, y0, theta0], dtype=float)
        self.P = np.diag([0.01, 0.01, 0.01])

    def predict(self, v, omega, dt=DT):
        x, y, th = self.x
        self.x = np.array([
            x + v * np.cos(th) * dt,
            y + v * np.sin(th) * dt,
            normalize(th + omega * dt),
        ])
        F = np.array([
            [1, 0, -v * np.sin(th) * dt],
            [0, 1,  v * np.cos(th) * dt],
            [0, 0,  1],
        ])
        self.P = F @ self.P @ F.T + Q

    def update(self, z):
        inn = z - H @ self.x
        inn[2] = normalize(inn[2])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ inn
        self.x[2] = normalize(self.x[2])
        self.P = (np.eye(3) - K @ H) @ self.P
        return K


# ─────────────────────────────────────────────
# SIMULATED ROBOT
# ─────────────────────────────────────────────
class RealRobot:
    """Simulates the real robot with encoder noise."""
    def __init__(self, x0=0, y0=0, theta0=0):
        self.pos = np.array([x0, y0, theta0], dtype=float)

    def step(self, v, omega, dt=DT):
        vn = v + np.random.normal(0, ENCODER_NOISE_V)
        wn = omega + np.random.normal(0, ENCODER_NOISE_W)
        x, y, th = self.pos
        self.pos = np.array([
            x + vn * np.cos(th) * dt,
            y + vn * np.sin(th) * dt,
            normalize(th + wn * dt),
        ])
        return self.pos.copy()


# ─────────────────────────────────────────────
# COMMAND GENERATOR
# ─────────────────────────────────────────────
def get_commands(t):
    """Generate velocity commands for a test trajectory."""
    if   t < 3.0:  return 0.30,  0.0       # straight
    elif t < 5.0:  return 0.25, -0.50      # right curve
    elif t < 8.0:  return 0.30,  0.0       # straight
    elif t < 10.0: return 0.25,  0.40      # left curve
    elif t < 13.0: return 0.30,  0.0       # straight
    elif t < 15.0: return 0.20, -0.60      # right curve
    elif t < 18.0: return 0.25,  0.0       # straight
    else:          return 0.0,   0.0       # stop


# ─────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────
def simulate():
    np.random.seed(42)

    T_TOTAL = 20.0
    N_STEPS = int(T_TOTAL / DT)

    robot = RealRobot(0, 0, 0)
    ekf = EKF(0, 0, 0)
    dr = EKF(0, 0, 0)  # Dead Reckoning = predict only, never update

    hist_real = [np.array([0.0, 0.0, 0.0])]
    hist_ekf  = [np.array([0.0, 0.0, 0.0])]
    hist_dr   = [np.array([0.0, 0.0, 0.0])]
    hist_P    = [ekf.P.copy()]
    corrections = []
    last_seen = {m["id"]: -999 for m in ARUCO_MARKERS}

    for i in range(N_STEPS):
        t = i * DT
        v, w = get_commands(t)

        # 1. Move real robot
        real = robot.step(v, w)

        # 2. Predict (both filters use same commands)
        ekf.predict(v, w)
        dr.predict(v, w)

        # 3. Check for ArUco detections (EKF only)
        for marker in ARUCO_MARKERS:
            dist = np.linalg.norm(real[:2] - marker["pos"][:2])
            if dist < DETECTION_RADIUS and (i - last_seen[marker["id"]]) > DETECTION_COOLDOWN:
                # Simulate noisy measurement of robot pose
                noise = np.random.multivariate_normal([0, 0, 0], R)
                z = real + noise  # measurement = true pose + noise
                ekf.update(z)
                corrections.append({"step": i, "pos": ekf.x[:2].copy(), "id": marker["id"]})
                last_seen[marker["id"]] = i
                break  # max one correction per timestep

        hist_real.append(real.copy())
        hist_ekf.append(ekf.x.copy())
        hist_dr.append(dr.x.copy())
        hist_P.append(ekf.P.copy())

    return (np.array(hist_real), np.array(hist_ekf),
            np.array(hist_dr), hist_P, corrections)


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────
def plot(hist_real, hist_ekf, hist_dr, hist_P, corrections):
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#FAFAFA')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ── Plot 1: Trajectories ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#F8F8F8')

    ax1.plot(hist_real[:, 0], hist_real[:, 1],
             color='#2C2C2A', lw=2.5, label='Real trajectory', zorder=4)
    ax1.plot(hist_dr[:, 0], hist_dr[:, 1],
             color='#D85A30', lw=1.8, ls='--', alpha=0.85,
             label='Dead Reckoning only', zorder=3)
    ax1.plot(hist_ekf[:, 0], hist_ekf[:, 1],
             color='#185FA5', lw=2.2, label='EKF (encoders + ArUco)', zorder=4)

    # Covariance ellipses every 50 steps
    for i in range(0, len(hist_P), 50):
        w, h, ang = get_ellipse(hist_ekf[i], hist_P[i])
        if w < 2 and h < 2:
            ell = Ellipse(xy=hist_ekf[i, :2], width=w, height=h, angle=ang,
                          edgecolor='#378ADD', facecolor='#B5D4F4',
                          alpha=0.3, lw=1, zorder=2)
            ax1.add_patch(ell)

    # ArUco markers
    for marker in ARUCO_MARKERS:
        ax1.plot(marker["pos"][0], marker["pos"][1], 's',
                 color='#3B6D11', ms=14, zorder=5)
        circle = plt.Circle(marker["pos"][:2], DETECTION_RADIUS,
                            color='#3B6D11', fill=False, ls='--', alpha=0.3, lw=1)
        ax1.add_patch(circle)
        ax1.annotate(f'  Aruco {marker["id"]}', xy=marker["pos"][:2],
                     fontsize=9, color='#3B6D11', va='center')

    # Correction stars
    for c in corrections:
        ax1.plot(c["pos"][0], c["pos"][1], '*',
                 color='#EF9F27', ms=14, zorder=7)

    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('Trajectory Comparison', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: Position error over time ──
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor('#F8F8F8')

    err_ekf = np.sqrt((hist_ekf[:, 0] - hist_real[:, 0])**2 +
                       (hist_ekf[:, 1] - hist_real[:, 1])**2)
    err_dr = np.sqrt((hist_dr[:, 0] - hist_real[:, 0])**2 +
                      (hist_dr[:, 1] - hist_real[:, 1])**2)
    time = np.arange(len(err_ekf)) * DT

    ax2.plot(time, err_dr * 100, color='#D85A30', lw=1.5, label='Dead Reckoning')
    ax2.plot(time, err_ekf * 100, color='#185FA5', lw=1.5, label='EKF')

    for c in corrections:
        ax2.axvline(c["step"] * DT, color='#EF9F27', alpha=0.4, lw=1, ls='--')

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position error (cm)')
    ax2.set_title('Error Over Time', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Plot 3: Covariance trace ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor('#F8F8F8')

    traces = [np.trace(P[:2, :2]) for P in hist_P]
    ax3.plot(time, traces, color='#185FA5', lw=1.5)

    for c in corrections:
        ax3.axvline(c["step"] * DT, color='#EF9F27', alpha=0.4, lw=1, ls='--')

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('tr(P) — position uncertainty')
    ax3.set_title('Covariance Trace', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    plt.savefig('ekf_simulation_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print summary
    print(f'\nFinal position error:')
    print(f'  Dead Reckoning: {err_dr[-1]*100:.1f} cm')
    print(f'  EKF:            {err_ekf[-1]*100:.1f} cm')
    print(f'  Corrections:    {len(corrections)}')


# ─────────────────────────────────────────────
if __name__ == '__main__':
    results = simulate()
    plot(*results)
