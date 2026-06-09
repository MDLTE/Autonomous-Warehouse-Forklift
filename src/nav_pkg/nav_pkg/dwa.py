import numpy as np
import math


class DWA():
    """
    Dynamic Window Approach para robot que maneja HACIA ADELANTE.
    El LiDAR ya viene corregido (+π) desde el nav node, asi que 0° = frente.

    Mejoras incluidas:
      - manejo hacia adelante (velocidades positivas, sin +π en goal cost)
      - horizonte de prediccion adaptativo (mira distancia fija, no tiempo fijo)
      - penalizacion de retroceso (no se aleja de la meta)
      - filtro de auto-estructura por distancia (ignora uñas del montacargas)
      - telemetria de debug que el nav node puede leer
    """

    def __init__(self):
        # ── limites de velocidad ──────────────────────────────────
        self.max_v =  0.15
        self.min_v = -0.15
        self.max_w =  0.30
        self.min_w = -0.30

        # ── limites de aceleracion ────────────────────────────────
        self.max_linear_acc  = 0.43
        self.max_angular_acc = 6.92

        # ── tiempo de simulacion ──────────────────────────────────
        self.dt = 0.1

        # horizonte adaptativo: mira una DISTANCIA fija adelante.
        # entre mas lento va, mas tiempo simula (t = distancia / v).
        # NO se ata al obstaculo mas cercano: en pasillo ese suele ser
        # la pared lateral y acortaria la vista al frente.
        self.lookahead_horizon = 0.6   # m que mira adelante
        self.predict_time_min  = 2.0   # s
        self.predict_time_max  = 6.0   # s

        # ── muestreo de velocidades ───────────────────────────────
        self.v_resolution = 0.05
        self.w_resolution = math.radians(5.0)

        # ── pesos de costo ────────────────────────────────────────
        # obstacle domina (seguridad), goal lo empuja a la meta,
        # speed solo desempata para que no se quede quieto.
        self.to_goal_cost_gain  = 0.20
        self.speed_cost_gain    = 0.05
        self.obstacle_cost_gain = 0.12

        # ── seguridad ─────────────────────────────────────────────
        self.robot_radius = 0.15

        # ── filtro de auto-estructura ─────────────────────────────
        # ignora todo dato < base_filter_dist (footprint del robot).
        # dentro del sector del montacargas usa un umbral mayor porque
        # las uñas/soportes (~0.14m) caen ahi.
        self.fork_sector_inner = math.radians(20)  # banda angular del montacargas
        self.fork_sector_outer = math.radians(42)
        self.fork_filter_dist  = 0.17               # corta dentro del sector
        self.base_filter_dist  = 0.15               # corta en todos lados (pediste 0.15)

        # ── limite angular en evasion ─────────────────────────────
        self.avoid_max_w = 0.30

        # ── rango maximo del LiDAR considerado ────────────────────
        self.max_range = 3.5

        # ── telemetria de debug (la lee el nav node) ──────────────
        self.last_n_obs         = 0
        self.last_min_obs_dist  = float('inf')
        self.last_excl_count    = 0
        self.last_excl_min_dist = float('inf')
        self.last_best_cost     = float('inf')
        self.last_best_v        = 0.0
        self.last_best_w        = 0.0
        self.last_cost_goal     = 0.0
        self.last_cost_speed    = 0.0
        self.last_cost_obs      = 0.0
        self.last_all_blocked   = False

    # ═══════════════════════════════════════════════════════════════
    #  VENTANA DINAMICA
    # ═══════════════════════════════════════════════════════════════

    def get_dynamic_window(self, current_v, current_w):
        v_min = max(self.min_v, current_v - self.max_linear_acc  * self.dt)
        v_max = min(self.max_v, current_v + self.max_linear_acc  * self.dt)
        w_min = max(self.min_w, current_w - self.max_angular_acc * self.dt)
        w_max = min(self.max_w, current_w + self.max_angular_acc * self.dt)
        return v_min, v_max, w_min, w_max

    # ═══════════════════════════════════════════════════════════════
    #  PREDICCION DE TRAYECTORIA
    # ═══════════════════════════════════════════════════════════════

    def predict_trajectory(self, robot_x, robot_y, robot_theta,
                           robot_v, robot_w, v, w, predict_time):
        state      = np.array([robot_x, robot_y, robot_theta, robot_v, robot_w])
        trajectory = np.array(state)
        t          = 0.0
        while t <= predict_time:
            state[2] += w * self.dt
            state[0] += v * math.cos(state[2]) * self.dt
            state[1] += v * math.sin(state[2]) * self.dt
            state[3]  = v
            state[4]  = w
            trajectory = np.vstack((trajectory, state))
            t += self.dt
        return trajectory

    # ═══════════════════════════════════════════════════════════════
    #  FUNCIONES DE COSTO
    # ═══════════════════════════════════════════════════════════════

    def calc_to_goal_cost(self, trajectory, goal):
        """
        Costo de rumbo para robot que maneja hacia adelante.
        effective_yaw = yaw de la trayectoria (sin +π).
        Penaliza alejarse de la meta (retreat_penalty).
        """
        dx          = goal[0] - trajectory[-1, 0]
        dy          = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)

        effective_yaw = trajectory[-1, 2]
        cost_angle    = error_angle - effective_yaw
        heading_cost  = abs(math.atan2(
            math.sin(cost_angle), math.cos(cost_angle)))

        start_dist = math.hypot(trajectory[0, 0] - goal[0],
                                trajectory[0, 1] - goal[1])
        end_dist   = math.hypot(trajectory[-1, 0] - goal[0],
                                trajectory[-1, 1] - goal[1])
        retreat_penalty = max(0.0, end_dist - start_dist) * 2.0

        return heading_cost + retreat_penalty

    def calc_obstacle_cost(self, trajectory, obstacle_points):
        if not obstacle_points:
            return 0.0
        obs = np.array(obstacle_points)
        min_dist = float('inf')
        for i in range(len(trajectory)):
            dists   = np.sqrt(
                (trajectory[i, 0] - obs[:, 0])**2 +
                (trajectory[i, 1] - obs[:, 1])**2)
            nearest = float(np.min(dists))
            if nearest <= self.robot_radius:
                return float('inf')   # colision — descartar
            min_dist = min(min_dist, nearest)
        return 1.0 / min_dist

    def calc_speed_cost(self, trajectory):
        return self.max_v - abs(trajectory[-1, 3])

    # ═══════════════════════════════════════════════════════════════
    #  LIDAR → PUNTOS DE OBSTACULO
    # ═══════════════════════════════════════════════════════════════

    def get_obstacle_points(self, robot_x, robot_y, robot_theta,
                            scan_angles, scan_ranges):
        """
        Convierte el scan a puntos de obstaculo en marco mundial.
        Filtro fuerte (fork_filter_dist) dentro del sector del montacargas,
        filtro base (base_filter_dist = 0.15) en todos lados.
        Sin exclusion angular total — los costados siguen viendo.
        """
        points     = []
        min_kept   = float('inf')
        near_count = 0
        near_min   = float('inf')

        for beam, dist in zip(scan_angles, scan_ranges):
            if np.isinf(dist) or dist <= 0 or dist >= self.max_range:
                continue

            in_fork = (self.fork_sector_inner <= abs(beam)
                       <= self.fork_sector_outer)
            thr = self.fork_filter_dist if in_fork else self.base_filter_dist
            if dist < thr:
                near_count += 1
                if dist < near_min:
                    near_min = dist
                continue

            wa = robot_theta + beam
            wx = robot_x + dist * np.cos(wa)
            wy = robot_y + dist * np.sin(wa)
            points.append((wx, wy))
            if dist < min_kept:
                min_kept = dist

        self.last_min_obs_dist  = min_kept
        self.last_excl_count    = near_count
        self.last_excl_min_dist = near_min
        return points

    # ═══════════════════════════════════════════════════════════════
    #  COMPUTE PRINCIPAL
    # ═══════════════════════════════════════════════════════════════

    def compute(self, robot_x, robot_y, robot_theta, robot_v, robot_w,
                goal, scan_angles, scan_ranges, avoiding=False):
        obstacles = self.get_obstacle_points(
            robot_x, robot_y, robot_theta, scan_angles, scan_ranges)

        if avoiding:
            # robot hacia adelante — solo velocidades positivas
            v_min, v_max = 0.05, self.max_v
            w_min, w_max = -self.avoid_max_w, self.avoid_max_w
        else:
            v_min, v_max, w_min, w_max = self.get_dynamic_window(
                robot_v, robot_w)

        v_values = np.arange(v_min, v_max, self.v_resolution)
        w_values = np.arange(w_min, w_max, self.w_resolution)

        best_cost = float('inf')
        best_v    = 0.0
        best_w    = 0.0
        best_cg   = 0.0
        best_cs   = 0.0
        best_co   = 0.0

        for v in v_values:
            # horizonte ajustado a la velocidad: distancia fija / v
            t_pred = min(self.predict_time_max,
                         max(self.predict_time_min,
                             self.lookahead_horizon / max(v, 1e-3)))
            for w in w_values:
                traj = self.predict_trajectory(
                    robot_x, robot_y, robot_theta, robot_v, robot_w,
                    v, w, t_pred)
                cg = self.to_goal_cost_gain  * self.calc_to_goal_cost(traj, goal)
                cs = self.speed_cost_gain    * self.calc_speed_cost(traj)
                co = self.obstacle_cost_gain * self.calc_obstacle_cost(traj, obstacles)
                total = cg + cs + co

                if total == float('inf'):
                    continue
                if total < best_cost:
                    best_cost = total
                    best_v    = v
                    best_w    = w
                    best_cg   = cg
                    best_cs   = cs
                    best_co   = co

        # telemetria
        self.last_n_obs       = len(obstacles)
        self.last_best_cost   = best_cost
        self.last_best_v      = best_v
        self.last_best_w      = best_w
        self.last_cost_goal   = best_cg
        self.last_cost_speed  = best_cs
        self.last_cost_obs    = best_co
        self.last_all_blocked = (best_cost == float('inf'))

        return best_v, best_w