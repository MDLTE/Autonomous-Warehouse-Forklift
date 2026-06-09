import numpy as np
from slam_pkg.particle import Particle

class MCL():
    def __init__(self, n_particles, initial_x, initial_y, initial_theta, grid, global_init=False):
        # Initialize parameters
        self.particles = []
        self.n = n_particles
        self.grid = grid

        if global_init:
            for _ in range(n_particles):
                p = Particle(
                    x = np.random.uniform(-0.015, 3.673),   
                    y = np.random.uniform(-0.626, 4.865),   
                    theta = np.random.uniform(-np.pi, np.pi),
                    weight = 1.0/n_particles
                )
                self.particles.append(p)
        else:
            for _ in range(n_particles):
                p = Particle(
                    x = np.random.normal(initial_x, 0.3),     
                    y = np.random.normal(initial_y, 0.3),
                    theta = np.random.normal(initial_theta, 0.01), 
                    weight = 1.0/n_particles
                )
                self.particles.append(p)        

    def predict(self, dx, dy, dtheta):
        for p in self.particles:
            p.x += dx + np.random.normal(0, abs(dx) * 0.1 + 0.01)
            p.y += dy + np.random.normal(0, abs(dy) * 0.1 + 0.01)
            p.theta += dtheta + np.random.normal(0, abs(dtheta) * 0.05 + 0.001)
            p.theta = np.arctan2(np.sin(p.theta), np.cos(p.theta))

    def update_weights(self, scan_angles, scan_ranges, range_max=3.5):
        step = max(1, len(scan_angles) // 20)
        sigma = 0.15
        for p in self.particles: 
            log_weight = 0.0
            for i in range(0, len(scan_angles), step):
                distance = scan_ranges[i]
                beam_angle = scan_angles[i] 
                # Skip if not in range
                if np.isinf(distance) or distance >= range_max:
                    continue
                world_angle = p.theta + beam_angle

                expected = self.raycast(p, world_angle, range_max)
                diff = distance - expected
                log_weight += -(diff**2) / (2 * sigma ** 2)
            p.weight = log_weight
        self.normalize_weights()

    def raycast(self, p, world_angle, range_max=3.5, step=0.05):
        for d in np.arange(0, range_max, step):
            wx = p.x + d * np.cos(world_angle)
            wy = p.y + d * np.sin(world_angle)
            result = self.grid.world_to_grid(wx, wy)
            if result is None:
                return range_max
            gx, gy = result
            if self.grid.grid[gy, gx] > 0:
                return d
        return range_max
    
    def normalize_weights(self):
        weights = np.array([p.weight for p in self.particles])
        weights -= np.max(weights)
        weights = np.exp(weights)
        total = np.sum(weights)
        if total > 0:
            weights /= total
        for i, p in enumerate(self.particles):
            p.weight = weights[i]
        
    def resample(self):
        weights = np.array([p.weight for p in self.particles])
        step = 1.0 / self.n
        start = np.random.uniform(0,step)
        new_particles = []
        cumulative = np.cumsum(weights)
        for i in range(self.n):
            pointer = start + i * step
            idx = min(np.searchsorted(cumulative, pointer), self.n - 1)
            p = self.particles[idx]
            new_particles.append(Particle(p.x, p.y, p.theta, 1.0 / self.n))

        self.particles = new_particles

    def estimate(self):
        # First extract all the values from the particles object
        xs = np.array([p.x for p in self.particles])
        ys = np.array([p.y for p in self.particles])
        thetas = np.array([p.theta for p in self.particles])
        weights = np.array([p.weight for p in self.particles])

        # Estimate the best position
        estimated_x = np.sum(xs * weights)
        estimated_y = np.sum(ys * weights)
        estimated_theta = np.arctan2(np.sum(np.sin(thetas) * weights), np.sum(np.cos(thetas) * weights))
        return estimated_x, estimated_y, estimated_theta


                



