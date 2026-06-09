import numpy as np
import heapq

class Astar():
    def __init__(self, grid):
        self.grid = grid
        self.directions = [(0,1,1),
                           (1,1,1.414),
                           (1,0,1),
                           (1,-1,1.414),
                           (0,-1, 1),
                           (-1,-1,1.414),
                           (-1,0,1),
                           (-1,1,1.414)]
        self.obstacleThreshold = -0.5

    def isWalkable(self, gx, gy):
        if not (0 <= gx < self.grid.cols and 0 <= gy < self.grid.rows):
            return False
        return self.grid.grid[gy, gx] < self.obstacleThreshold

    def find_nearest_walkable(self, gx, gy, search_radius=50):
        """
        Search outward in expanding rings from (gx, gy) until a
        walkable cell is found. Used when robot or goal is outside
        the painted corridor zone.
        """
        for r in range(1, search_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    # only check cells on the outer edge of this ring
                    if abs(dx) != r and abs(dy) != r:
                        continue
                    nx, ny = gx + dx, gy + dy
                    if self.isWalkable(nx, ny):
                        return (nx, ny)
        return None

    def find_path(self, start_world, end_world):
        start_grid = self.grid.world_to_grid(start_world[0], start_world[1])
        end_grid   = self.grid.world_to_grid(end_world[0],   end_world[1])

        if start_grid is None or end_grid is None:
            return None

        # if start is outside corridor — snap to nearest walkable cell
        if not self.isWalkable(start_grid[0], start_grid[1]):
            start_grid = self.find_nearest_walkable(start_grid[0], start_grid[1])
            if start_grid is None:
                return None

        # if goal is outside corridor — snap to nearest walkable cell
        if not self.isWalkable(end_grid[0], end_grid[1]):
            end_grid = self.find_nearest_walkable(end_grid[0], end_grid[1])
            if end_grid is None:
                return None

        open_list  = []
        closed_set = set()
        came_from  = {}
        g_scores   = {start_grid: 0}

        heapq.heappush(open_list, (self.heuristic(start_grid, end_grid), start_grid))

        while open_list:
            _, current = heapq.heappop(open_list)

            if current == end_grid:
                return self.reconstruct_path(came_from, current)

            closed_set.add(current)

            for dx, dy, cost in self.directions:
                neighbor = (current[0] + dx, current[1] + dy)

                if neighbor in closed_set:
                    continue
                if not self.isWalkable(neighbor[0], neighbor[1]):
                    continue

                new_g = g_scores[current] + cost
                if neighbor not in g_scores or new_g < g_scores[neighbor]:
                    g_scores[neighbor] = new_g
                    f = new_g + self.heuristic(neighbor, end_grid)
                    heapq.heappush(open_list, (f, neighbor))
                    came_from[neighbor] = current

        return None

    def simplify_path_count(self, path, max_waypoints=15):
        """Reduce path to at most max_waypoints evenly spaced points."""
        if len(path) <= max_waypoints:
            return path
        indices    = np.linspace(0, len(path) - 1, max_waypoints, dtype=int)
        simplified = [path[i] for i in indices]
        if simplified[-1] != path[-1]:
            simplified[-1] = path[-1]
        return simplified

    def heuristic(self, a, b):
        return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

    def reconstruct_path(self, came_from, current):
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        path.reverse()

        world_path = []
        for gx, gy in path:
            wx, wy = self.grid_to_world(gx, gy)
            world_path.append((wx, wy))

        world_path = self.simplify_path_count(world_path, max_waypoints=15)
        return world_path

    def grid_to_world(self, gx, gy):
        wx = (gx - self.grid.center_x) * self.grid.resolution
        wy = (gy - self.grid.center_y) * self.grid.resolution
        return wx, wy