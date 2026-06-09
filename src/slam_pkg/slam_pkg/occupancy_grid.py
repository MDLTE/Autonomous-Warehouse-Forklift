import numpy as np
from PIL import Image
import yaml
import math
from scipy.ndimage import binary_dilation
import copy
import numpy as np

class OccupancyGrid():
    def __init__(self, rows, cols, resolution, origin_x=None, origin_y=None):
        # Grid parameters
        self.rows = rows
        self.cols = cols
        self.resolution = resolution
        self.center_x = origin_x if origin_x is not None else cols // 2
        self.center_y = origin_y if origin_y is not None else rows // 2
        self.grid = np.zeros((self.rows,self.cols)) 

        # Robot position - for resetting the robot's position
        self.initial_y = 0 
        self.initial_x = 0
        self.initial_theta = 0

        self.x = 0
        self.y = 0
        self.theta = 0

        # Probability parameters
        self.L_OCCUPIED = 3.0
        self.L_FREE = 0.1


    def inflate_obstacles(self, robot_radius_m):
        
        # convert radius from meters to grid cells
        radius_cells = int(robot_radius_m / self.resolution)

        # binary mask of occupied cells
        occupied = self.grid > 0

        # circular kernel for accurate round inflation
        y, x   = np.ogrid[-radius_cells:radius_cells+1, -radius_cells:radius_cells+1]
        kernel = (x**2 + y**2) <= radius_cells**2

        # expand occupied regions outward using circular kernel
        inflated = binary_dilation(occupied, structure=kernel)

        # create a copy of the grid — preserves center_x, center_y, resolution
        inflated_grid = copy.deepcopy(self)

        # mark inflated cells as occupied in the copy only
        inflated_grid.grid[inflated] = 10.0

        print(f'Map inflated by {robot_radius_m}m ({radius_cells} cells)')
        return inflated_grid
    
    def world_to_grid(self, world_x, world_y):
        # We need the center of the grid, the resolution, and the world position
        grid_x = int(world_x / self.resolution + self.center_x)
        grid_y = int(world_y / self.resolution + self.center_y)
        # If the calculated indices are within the boundries of our maps, return them
        if 0 <= grid_x < self.cols and 0<= grid_y < self.rows: 
            return grid_x, grid_y
        return None
    
    def update(self, grid_x, grid_y, occupied):
        # Redundency - Check if the targeted cell is inside our map
        if not (0 <= grid_x < self.cols and 0 <= grid_y < self.rows):
            return 
        if occupied:
            self.grid[grid_y,grid_x] += self.L_OCCUPIED
        else:
            self.grid[grid_y,grid_x] -= self.L_FREE

        self.grid[grid_y, grid_x] = np.clip(self.grid[grid_y, grid_x], -10, 20)

    def bresenham(self, x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1 
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        cells = []

        while True:
            cells.append((x0,y0))
            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * err

            if e2 > -dy:
                err -= dy
                x0 += sx 

            if e2 < dx:
                err += dx
                y0 += sy

        return cells
    
    def update_map_with_scan(self, robot_x, robot_y, robot_theta, scan_angles, scan_ranges, range_max=3.5):
        robot_grid = self.world_to_grid(robot_x, robot_y)
        if robot_grid is None:
            return
        robot_grid_x, robot_grid_y = robot_grid

        for beam_angle, distance in zip(scan_angles, scan_ranges):
            # Skip over readings which we can't use
            if np.isinf(distance) or distance > range_max:
                continue
            
            # Find the world angle of each beam
            world_angle = robot_theta + beam_angle + math.pi

            # Find where it landed based on the robot's current position
            end_x = robot_x + distance * np.cos(world_angle)
            end_y = robot_y + distance * np.sin(world_angle)

            # Convert it to grid units
            result = self.world_to_grid(end_x, end_y)
            if result is None:
                continue
            end_grid_x, end_grid_y = result

            # Use Bresenham's algorithm to find all contacted cells
            cells = self.bresenham(robot_grid_x, robot_grid_y, end_grid_x, end_grid_y)
            if not cells:
                continue

            # Update probability matrix for each cell contacted
            for cell in cells[:-1]: # All the cells, except the last one
                self.update(cell[0], cell[1], False)
            self.update(cells[-1][0], cells[-1][1], True) # The occupied cell

    def save_map(self, path='map'):
        # Convert odds to probability
        prob_map = 1 - (1 / (1 + np.exp(self.grid)))

        # Convert probabilities into pixel values
        # free = 255 | unkown = 205 | occupied = 0 
        image = np.zeros_like(prob_map, dtype=np.uint8)
        image[prob_map < 0.2] = 255
        image[prob_map > 0.8] = 0
        image[(prob_map >= 0.2) & (prob_map <= 0.8)] = 205

        img = Image.fromarray(image)
        img.save(path + '.pgm')

        metadata = {
            'image': path + '.pgm',
            'resolution': float(self.resolution),
            'origin': [-self.center_x * self.resolution, 
                    -self.center_y * self.resolution, 
                    0.0],
            'negate': 0,
            'occupied_thresh': 0.65,
            'free_thresh': 0.196
        }
        with open(path + '.yaml', 'w') as f:
            yaml.dump(metadata, f)

        print(f'Map saved: {path}.pgm and {path}.yaml')
        print(f'Grid size: {self.rows}x{self.cols}, resolution: {self.resolution}m/cell')



        
        

