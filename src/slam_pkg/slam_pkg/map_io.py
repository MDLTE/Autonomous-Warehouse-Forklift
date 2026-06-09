import yaml
from PIL import Image
import numpy as np
from slam_pkg.occupancy_grid import OccupancyGrid


def load_map(path, origin_x=None, origin_y=None):
    with open(path + '.yaml', 'r') as f:
        metadata = yaml.safe_load(f)

    resolution = metadata['resolution']
    img = Image.open(path + '.pgm')
    pixels = np.array(img)

    rows, cols = pixels.shape
    grid = OccupancyGrid(rows=rows, cols=cols, resolution=resolution,
                         origin_x=origin_x, origin_y=origin_y)

    grid.grid[pixels == 255] = -10
    grid.grid[pixels == 0] = 20
    grid.grid[pixels == 205] = 0

    return grid

# ── Add this function to map_io.py ────────────────────────────────
# (paste alongside the existing load_map function)

def load_corridor_map(corridor_path, original_path, origin_x=None, origin_y=None):
    """
    Load the corridor mask as an OccupancyGrid for A* path planning.

    Corridor PGM convention:
      255 = walkable street  → log odds = -10 (free)
        0 = blocked          → log odds = +20 (occupied)

    A* uses obstacle threshold < 0 to determine walkability,
    so only corridor cells (log odds = -10) will be walkable.

    Args:
        corridor_path: path to corridors.pgm (without extension)
        original_path: path to original map.pgm (for metadata)
        origin_x, origin_y: grid origin override
    """
    import yaml
    import numpy as np
    from PIL import Image
    from slam_pkg.occupancy_grid import OccupancyGrid

    # load metadata from original map yaml
    with open(original_path + '.yaml', 'r') as f:
        metadata = yaml.safe_load(f)

    resolution = metadata['resolution']

    # load corridor PGM
    img    = Image.open(corridor_path + '.pgm')
    pixels = np.array(img)
    rows, cols = pixels.shape

    grid = OccupancyGrid(
        rows=rows, cols=cols,
        resolution=resolution,
        origin_x=origin_x,
        origin_y=origin_y
    )

    # 255 = corridor (walkable) → free log odds
    # 0   = blocked             → occupied log odds
    grid.grid[pixels == 255] = -10.0
    grid.grid[pixels == 0]   =  20.0

    return grid



