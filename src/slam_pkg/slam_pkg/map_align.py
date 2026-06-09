import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift
from slam_pkg.map_io import load_map
import yaml
from PIL import Image

# globals
points = []
grid = None
fig, ax = None, None

def on_click(event):
    global points, grid

    if event.inaxes != ax:
        return

    x, y = int(event.xdata), int(event.ydata)
    points.append((x, y))
    ax.scatter(x, y, c='red', s=50, zorder=5)
    plt.pause(0.001)

    if len(points) == 1:
        print(f'Point 1: {points[0]} — now click second point')

    elif len(points) == 2:
        # draw reference line
        ax.plot([points[0][0], points[1][0]],
                [points[0][1], points[1][1]],
                'r-', linewidth=2)
        plt.pause(0.001)

        # calculate angle
        dx = points[1][0] - points[0][0]
        dy = points[1][1] - points[0][1]
        angle = np.degrees(np.arctan2(dy, dx))
        print(f'Line angle: {angle:.2f} degrees')

        # try positive angle first — flip sign if map looks wrong
        rotation_angle = angle
        print(f'Rotating by: {rotation_angle:.2f} degrees')

        # rotate
        rotated = rotate(
            grid.grid,
            rotation_angle + 180,
            reshape=False,
            order=1,
            mode='constant',
            cval=0.0
        )

        # find bounding box of all non-unknown cells
        non_unknown = rotated != 0.0
        rows_with_data = np.any(non_unknown, axis=1)
        cols_with_data = np.any(non_unknown, axis=0)

        if not np.any(rows_with_data) or not np.any(cols_with_data):
            print('No data found after rotation — try clicking different points')
            points = []
            return

        row_indices = np.where(rows_with_data)[0]
        col_indices = np.where(cols_with_data)[0]

        row_min, row_max = row_indices[0], row_indices[-1]
        col_min, col_max = col_indices[0], col_indices[-1]

        print(f'Content bounding box: rows [{row_min},{row_max}], cols [{col_min},{col_max}]')

        # calculate shift to center content
        content_center_row = (row_min + row_max) / 2.0
        content_center_col = (col_min + col_max) / 2.0
        grid_center_row = grid.rows / 2.0
        grid_center_col = grid.cols / 2.0

        manual_offset = -12

        shift_row = grid_center_row - content_center_row
        shift_col = grid_center_col - content_center_col + manual_offset
        print(f'Shifting by: row={shift_row:.1f}, col={shift_col:.1f}')

        # apply shift
        shifted = shift(rotated, [shift_row, shift_col], mode='constant', cval=0.0)

        # display result
        prob_map = 1 - (1 / (1 + np.exp(shifted)))
        ax.cla()
        ax.imshow(prob_map, cmap='gray_r', origin='lower')
        ax.set_title(f'Rotated {rotation_angle:.1f}° and centered — Ctrl+C to save')
        plt.pause(0.001)

        # store result
        grid.grid = shifted
        print('Done. Press Ctrl+C to save.')

def main():
    global grid, fig, ax

    map_path = '/home/patricio/ros2_ws/src/slam_pkg/maps/map'
    save_path = '/home/patricio/ros2_ws/src/slam_pkg/maps/map_aligned'

    grid = load_map(map_path)
    print(f'Map loaded: {grid.rows}x{grid.cols} at {grid.resolution}m/cell')
    print('Click 2 points on the bottom wall to define the X axis')

    fig, ax = plt.subplots(figsize=(8, 8))
    prob_map = 1 - (1 / (1 + np.exp(grid.grid)))
    ax.imshow(prob_map, cmap='gray_r', origin='lower')
    ax.set_title('Click 2 points on the bottom wall (X axis reference)')
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.ion()
    plt.show(block=False)

    try:
        while plt.get_fignums():
            plt.pause(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        grid.save_map(save_path)
        print(f'Saved to {save_path}')

if __name__ == '__main__':
    main()