import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import os
from slam_pkg.map_io import load_map

home = os.path.expanduser('~')
MAP_PATH  = os.path.join(home, 'ros2_ws', 'src', 'slam_pkg', 'maps', 'map')
SAVE_PATH = os.path.join(home, 'ros2_ws', 'src', 'slam_pkg', 'maps', 'map_clean')

# load map
grid = load_map(MAP_PATH)

# state
brush_size = 3      # radius in cells
mode = 'clear'      # 'clear' or 'occupy'
painting = False

def get_prob_map():
    return 1 - (1 / (1 + np.exp(grid.grid)))

def apply_brush(gx, gy):
    for dx in range(-brush_size, brush_size + 1):
        for dy in range(-brush_size, brush_size + 1):
            if dx**2 + dy**2 <= brush_size**2:
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < grid.cols and 0 <= ny < grid.rows:
                    if mode == 'clear':
                        grid.grid[ny, nx] = -10.0
                    else:
                        grid.grid[ny, nx] = 10.0

def update_display():
    prob_map = get_prob_map()
    im.set_data(prob_map)
    fig.canvas.draw_idle()

def on_press(event):
    global painting
    if event.inaxes != ax:
        return
    painting = True
    gx, gy = int(event.xdata), int(event.ydata)
    apply_brush(gx, gy)
    update_display()

def on_release(event):
    global painting
    painting = False

def on_motion(event):
    if not painting or event.inaxes != ax:
        return
    gx, gy = int(event.xdata), int(event.ydata)
    apply_brush(gx, gy)
    update_display()

def on_save(event):
    grid.save_map(SAVE_PATH)
    print(f'Map saved to {SAVE_PATH}')

def on_mode_change(label):
    global mode
    mode = 'clear' if label == 'Clear (free)' else 'occupy'

def on_brush_change(val):
    global brush_size
    brush_size = int(val)

# ── build UI ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 10))
plt.subplots_adjust(left=0.2, bottom=0.15)

prob_map = get_prob_map()
im = ax.imshow(prob_map, cmap='gray_r', origin='lower')
ax.set_title('Left click to paint — hold and drag for continuous painting')

# brush size slider
ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
slider = Slider(ax_slider, 'Brush size', 1, 20, valinit=brush_size, valstep=1)
slider.on_changed(on_brush_change)

# mode radio buttons
ax_radio = plt.axes([0.02, 0.4, 0.15, 0.15])
radio = RadioButtons(ax_radio, ['Clear (free)', 'Mark occupied'])
radio.on_clicked(on_mode_change)

# save button
ax_save = plt.axes([0.75, 0.05, 0.1, 0.04])
btn_save = Button(ax_save, 'Save')
btn_save.on_clicked(on_save)

# connect events
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)

print('Controls:')
print('  Left click + drag → paint')
print('  Radio buttons     → switch between clear/occupy mode')
print('  Brush size slider → change brush size')
print('  Save button       → save to map_clean')

plt.show()