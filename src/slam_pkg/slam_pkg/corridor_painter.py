import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from slam_pkg.map_io import load_map
import os
import yaml
from PIL import Image

home      = os.path.expanduser('~')
MAP_PATH  = os.path.join(home, 'ros2_ws', 'src', 'slam_pkg', 'maps', 'map')
SAVE_PATH = os.path.join(home, 'ros2_ws', 'src', 'slam_pkg', 'maps', 'corridors')

# ── load map ──────────────────────────────────────────────────────
grid = load_map(MAP_PATH)

rows, cols = grid.grid.shape

# corridor mask — False = blocked, True = walkable street
# starts fully blocked — user paints walkable areas
corridor_mask = np.zeros((rows, cols), dtype=bool)

# state
brush_size = 5
painting   = False

# ── helpers ───────────────────────────────────────────────────────

def apply_brush(gx, gy):
    """Mark cells within brush_size radius as walkable corridor."""
    for dx in range(-brush_size, brush_size + 1):
        for dy in range(-brush_size, brush_size + 1):
            if dx**2 + dy**2 <= brush_size**2:
                nx, ny = int(gx + dx), int(gy + dy)
                if 0 <= nx < cols and 0 <= ny < rows:
                    # only paint on free cells — don't paint over walls
                    if grid.grid[ny, nx] <= 0:
                        corridor_mask[ny, nx] = True

def erase_brush(gx, gy):
    """Remove corridor marking from cells within brush radius."""
    for dx in range(-brush_size, brush_size + 1):
        for dy in range(-brush_size, brush_size + 1):
            if dx**2 + dy**2 <= brush_size**2:
                nx, ny = int(gx + dx), int(gy + dy)
                if 0 <= nx < cols and 0 <= ny < rows:
                    corridor_mask[ny, nx] = False

def build_display():
    """
    Build RGB display image:
      white   = free space (not painted)
      black   = wall/obstacle
      green   = painted corridor
      gray    = unknown
    """
    prob = 1 - (1 / (1 + np.exp(grid.grid)))
    rgb  = np.ones((rows, cols, 3))

    # white = free
    rgb[:, :, 0] = 1 - prob
    rgb[:, :, 1] = 1 - prob
    rgb[:, :, 2] = 1 - prob

    # overlay corridors in green
    rgb[corridor_mask, 0] = 0.0
    rgb[corridor_mask, 1] = 0.8
    rgb[corridor_mask, 2] = 0.2

    return rgb

def update_display():
    im.set_data(build_display())
    fig.canvas.draw_idle()

# ── mouse events ──────────────────────────────────────────────────

def on_press(event):
    global painting
    if event.inaxes != ax:
        return
    painting = True
    gx, gy = int(event.xdata), int(event.ydata)
    if mode == 'paint':
        apply_brush(gx, gy)
    else:
        erase_brush(gx, gy)
    update_display()

def on_release(event):
    global painting
    painting = False

def on_motion(event):
    if not painting or event.inaxes != ax:
        return
    gx, gy = int(event.xdata), int(event.ydata)
    if mode == 'paint':
        apply_brush(gx, gy)
    else:
        erase_brush(gx, gy)
    update_display()

# ── button callbacks ──────────────────────────────────────────────

def on_mode_change(label):
    global mode
    mode = 'paint' if label == 'Paint' else 'erase'

def on_brush_change(val):
    global brush_size
    brush_size = int(val)

def on_clear(event):
    global corridor_mask
    corridor_mask = np.zeros((rows, cols), dtype=bool)
    update_display()
    print('Corridor mask cleared')

def on_save(event):
    """
    Save corridor mask as PGM + YAML.
    Walkable corridor cells → 255 (free)
    Everything else → 0 (occupied/blocked)
    A* will use this map — only white cells are walkable.
    """
    img_array = np.zeros((rows, cols), dtype=np.uint8)
    img_array[corridor_mask] = 255

    # save PGM
    img = Image.fromarray(img_array)
    img.save(SAVE_PATH + '.pgm')

    # save YAML — same metadata as original map
    with open(MAP_PATH + '.yaml', 'r') as f:
        metadata = yaml.safe_load(f)
    metadata['image'] = SAVE_PATH + '.pgm'
    with open(SAVE_PATH + '.yaml', 'w') as f:
        yaml.dump(metadata, f)

    print(f'Corridors saved to {SAVE_PATH}.pgm and {SAVE_PATH}.yaml')
    print(f'Painted cells: {np.sum(corridor_mask)}')

# ── build UI ──────────────────────────────────────────────────────

mode = 'paint'

fig, ax = plt.subplots(figsize=(10, 12))
plt.subplots_adjust(left=0.2, bottom=0.18)

im = ax.imshow(build_display(), origin='lower')
ax.set_title('Corridor Painter — left click to paint walkable streets')

# brush size slider
ax_slider = plt.axes([0.25, 0.08, 0.5, 0.03])
slider    = Slider(ax_slider, 'Brush size', 1, 40, valinit=brush_size, valstep=1)
slider.on_changed(on_brush_change)

# paint / erase radio
ax_radio = plt.axes([0.02, 0.45, 0.14, 0.12])
radio    = RadioButtons(ax_radio, ['Paint', 'Erase'])
radio.on_clicked(on_mode_change)

# clear button
ax_clear = plt.axes([0.25, 0.03, 0.15, 0.04])
btn_clear = Button(ax_clear, 'Clear all')
btn_clear.on_clicked(on_clear)

# save button
ax_save  = plt.axes([0.60, 0.03, 0.15, 0.04])
btn_save = Button(ax_save, 'Save corridors')
btn_save.on_clicked(on_save)

# connect events
fig.canvas.mpl_connect('button_press_event',   on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event',  on_motion)

print('Corridor Painter ready')
print('  Left click + drag → paint corridor (green)')
print('  Switch to Erase   → remove corridor')
print('  Brush size slider → adjust brush')
print('  Save corridors    → saves corridors.pgm + corridors.yaml')
print()
print('Note: corridors can only be painted on free space (white cells)')
print('      walls (black cells) cannot be painted over')

plt.show()