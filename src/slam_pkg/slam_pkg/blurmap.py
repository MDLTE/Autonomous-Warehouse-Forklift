import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

home = os.path.expanduser('~')
MAP_PATH  = os.path.join(home, 'ros2_ws', 'src', 'slam_pkg', 'maps', 'map_clean')
SAVE_PATH = os.path.join(home, 'ros2_ws', 'src', 'slam_pkg', 'maps', 'map_processed')

# load map as grayscale
map_img = cv2.imread(MAP_PATH + '.pgm', cv2.IMREAD_GRAYSCALE)
if map_img is None:
    print(f'Error — could not load map from {MAP_PATH}.pgm')
    exit()
print(f'Map loaded: {map_img.shape}')

# treat unknown cells (205) as free before processing
map_clean = map_img.copy()
map_clean[map_clean == 205] = 255

# apply gaussian blur to remove noise
blur = cv2.GaussianBlur(map_clean, (5, 5), 0)

# apply otsu thresholding — auto finds best threshold
ret, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f'Otsu threshold: {ret:.1f}')

# apply dilation to connect broken wall segments
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(binary, kernel, iterations=1)

# show all steps side by side
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(map_img,   cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Original')
axes[1].imshow(blur,      cmap='gray', vmin=0, vmax=255)
axes[1].set_title('Gaussian blur')
axes[2].imshow(binary,    cmap='gray', vmin=0, vmax=255)
axes[2].set_title(f'Otsu (threshold={ret:.0f})')
axes[3].imshow(dilated,   cmap='gray', vmin=0, vmax=255)
axes[3].set_title('Dilated')
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()

# save result as PGM
cv2.imwrite(SAVE_PATH + '.pgm', dilated)

# copy YAML from original map with updated image name
import yaml
with open(MAP_PATH + '.yaml', 'r') as f:
    metadata = yaml.safe_load(f)
metadata['image'] = SAVE_PATH + '.pgm'
with open(SAVE_PATH + '.yaml', 'w') as f:
    yaml.dump(metadata, f)

print(f'Saved to {SAVE_PATH}.pgm and {SAVE_PATH}.yaml')