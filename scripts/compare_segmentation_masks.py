
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Load original CT volume
volume = np.load(
    "data/processed/ct_volumes/patient_100012.npz"
)["volume"]

# Step 1: Original threshold segmentation
threshold_mask = volume < -400

# Step 2: Morphological closing, slice by slice
closed_mask = np.zeros_like(threshold_mask, dtype=bool)

structure = ndimage.generate_binary_structure(2, 1)

for z in range(volume.shape[0]):
    closed_mask[z] = ndimage.binary_closing(
        threshold_mask[z],
        structure=structure,
        iterations=3
    )

# Step 3: Hole filling, slice by slice
filled_mask = np.zeros_like(closed_mask, dtype=bool)

for z in range(volume.shape[0]):
    filled_mask[z] = ndimage.binary_fill_holes(
        closed_mask[z]
    )

# Annotation location
z = 125
row = 313
col = 199

# Display comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

masks = [
    threshold_mask[z],
    closed_mask[z],
    filled_mask[z]
]

titles = [
    "Original Threshold",
    "After Closing",
    "After Hole Filling"
]

for ax, mask, title in zip(axes, masks, titles):
    ax.imshow(mask, cmap="gray", origin="lower")
    ax.scatter(
        col,
        row,
        color="red",
        marker="+",
        s=150
    )
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    "outputs/compare_segmentation_masks.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

# Print foreground voxel counts
print("Threshold foreground:", threshold_mask[z].sum())
print("After closing:", closed_mask[z].sum())
print("After filling:", filled_mask[z].sum())
