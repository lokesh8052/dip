
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from skimage.morphology import disk

# 1. Load CT volume
volume = np.load(
    "data/processed/ct_volumes/patient_100012.npz"
)["volume"]

print("Volume shape:", volume.shape)

# 2. HU thresholding
binary_mask = volume < -400

print("Thresholding completed.")

# 3. Morphological closing, slice by slice
structure = disk(3)

closed_mask = np.zeros_like(binary_mask, dtype=bool)

for i in range(volume.shape[0]):
    closed_mask[i] = ndimage.binary_closing(
        binary_mask[i],
        structure=structure
    )

    if (i + 1) % 20 == 0:
        print(f"Closing: {i + 1}/{volume.shape[0]}")

print("Morphological closing completed.")

# 4. Fill enclosed holes slice by slice
refined_mask = np.zeros_like(closed_mask, dtype=bool)

for i in range(volume.shape[0]):
    refined_mask[i] = ndimage.binary_fill_holes(
        closed_mask[i]
    )

print("Hole filling completed.")

# 5. Known annotation
z, row, col = 125, 313, 199

# 6. Visualize the results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].imshow(
    volume[z],
    cmap="gray",
    vmin=-1000,
    vmax=400
)
axes[0, 0].set_title("Original CT")

axes[0, 1].imshow(
    binary_mask[z],
    cmap="gray"
)
axes[0, 1].set_title("Threshold Mask")

axes[1, 0].imshow(
    closed_mask[z],
    cmap="gray"
)
axes[1, 0].set_title("After Morphological Closing")

axes[1, 1].imshow(
    refined_mask[z],
    cmap="gray"
)
axes[1, 1].set_title("After Hole Filling")

for ax in axes.flat:
    ax.scatter(
        col, row,
        color="red",
        marker="+",
        s=100
    )
    ax.axis("off")

plt.tight_layout()

os.makedirs("outputs", exist_ok=True)

plt.savefig(
    "outputs/morphological_refinement_patient_100012.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

# 7. Print statistics
print("\nMASK STATISTICS")
print("Original threshold mask:", binary_mask.sum())
print("After closing:", closed_mask.sum())
print("After hole filling:", refined_mask.sum())

# 8. Save intermediate mask
output_path = (
    "data/processed/ct_volumes/"
    "patient_100012_lung_mask.npz"
)

np.savez_compressed(
    output_path,
    mask=refined_mask
)

print("\nRefined mask saved:", output_path)
