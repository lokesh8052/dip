
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import disk

# --------------------------------------------------
# 1. LOAD CT VOLUME
# --------------------------------------------------

INPUT_PATH = "data/processed/ct_volumes/patient_106046.npz"
OUTPUT_DIR = "data/processed/ct_volumes"
OUTPUT_MASK = os.path.join(
    OUTPUT_DIR, "patient_106046_lung_mask.npz"
)
OUTPUT_IMAGE = "outputs/patient_106046_lung_segmentation.png"

volume = np.load(INPUT_PATH)["volume"]
print("CT volume shape:", volume.shape)

# --------------------------------------------------
# 2. INITIALIZE SEGMENTATION
# --------------------------------------------------

lung_mask = np.zeros(volume.shape, dtype=bool)

HU_THRESHOLD = -400
MIN_COMPONENT_SIZE = 500
structure = disk(5)

# --------------------------------------------------
# 3. PROCESS EACH AXIAL SLICE
# --------------------------------------------------

for z in range(volume.shape[0]):

    # Air and low-density regions
    binary = volume[z] < HU_THRESHOLD

    # Find 2D connected components
    labels, count = ndimage.label(
        binary,
        structure=np.ones((3, 3), dtype=bool)
    )

    if count == 0:
        continue

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0

    # Remove components touching image boundaries
    border_labels = np.unique(np.concatenate([
        labels[0, :],
        labels[-1, :],
        labels[:, 0],
        labels[:, -1]
    ]))

    sizes[border_labels] = 0

    # Select the two largest internal air regions
    largest = np.argsort(sizes)[::-1][:2]

    selected = np.isin(labels, largest + 0)
    selected &= labels != 0

    # Remove tiny regions
    selected_labels, selected_count = ndimage.label(selected)

    if selected_count:
        selected_sizes = np.bincount(selected_labels.ravel())
        valid = np.where(selected_sizes >= MIN_COMPONENT_SIZE)[0]
        selected = np.isin(selected_labels, valid)
        selected &= selected_labels != 0

    # Morphological closing and hole filling
    selected = ndimage.binary_closing(
        selected, structure=structure
    )
    selected = ndimage.binary_fill_holes(selected)

    lung_mask[z] = selected

    if (z + 1) % 20 == 0:
        print(f"Processed slices: {z + 1}/{volume.shape[0]}")

# --------------------------------------------------
# 4. SAVE MASK
# --------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

np.savez_compressed(OUTPUT_MASK, mask=lung_mask)

print("\nSEGMENTATION RESULTS")
print("Volume shape:", volume.shape)
print("Lung mask shape:", lung_mask.shape)
print("Lung voxels:", int(lung_mask.sum()))
print("Lung percentage:", round(100 * lung_mask.mean(), 2), "%")
print("Saved:", OUTPUT_MASK)

# --------------------------------------------------
# 5. VISUALIZE SELECTED SLICES
# --------------------------------------------------

slice_indices = [
    int(volume.shape[0] * p)
    for p in [0.25, 0.35, 0.45, 0.55, 0.65]
]

fig, axes = plt.subplots(
    len(slice_indices), 2,
    figsize=(10, 4 * len(slice_indices))
)

for i, z in enumerate(slice_indices):

    axes[i, 0].imshow(
        volume[z], cmap="gray",
        vmin=-1000, vmax=400
    )
    axes[i, 0].set_title(f"Original CT - Slice {z}")

    axes[i, 1].imshow(
        volume[z], cmap="gray",
        vmin=-1000, vmax=400
    )
    axes[i, 1].imshow(
        lung_mask[z],
        cmap="autumn",
        alpha=0.4
    )
    axes[i, 1].set_title(f"Lung Segmentation - Slice {z}")

    axes[i, 0].axis("off")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, dpi=150)
plt.close()

print("Visualization saved:", OUTPUT_IMAGE)
print("\nPATIENT 106046 SEGMENTATION COMPLETED")
