
import os
import numpy as np
from scipy import ndimage

# Load original CT volume
volume = np.load(
    "data/processed/ct_volumes/patient_100012.npz"
)["volume"]

# Load original 3D component labels
labels_3d = np.load(
    "data/processed/ct_volumes/patient_100012_3d_components.npz"
)["labels"]

# Start from original lung-air candidate
lung_air = labels_3d == 25

mid_col = volume.shape[2] // 2

# Separate sides before hole filling
refined_mask = np.zeros_like(lung_air, dtype=bool)

structure = np.ones((3, 3), dtype=bool)

for z in range(volume.shape[0]):

    for start_col, end_col in [
        (0, mid_col),
        (mid_col, volume.shape[2])
    ]:

        half_mask = lung_air[z, :, start_col:end_col]

        # Find connected air regions
        component_labels, count = ndimage.label(
            half_mask,
            structure=structure
        )

        if count == 0:
            continue

        sizes = np.bincount(component_labels.ravel())
        sizes[0] = 0

        largest_id = np.argmax(sizes)

        # Retain the largest air component
        largest_component = component_labels == largest_id

        # Fill holes only after separating the sides
        filled_component = ndimage.binary_fill_holes(
            largest_component
        )

        refined_mask[
            z, :, start_col:end_col
        ] = filled_component

# Check annotation
z, row, col = 125, 313, 199

print("Original air voxels:", lung_air.sum())
print("Refined mask voxels:", refined_mask.sum())
print("Annotation included:", bool(refined_mask[z, row, col]))

# Save as a new experiment
output_path = (
    "data/processed/ct_volumes/"
    "patient_100012_refined_lung_mask_v2.npz"
)

os.makedirs("data/processed/ct_volumes", exist_ok=True)

np.savez_compressed(
    output_path,
    mask=refined_mask
)

print("\nSaved:", output_path)

for z in [20, 50, 80, 100, 125, 150]:
    print(
        f"Slice {z}:",
        refined_mask[z].sum(),
        "voxels"
    )
