
import os
import numpy as np
from scipy import ndimage

# Load the existing filled lung mask
mask = np.load(
    "data/processed/ct_volumes/patient_100012_lung_region_mask.npz"
)["mask"]

# Create a separate output mask
separated_mask = np.zeros_like(mask, dtype=bool)

# Approximate center column
mid_col = mask.shape[2] // 2

structure = np.ones((3, 3), dtype=bool)

# Process every axial slice
for z in range(mask.shape[0]):

    # Process left and right halves independently
    for start_col, end_col in [
        (0, mid_col),
        (mid_col, mask.shape[2])
    ]:

        half_mask = mask[z, :, start_col:end_col]

        labels, count = ndimage.label(
            half_mask,
            structure=structure
        )

        if count == 0:
            continue

        sizes = np.bincount(labels.ravel())
        sizes[0] = 0

        largest_id = np.argmax(sizes)

        separated_mask[
            z, :, start_col:end_col
        ] = labels == largest_id

# Check annotation inclusion
z, row, col = 125, 313, 199

print("Original mask voxels:", mask.sum())
print("Separated mask voxels:", separated_mask.sum())

print(
    "Annotation included:",
    bool(separated_mask[z, row, col])
)

# Save separately
output_path = (
    "data/processed/ct_volumes/"
    "patient_100012_separated_lung_mask.npz"
)

os.makedirs("data/processed/ct_volumes", exist_ok=True)

np.savez_compressed(
    output_path,
    mask=separated_mask
)

print("\nSaved:", output_path)

# Check selected slices
print("\nSelected slice statistics:")

for z in [20, 50, 80, 100, 125, 150]:

    labels, count = ndimage.label(
        separated_mask[z],
        structure=structure
    )

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0

    largest = np.sort(sizes[1:])[::-1]

    print(f"\nSlice {z}:")
    print("Connected components:", count)
    print("Largest component sizes:", largest[:5])
