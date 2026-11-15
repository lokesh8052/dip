
import os
import numpy as np
from scipy import ndimage

# Load original CT volume
volume = np.load(
    "data/processed/ct_volumes/patient_100012.npz"
)["volume"]

# Load 3D connected-component labels
labels = np.load(
    "data/processed/ct_volumes/patient_100012_3d_components.npz"
)["labels"]

# Component 25 is our lung-air candidate
lung_air_mask = labels == 25

print("Lung-air voxels:", lung_air_mask.sum())

# Fill enclosed structures slice by slice
lung_region_mask = np.zeros_like(lung_air_mask, dtype=bool)

for z in range(volume.shape[0]):
    lung_region_mask[z] = ndimage.binary_fill_holes(
        lung_air_mask[z]
    )

# Annotation location
z, row, col = 125, 313, 199

print("\nAnnotation location:")
print("Original HU:", volume[z, row, col])
print("Inside lung-air mask:", lung_air_mask[z, row, col])
print("Inside filled lung-region mask:", lung_region_mask[z, row, col])

# Save as a separate experimental mask
output_path = (
    "data/processed/ct_volumes/"
    "patient_100012_lung_region_mask.npz"
)

os.makedirs("data/processed/ct_volumes", exist_ok=True)

np.savez_compressed(
    output_path,
    mask=lung_region_mask
)

print("\nLung-region mask saved:", output_path)
print("Total lung-region voxels:", lung_region_mask.sum())

# Slice-wise statistics
print("\nSelected slice statistics:")

for slice_index in [20, 50, 80, 100, 125, 150]:
    print(
        f"Slice {slice_index}:",
        lung_region_mask[slice_index].sum(),
        "voxels"
    )
