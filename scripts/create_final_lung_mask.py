
import os
import numpy as np
from scipy import ndimage

INPUT_PATH = (
    "data/processed/ct_volumes/"
    "patient_100012_refined_lung_mask_v2.npz"
)

OUTPUT_PATH = (
    "data/processed/ct_volumes/"
    "patient_100012_final_lung_mask.npz"
)

# Load refined mask
mask = np.load(INPUT_PATH)["mask"].astype(bool)

# Find 3D connected components
structure = np.ones((3, 3, 3), dtype=bool)

labels, count = ndimage.label(
    mask,
    structure=structure
)

sizes = np.bincount(labels.ravel())
sizes[0] = 0

# Retain the largest component
largest_id = np.argmax(sizes)

final_mask = labels == largest_id

# Verify annotation inclusion
annotation = (125, 313, 199)

print("Original components:", count)
print("Retained component:", largest_id)
print("Retained voxels:", int(final_mask.sum()))
print("Removed voxels:", int(mask.sum() - final_mask.sum()))
print("Annotation included:", bool(final_mask[annotation]))

# Save separately
os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

np.savez_compressed(
    OUTPUT_PATH,
    mask=final_mask
)

print("\nFinal lung mask saved:")
print(OUTPUT_PATH)
