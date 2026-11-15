
import numpy as np
from scipy import ndimage

# Load the filled lung-region mask
mask = np.load(
    "data/processed/ct_volumes/patient_100012_lung_region_mask.npz"
)["mask"]

# Slices to inspect
slice_indices = [20, 50, 80, 100, 125, 150]

structure = np.ones((3, 3), dtype=bool)

print("Lung-region mask shape:", mask.shape)

for z in slice_indices:

    # Label 2D connected components
    labels, count = ndimage.label(
        mask[z],
        structure=structure
    )

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0

    largest = np.sort(sizes[1:])[::-1]

    print(f"\nSlice {z}")
    print("Total connected components:", count)
    print("Largest component sizes:", largest[:10])

# Verify annotation inclusion
z, row, col = 125, 313, 199

print("\nAnnotation check:")
print("Mask includes annotation:", bool(mask[z, row, col]))

print("\nAnalysis completed.")
