
import numpy as np
from scipy import ndimage

BASE = "data/processed/ct_volumes/"

# Load both masks
preliminary = np.load(
    BASE + "patient_100012_preliminary_lung_mask.npz"
)["mask"].astype(bool)

refined = np.load(
    BASE + "patient_100012_refined_lung_mask_v2.npz"
)["mask"].astype(bool)

# Known annotation location
annotation = (125, 313, 199)

print("========== MASK COMPARISON ==========")

print("Preliminary shape:", preliminary.shape)
print("Refined shape:", refined.shape)

print("\nTotal foreground voxels:")
print("Preliminary:", int(preliminary.sum()))
print("Refined:", int(refined.sum()))

print("\nAnnotation inclusion:")
print("Preliminary:", bool(preliminary[annotation]))
print("Refined:", bool(refined[annotation]))

# Compare overlap
intersection = np.logical_and(
    preliminary, refined
).sum()

union = np.logical_or(
    preliminary, refined
).sum()

iou = intersection / union if union else 0

print("\nIntersection over Union:", round(float(iou), 4))

# 3D connected components
structure = np.ones((3, 3, 3), dtype=bool)

for name, mask in [
    ("Preliminary", preliminary),
    ("Refined V2", refined)
]:
    labels, count = ndimage.label(
        mask,
        structure=structure
    )

    sizes = np.bincount(labels.ravel())[1:]

    largest = np.sort(sizes)[::-1]

    print(f"\n{name} 3D components:", count)
    print("Largest component sizes:", largest[:10])

print("\n========== COMPARISON COMPLETED ==========")
