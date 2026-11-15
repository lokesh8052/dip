
import numpy as np
from scipy import ndimage

# Load original CT and 3D component labels
volume = np.load(
    "data/processed/ct_volumes/patient_100012.npz"
)["volume"]

labels = np.load(
    "data/processed/ct_volumes/patient_100012_3d_components.npz"
)["labels"]

# Inspect the two largest components
component_ids = [1, 25]

print("CT shape:", volume.shape)

# Get bounding boxes of all components
objects = ndimage.find_objects(labels)

for component_id in component_ids:
    component_slice = objects[component_id - 1]

    print(f"\n{'=' * 45}")
    print(f"COMPONENT {component_id}")

    if component_slice is None:
        print("Component not found.")
        continue

    z_range, row_range, col_range = component_slice

    print(
        "Z range:",
        z_range.start,
        "to",
        z_range.stop - 1
    )

    print(
        "Row range:",
        row_range.start,
        "to",
        row_range.stop - 1
    )

    print(
        "Column range:",
        col_range.start,
        "to",
        col_range.stop - 1
    )

    # Count component voxels in selected slices
    print("\nVoxel count at selected slices:")

    for z in [20, 50, 80, 100, 125, 150]:
        count = np.count_nonzero(labels[z] == component_id)
        print(f"Slice {z}: {count}")

# Inspect annotated nodule voxel
z, row, col = 125, 313, 199

print("\n" + "=" * 45)
print("ANNOTATION INSPECTION")

print("Annotation coordinates (z, row, col):", (z, row, col))
print("HU at annotation:", volume[z, row, col])
print("Threshold condition:", volume[z, row, col] < -400)
print("3D component label:", labels[z, row, col])

# Check a small neighborhood around annotation
print("\nNeighborhood HU statistics:")

roi = volume[
    z-2:z+3,
    row-2:row+3,
    col-2:col+3
]

print("Minimum HU:", roi.min())
print("Maximum HU:", roi.max())
print("Mean HU:", roi.mean())

print("\nInspection completed.")
