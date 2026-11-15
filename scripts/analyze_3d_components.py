
import os
import numpy as np
from scipy import ndimage

# Load original CT volume
volume = np.load(
    "data/processed/ct_volumes/patient_100012.npz"
)["volume"]

print("CT volume shape:", volume.shape)

# Step 1: Threshold the original CT
threshold_mask = volume < -400

print("Threshold foreground voxels:", threshold_mask.sum())

# Step 2: Create 3D connectivity structure
structure = np.ones((3, 3, 3), dtype=bool)

# Step 3: Label connected components in 3D
labels, num_components = ndimage.label(
    threshold_mask,
    structure=structure
)

print("\nTotal 3D components:", num_components)

# Step 4: Count component sizes
component_sizes = np.bincount(labels.ravel())

# Ignore background label 0
component_sizes[0] = 0

largest_components = np.argsort(component_sizes)[::-1][:10]

print("\nTop 10 components:")
for rank, component_id in enumerate(largest_components, start=1):
    print(
        f"{rank}. Component {component_id}: "
        f"{component_sizes[component_id]} voxels"
    )

# Step 5: Check the annotated nodule location
z, row, col = 125, 313, 199

annotation_component = labels[z, row, col]

print("\nAnnotation component:", annotation_component)
print(
    "Annotation component size:",
    component_sizes[annotation_component]
)

# Step 6: Find components touching the volume boundaries
boundary_labels = np.unique(np.concatenate([
    labels[0, :, :].ravel(),
    labels[-1, :, :].ravel(),
    labels[:, 0, :].ravel(),
    labels[:, -1, :].ravel(),
    labels[:, :, 0].ravel(),
    labels[:, :, -1].ravel()
]))

print("\nNon-background components touching boundaries:")
for component_id in boundary_labels:
    if component_id != 0:
        print(
            f"Component {component_id}: "
            f"{component_sizes[component_id]} voxels"
        )

# Step 7: Save results separately
os.makedirs("data/processed/ct_volumes", exist_ok=True)

np.savez_compressed(
    "data/processed/ct_volumes/patient_100012_3d_components.npz",
    labels=labels
)

print("\nSaved 3D component labels successfully.")
