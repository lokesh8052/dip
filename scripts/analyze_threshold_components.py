
import numpy as np
from scipy import ndimage

# Load original CT volume
volume = np.load(
    "data/processed/ct_volumes/patient_100012.npz"
)["volume"]

# Original threshold mask
mask = volume < -400

print("Volume shape:", volume.shape)
print("Threshold foreground voxels:", mask.sum())

# Label 3D connected components using 26-connectivity
structure = np.ones((3, 3, 3), dtype=bool)

labels, total_components = ndimage.label(
    mask,
    structure=structure
)

print("\nTotal connected components:", total_components)

# Count voxels in each component
component_sizes = np.bincount(labels.ravel())

# Find largest components, excluding background label 0
largest = np.argsort(component_sizes[1:])[::-1][:10] + 1

print("\nLargest components:")

for component_id in largest:
    size = component_sizes[component_id]

    coords = np.where(labels == component_id)

    z_min, z_max = coords[0].min(), coords[0].max()
    row_min, row_max = coords[1].min(), coords[1].max()
    col_min, col_max = coords[2].min(), coords[2].max()

    print(
        f"Component {component_id}: "
        f"size={size}, "
        f"Z=({z_min},{z_max}), "
        f"Row=({row_min},{row_max}), "
        f"Col=({col_min},{col_max})"
    )

# Check annotation location
annotation = (125, 313, 199)

annotation_component = labels[annotation]

print("\nAnnotation component:", annotation_component)

if annotation_component != 0:
    print(
        "Annotation component size:",
        component_sizes[annotation_component]
    )

# Check whether each component touches any volume boundary
print("\nBoundary-touching components:")

boundary_labels = np.unique(np.concatenate([
    labels[0, :, :].ravel(),
    labels[-1, :, :].ravel(),
    labels[:, 0, :].ravel(),
    labels[:, -1, :].ravel(),
    labels[:, :, 0].ravel(),
    labels[:, :, -1].ravel()
]))

for component_id in boundary_labels:
    if component_id != 0:
        print(
            f"Component {component_id}: "
            f"size={component_sizes[component_id]}"
        )

# Save component labels separately
np.savez_compressed(
    "data/processed/ct_volumes/patient_100012_threshold_components.npz",
    labels=labels
)

print("\nComponent labels saved successfully.")
