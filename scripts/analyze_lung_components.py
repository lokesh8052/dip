
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage

# Load refined lung mask
mask_path = (
    "data/processed/ct_volumes/"
    "patient_100012_lung_mask.npz"
)

mask = np.load(mask_path)["mask"].astype(bool)

print("Mask shape:", mask.shape)

# Use 26-connectivity in 3D
structure = np.ones((3, 3, 3), dtype=bool)

labels, num_components = ndimage.label(
    mask,
    structure=structure
)

print("\nTotal connected components:", num_components)

# Calculate component sizes
component_sizes = np.bincount(labels.ravel())

# Exclude background (label 0)
component_sizes[0] = 0

largest_labels = np.argsort(component_sizes)[::-1]

print("\nTop 15 components:")

for label_id in largest_labels[:15]:
    if component_sizes[label_id] == 0:
        continue

    print(
        f"Component {label_id}: "
        f"{component_sizes[label_id]:,} voxels"
    )

# Identify the component containing the annotation
z, row, col = 125, 313, 199

annotation_label = labels[z, row, col]

print("\nAnnotation component:", annotation_label)

if annotation_label != 0:
    print(
        "Annotation component size:",
        component_sizes[annotation_label]
    )
else:
    print("WARNING: Annotation is outside the foreground mask.")

# Display component labels at the annotation slice
plt.figure(figsize=(9, 8))

plt.imshow(
    labels[z],
    cmap="nipy_spectral",
    interpolation="nearest"
)

plt.scatter(
    col,
    row,
    color="red",
    marker="+",
    s=150
)

plt.title("3D Connected Components - Axial Slice 125")
plt.axis("off")

os.makedirs("outputs", exist_ok=True)

plt.savefig(
    "outputs/lung_components_patient_100012.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

# Save labels for further analysis
np.savez_compressed(
    "data/processed/ct_volumes/"
    "patient_100012_component_labels.npz",
    labels=labels
)

print("\nComponent analysis completed.")
