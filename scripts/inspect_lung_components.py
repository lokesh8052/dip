
import numpy as np
from scipy import ndimage

path = (
    "data/processed/ct_volumes/"
    "patient_100012_component_labels.npz"
)

labels = np.load(path)["labels"]

components = [1, 22]

print("COMPONENT SPATIAL ANALYSIS")

for component_id in components:

    mask = labels == component_id

    coordinates = np.where(mask)

    z_min, z_max = coordinates[0].min(), coordinates[0].max()
    row_min, row_max = coordinates[1].min(), coordinates[1].max()
    col_min, col_max = coordinates[2].min(), coordinates[2].max()

    centroid = ndimage.center_of_mass(mask)

    print(f"\nComponent {component_id}")
    print("Voxel count:", mask.sum())

    print("Bounding box:")
    print("Z:", z_min, "to", z_max)
    print("Row:", row_min, "to", row_max)
    print("Column:", col_min, "to", col_max)

    print("Centroid (Z, Row, Column):", centroid)

# Verify annotation
annotation = (125, 313, 199)

print("\nAnnotation label:", labels[annotation])
