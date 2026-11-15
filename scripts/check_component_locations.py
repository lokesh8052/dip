
import numpy as np

path = (
    "data/processed/ct_volumes/"
    "patient_100012_component_labels.npz"
)

labels = np.load(path)["labels"]

z = 125

# Inspect labels at selected points.
# Coordinates are (row, column).
points = {
    "Image center": (256, 256),
    "Upper left region": (200, 150),
    "Upper right region": (200, 350),
    "Lower left region": (350, 150),
    "Lower right region": (350, 350),
    "Annotated nodule": (313, 199),
    "Top image boundary": (5, 256),
    "Left image boundary": (256, 5),
    "Right image boundary": (256, 506),
}

print(f"Component labels at axial slice {z}\n")

for name, (row, col) in points.items():
    print(
        f"{name:24s} "
        f"row={row:3d}, col={col:3d} "
        f"label={labels[z, row, col]}"
    )

print("\nUnique labels at this slice:")
print(np.unique(labels[z]))
