
import numpy as np
from scipy import ndimage

path = (
    "data/processed/ct_volumes/"
    "patient_100012_refined_lung_mask_v2.npz"
)

mask = np.load(path)["mask"].astype(bool)

structure = np.ones((3, 3, 3), dtype=bool)

labels, count = ndimage.label(
    mask,
    structure=structure
)

print("Total components:", count)

for component_id in range(1, count + 1):

    component = labels == component_id
    coords = np.where(component)

    print(f"\nComponent {component_id}")
    print("Voxel count:", int(component.sum()))

    print(
        "Z range:",
        int(coords[0].min()),
        "to",
        int(coords[0].max())
    )

    print(
        "Row range:",
        int(coords[1].min()),
        "to",
        int(coords[1].max())
    )

    print(
        "Column range:",
        int(coords[2].min()),
        "to",
        int(coords[2].max())
    )

print("\nComponent inspection completed.")
