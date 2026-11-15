
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Load CT and lung-region mask
volume = np.load(
    "data/processed/ct_volumes/patient_100012.npz"
)["volume"]

mask = np.load(
    "data/processed/ct_volumes/patient_100012_lung_region_mask.npz"
)["mask"]

slices = [80, 100, 125]

fig, axes = plt.subplots(3, 2, figsize=(12, 16))

structure = np.ones((3, 3), dtype=bool)

for i, z in enumerate(slices):

    # Label connected components on this slice
    labels, count = ndimage.label(
        mask[z],
        structure=structure
    )

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0

    # Keep only components with at least 100 voxels
    filtered_labels = np.zeros_like(labels)

    for component_id in range(1, count + 1):
        if sizes[component_id] >= 100:
            filtered_labels[labels == component_id] = component_id

    # Original CT
    axes[i, 0].imshow(
        volume[z],
        cmap="gray",
        vmin=-1000,
        vmax=400,
        origin="lower"
    )
    axes[i, 0].set_title(f"Original CT - Slice {z}")

    # Colored connected components
    axes[i, 1].imshow(
        filtered_labels,
        cmap="nipy_spectral",
        origin="lower",
        interpolation="nearest"
    )
    axes[i, 1].set_title(
        f"Connected Components - Slice {z} ({count} total)"
    )

    if z == 125:
        axes[i, 0].scatter(
            199, 313,
            color="red",
            marker="+",
            s=150
        )

    for ax in axes[i]:
        ax.axis("off")

plt.tight_layout()

plt.savefig(
    "outputs/lung_components_2d_comparison.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("Visualization saved.")
