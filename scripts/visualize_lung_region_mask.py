
import numpy as np
import matplotlib.pyplot as plt

# Load original CT volume
volume = np.load(
    "data/processed/ct_volumes/patient_100012.npz"
)["volume"]

# Load newly generated lung-region mask
mask = np.load(
    "data/processed/ct_volumes/patient_100012_lung_region_mask.npz"
)["mask"]

# Slices to inspect
slice_indices = [20, 50, 80, 100, 125, 150]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for ax, z in zip(axes.flat, slice_indices):

    # Display original CT
    ax.imshow(
        volume[z],
        cmap="gray",
        vmin=-1000,
        vmax=400,
        origin="lower"
    )

    # Overlay lung-region mask boundary
    ax.contour(
        mask[z].astype(float),
        levels=[0.5],
        colors="lime",
        linewidths=0.8,
        origin="lower"
    )

    # Show annotation only on slice 125
    if z == 125:
        ax.scatter(
            199,
            313,
            color="red",
            marker="+",
            s=150,
            label="Annotated nodule"
        )
        ax.legend(loc="upper right")

    ax.set_title(f"Axial Slice {z}")
    ax.axis("off")

plt.suptitle(
    "Lung Region Mask - Multiple CT Slices",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    "outputs/lung_region_mask_multislice.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("Mask shape:", mask.shape)
print("Mask dtype:", mask.dtype)
print("Total mask voxels:", mask.sum())
print("Annotation included:", mask[125, 313, 199])
