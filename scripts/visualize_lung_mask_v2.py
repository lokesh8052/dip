
import numpy as np
import matplotlib.pyplot as plt

volume = np.load(
    "data/processed/ct_volumes/patient_100012.npz"
)["volume"]

mask = np.load(
    "data/processed/ct_volumes/patient_100012_refined_lung_mask_v2.npz"
)["mask"]

slices = [20, 50, 80, 100, 125, 150]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for ax, z in zip(axes.flat, slices):

    ax.imshow(
        volume[z],
        cmap="gray",
        vmin=-1000,
        vmax=400,
        origin="lower"
    )

    ax.contour(
        mask[z].astype(float),
        levels=[0.5],
        colors="lime",
        linewidths=0.9,
        origin="lower"
    )

    if z == 125:
        ax.scatter(
            199, 313,
            color="red",
            marker="+",
            s=150
        )

    ax.set_title(f"Version 2 - Slice {z}")
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    "outputs/refined_lung_mask_v2.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("Mask shape:", mask.shape)
print("Total voxels:", mask.sum())
print("Annotation included:", bool(mask[125, 313, 199]))
