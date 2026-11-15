
import numpy as np
import matplotlib.pyplot as plt

# Load original CT
volume = np.load(
    "data/processed/ct_volumes/patient_100012.npz"
)["volume"]

# Load separated lung mask
mask = np.load(
    "data/processed/ct_volumes/patient_100012_separated_lung_mask.npz"
)["mask"]

slices = [20, 50, 80, 100, 125, 150]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for ax, z in zip(axes.flat, slices):

    # Display original CT
    ax.imshow(
        volume[z],
        cmap="gray",
        vmin=-1000,
        vmax=400,
        origin="lower"
    )

    # Overlay mask boundary
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
            s=150,
            label="Annotated nodule"
        )
        ax.legend(loc="upper right")

    ax.set_title(f"Separated Lung Mask - Slice {z}")
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    "outputs/separated_lung_mask_multislice.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("Mask shape:", mask.shape)
print("Total mask voxels:", mask.sum())
print("Annotation included:", bool(mask[125, 313, 199]))
