
import numpy as np
import matplotlib.pyplot as plt

# Load original CT
volume = np.load(
    "data/processed/ct_volumes/patient_100012.npz"
)["volume"]

# Load full-volume watershed labels
labels = np.load(
    "data/processed/ct_volumes/patient_100012_watershed_full.npz"
)["labels"]

# Selected axial slices
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

    # Label 1 boundary
    ax.contour(
        (labels[z] == 1).astype(float),
        levels=[0.5],
        colors="lime",
        linewidths=0.9,
        origin="lower"
    )

    # Label 2 boundary
    ax.contour(
        (labels[z] == 2).astype(float),
        levels=[0.5],
        colors="cyan",
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

    ax.set_title(f"Watershed - Slice {z}")
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    "outputs/watershed_full_volume_axial.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

# Inspect a sagittal section
col = 199

fig, ax = plt.subplots(figsize=(12, 7))

ax.imshow(
    volume[:, :, col],
    cmap="gray",
    vmin=-1000,
    vmax=400,
    origin="lower",
    aspect="auto"
)

ax.contour(
    (labels[:, :, col] == 1).astype(float),
    levels=[0.5],
    colors="lime",
    linewidths=0.7,
    origin="lower"
)

ax.contour(
    (labels[:, :, col] == 2).astype(float),
    levels=[0.5],
    colors="cyan",
    linewidths=0.7,
    origin="lower"
)

ax.set_title(f"Sagittal View - Column {col}")
ax.set_xlabel("Column position")
ax.set_ylabel("Axial slice index")

plt.tight_layout()

plt.savefig(
    "outputs/watershed_sagittal.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("Axial visualization saved.")
print("Sagittal visualization saved.")
