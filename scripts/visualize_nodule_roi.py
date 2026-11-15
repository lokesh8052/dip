
import os
import numpy as np
import matplotlib.pyplot as plt

# Load reconstructed CT volume
volume = np.load(
    "data/processed/ct_volumes/patient_100012.npz"
)["volume"]

# Verified annotation coordinates
z, row, col = 125, 313, 199

# Extract a 96 x 96 ROI around the annotation
radius = 48

r1 = row - radius
r2 = row + radius
c1 = col - radius
c2 = col + radius

# Nine neighboring axial slices
slice_indices = range(z - 4, z + 5)

fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for ax, slice_idx in zip(axes.flat, slice_indices):

    roi = volume[
        slice_idx,
        r1:r2,
        c1:c2
    ]

    ax.imshow(
        roi,
        cmap="gray",
        vmin=-1000,
        vmax=400,
        origin="lower"
    )

    # Annotation location relative to ROI
    ax.scatter(
        col - c1,
        row - r1,
        color="red",
        marker="+",
        s=130,
        linewidths=2
    )

    ax.set_title(f"Axial Slice {slice_idx}")
    ax.axis("off")

fig.suptitle(
    "DIP-MONet: Nodule Region - Patient 100012",
    fontsize=16
)

plt.tight_layout()

os.makedirs("outputs", exist_ok=True)

output_path = "outputs/nodule_roi_patient_100012.png"

plt.savefig(
    output_path,
    dpi=160,
    bbox_inches="tight"
)

print("ROI visualization saved:", output_path)

plt.show()
