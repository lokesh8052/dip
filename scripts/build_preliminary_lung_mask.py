
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Step 1: Load original CT volume
volume = np.load(
    "data/processed/ct_volumes/patient_100012.npz"
)["volume"]

# Step 2: Load connected component labels
labels = np.load(
    "data/processed/ct_volumes/patient_100012_threshold_components.npz"
)["labels"]

# Step 3: Select internal lung air component
lung_air_mask = labels == 25

# Step 4: Fill enclosed holes slice by slice
preliminary_lung_mask = np.zeros_like(
    lung_air_mask,
    dtype=bool
)

for z in range(volume.shape[0]):
    preliminary_lung_mask[z] = ndimage.binary_fill_holes(
        lung_air_mask[z]
    )

# Step 5: Check annotated nodule location
annotation = (125, 313, 199)

print("Original lung air voxels:", lung_air_mask.sum())
print("Filled lung mask voxels:", preliminary_lung_mask.sum())

print(
    "Annotation included:",
    preliminary_lung_mask[annotation]
)

# Step 6: Save the preliminary mask separately
np.savez_compressed(
    "data/processed/ct_volumes/patient_100012_preliminary_lung_mask.npz",
    mask=preliminary_lung_mask
)

# Step 7: Visualize annotation slice
z = 125

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(
    volume[z],
    cmap="gray",
    vmin=-1000,
    vmax=400,
    origin="lower"
)
axes[0].set_title("Original CT")

axes[1].imshow(
    lung_air_mask[z],
    cmap="gray",
    origin="lower"
)
axes[1].set_title("Internal Lung Air")

axes[2].imshow(
    volume[z],
    cmap="gray",
    vmin=-1000,
    vmax=400,
    origin="lower"
)

axes[2].imshow(
    preliminary_lung_mask[z],
    cmap="autumn",
    alpha=0.35,
    origin="lower"
)
axes[2].set_title("Preliminary Lung Mask Overlay")

for ax in axes:
    ax.scatter(
        199,
        313,
        color="red",
        marker="+",
        s=150
    )
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    "outputs/preliminary_lung_mask.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("Preliminary lung mask saved successfully.")
