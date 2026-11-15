
import numpy as np
import matplotlib.pyplot as plt

# Load original CT volume
volume = np.load(
    "data/processed/ct_volumes/patient_100012.npz"
)["volume"]

# Load 3D component labels
labels = np.load(
    "data/processed/ct_volumes/patient_100012_3d_components.npz"
)["labels"]

# Annotated slice and location
z = 125
row = 313
col = 199

# Extract component masks
component_1 = labels[z] == 1
component_25 = labels[z] == 25

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original CT
axes[0].imshow(
    volume[z],
    cmap="gray",
    vmin=-1000,
    vmax=400,
    origin="lower"
)
axes[0].set_title("Original CT")

# Component 1
axes[1].imshow(
    component_1,
    cmap="gray",
    origin="lower"
)
axes[1].set_title("Component 1 - External Air Candidate")

# Component 25
axes[2].imshow(
    component_25,
    cmap="gray",
    origin="lower"
)
axes[2].set_title("Component 25 - Lung Air Candidate")

for ax in axes:
    ax.scatter(
        col,
        row,
        color="red",
        marker="+",
        s=150
    )
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    "outputs/verify_3d_components.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("Component 1 voxels on slice:", component_1.sum())
print("Component 25 voxels on slice:", component_25.sum())
print("Annotation label:", labels[z, row, col])
