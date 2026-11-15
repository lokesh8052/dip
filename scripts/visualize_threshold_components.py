
import numpy as np
import matplotlib.pyplot as plt

# Load original CT volume
volume = np.load(
    "data/processed/ct_volumes/patient_100012.npz"
)["volume"]

# Load connected component labels
labels = np.load(
    "data/processed/ct_volumes/patient_100012_threshold_components.npz"
)["labels"]

# Annotation slice
z = 125
row = 313
col = 199

# Extract masks
ct_slice = volume[z]
external_air = labels[z] == 1
internal_air = labels[z] == 25

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(
    ct_slice,
    cmap="gray",
    vmin=-1000,
    vmax=400,
    origin="lower"
)
axes[0].set_title("Original CT")

axes[1].imshow(
    external_air,
    cmap="gray",
    origin="lower"
)
axes[1].set_title("Component 1")

axes[2].imshow(
    internal_air,
    cmap="gray",
    origin="lower"
)
axes[2].set_title("Component 25")

# Mark annotation location
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
    "outputs/visualize_threshold_components.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("Component 1 pixels:", external_air.sum())
print("Component 25 pixels:", internal_air.sum())
print("Annotation label:", labels[z, row, col])
