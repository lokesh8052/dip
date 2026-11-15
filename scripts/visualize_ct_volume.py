
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load reconstructed CT volume
# --------------------------------------------------

volume_path = "data/processed/ct_volumes/patient_100012.npz"
metadata_path = "data/processed/ct_volumes/patient_100012.json"

volume = np.load(volume_path)["volume"]

with open(metadata_path, "r") as f:
    metadata = json.load(f)

print("CT volume loaded:", volume.shape)

# --------------------------------------------------
# 2. Known nodule voxel coordinates
# --------------------------------------------------

z = 125
row = 313
col = 199

print("Nodule location:", (z, row, col))

# --------------------------------------------------
# 3. Extract three anatomical views
# --------------------------------------------------

axial = volume[z, :, :]
coronal = volume[:, row, :]
sagittal = volume[:, :, col]

# --------------------------------------------------
# 4. Display using lung CT window
# --------------------------------------------------

vmin = -1000
vmax = 400

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Axial view
axes[0].imshow(
    axial,
    cmap="gray",
    vmin=vmin,
    vmax=vmax,
    origin="lower"
)
axes[0].scatter(
    col, row,
    color="red",
    marker="+",
    s=150,
    linewidths=2
)
axes[0].set_title(f"Axial View (Slice {z})")
axes[0].set_xlabel("Column")
axes[0].set_ylabel("Row")

# Coronal view
axes[1].imshow(
    coronal,
    cmap="gray",
    vmin=vmin,
    vmax=vmax,
    origin="lower",
    aspect="auto"
)
axes[1].scatter(
    col, z,
    color="red",
    marker="+",
    s=150,
    linewidths=2
)
axes[1].set_title(f"Coronal View (Row {row})")
axes[1].set_xlabel("Column")
axes[1].set_ylabel("Slice")

# Sagittal view
axes[2].imshow(
    sagittal,
    cmap="gray",
    vmin=vmin,
    vmax=vmax,
    origin="lower",
    aspect="auto"
)
axes[2].scatter(
    row, z,
    color="red",
    marker="+",
    s=150,
    linewidths=2
)
axes[2].set_title(f"Sagittal View (Column {col})")
axes[2].set_xlabel("Row")
axes[2].set_ylabel("Slice")

fig.suptitle(
    "DIP-MONet: Original CT Scan - Patient 100012",
    fontsize=15
)

plt.tight_layout()

# --------------------------------------------------
# 5. Save visualization
# --------------------------------------------------

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(
    output_dir,
    "ct_visualization_patient_100012.png"
)

plt.savefig(output_path, dpi=150, bbox_inches="tight")

print("Visualization saved:", output_path)

plt.show()
