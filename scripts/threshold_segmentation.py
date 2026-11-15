
import os
import numpy as np
import matplotlib.pyplot as plt

# Load reconstructed CT volume
volume = np.load(
    "data/processed/ct_volumes/patient_100012.npz"
)["volume"]

print("Volume shape:", volume.shape)

# HU thresholding
threshold = -400

binary_mask = volume < threshold

print("Threshold:", threshold, "HU")
print("Mask shape:", binary_mask.shape)
print(
    "Percentage of voxels below threshold:",
    100 * binary_mask.mean()
)

# Known nodule location
z, row, col = 125, 313, 199

# Display original image and threshold mask
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Original axial CT
axes[0, 0].imshow(
    volume[z],
    cmap="gray",
    vmin=-1000,
    vmax=400
)
axes[0, 0].scatter(
    col, row, color="red", marker="+", s=100
)
axes[0, 0].set_title("Original CT")

# Threshold mask
axes[0, 1].imshow(
    binary_mask[z],
    cmap="gray"
)
axes[0, 1].scatter(
    col, row, color="red", marker="+", s=100
)
axes[0, 1].set_title("Threshold Mask")

# Original nodule ROI
radius = 48

roi = volume[
    z,
    row-radius:row+radius,
    col-radius:col+radius
]

axes[1, 0].imshow(
    roi,
    cmap="gray",
    vmin=-1000,
    vmax=400
)
axes[1, 0].set_title("Original Nodule ROI")

# Threshold ROI
mask_roi = binary_mask[
    z,
    row-radius:row+radius,
    col-radius:col+radius
]

axes[1, 1].imshow(
    mask_roi,
    cmap="gray"
)
axes[1, 1].set_title("Thresholded Nodule ROI")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()

os.makedirs("outputs", exist_ok=True)

plt.savefig(
    "outputs/threshold_segmentation_patient_100012.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("Visualization saved successfully.")
