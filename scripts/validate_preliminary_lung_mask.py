
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load original CT volume
# --------------------------------------------------

volume = np.load(
    "data/processed/ct_volumes/patient_100012.npz"
)["volume"]

# --------------------------------------------------
# 2. Load preliminary lung mask
# --------------------------------------------------

mask = np.load(
    "data/processed/ct_volumes/patient_100012_preliminary_lung_mask.npz"
)["mask"]

# --------------------------------------------------
# 3. Basic validation
# --------------------------------------------------

print("========== BASIC VALIDATION ==========")

print("CT volume shape:", volume.shape)
print("Lung mask shape:", mask.shape)
print("Mask data type:", mask.dtype)

total_mask_voxels = int(mask.sum())
total_ct_voxels = volume.size

print("\nTotal mask voxels:", total_mask_voxels)
print("Total CT voxels:", total_ct_voxels)

mask_percentage = (
    total_mask_voxels / total_ct_voxels
) * 100

print("Mask percentage:", round(mask_percentage, 2), "%")

# --------------------------------------------------
# 4. Check annotation inclusion
# --------------------------------------------------

annotation = (125, 313, 199)

print("\n========== ANNOTATION CHECK ==========")

print(
    "Annotation coordinates:",
    annotation
)

print(
    "Annotation included:",
    bool(mask[annotation])
)

# --------------------------------------------------
# 5. Slice-wise mask statistics
# --------------------------------------------------

slice_counts = mask.sum(axis=(1, 2))

non_empty_slices = np.count_nonzero(slice_counts)

print("\n========== SLICE STATISTICS ==========")

print("Total slices:", volume.shape[0])
print("Non-empty slices:", non_empty_slices)
print("Empty slices:", volume.shape[0] - non_empty_slices)

print("Minimum mask voxels:", int(slice_counts.min()))
print("Maximum mask voxels:", int(slice_counts.max()))

# --------------------------------------------------
# 6. Select 12 slices across the CT volume
# --------------------------------------------------

slice_indices = np.linspace(
    0,
    volume.shape[0] - 1,
    12,
    dtype=int
)

# --------------------------------------------------
# 7. Visualize the segmentation
# --------------------------------------------------

fig, axes = plt.subplots(
    3,
    4,
    figsize=(16, 12)
)

for ax, z in zip(axes.ravel(), slice_indices):

    # Display original CT
    ax.imshow(
        volume[z],
        cmap="gray",
        vmin=-1000,
        vmax=400,
        origin="lower"
    )

    # Overlay lung mask
    ax.imshow(
        mask[z],
        cmap="autumn",
        alpha=0.35,
        origin="lower"
    )

    ax.set_title(
        f"Slice {z} | Voxels: {int(slice_counts[z])}"
    )

    ax.axis("off")

plt.tight_layout()

# --------------------------------------------------
# 8. Save visualization
# --------------------------------------------------

plt.savefig(
    "outputs/preliminary_lung_mask_validation.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("\nValidation visualization saved successfully.")
print(
    "Output: outputs/preliminary_lung_mask_validation.png"
)
