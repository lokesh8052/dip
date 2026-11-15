
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from skimage.segmentation import watershed

# Load original CT
volume = np.load(
    "data/processed/ct_volumes/patient_100012.npz"
)["volume"]

# Load the existing filled lung-region mask
mask_volume = np.load(
    "data/processed/ct_volumes/patient_100012_lung_region_mask.npz"
)["mask"]

# Work with axial slice 100
z = 100

ct_slice = volume[z]
lung_mask = mask_volume[z]

# Step 1: Calculate distance from the mask boundary
distance = ndimage.distance_transform_edt(lung_mask)

# Step 2: Find one seed inside each half
mid_col = lung_mask.shape[1] // 2

left_distance = distance.copy()
left_distance[:, mid_col:] = 0

right_distance = distance.copy()
right_distance[:, :mid_col] = 0

left_seed = np.unravel_index(
    np.argmax(left_distance),
    left_distance.shape
)

right_seed = np.unravel_index(
    np.argmax(right_distance),
    right_distance.shape
)

# Step 3: Create marker image
markers = np.zeros(lung_mask.shape, dtype=np.int32)

markers[left_seed] = 1
markers[right_seed] = 2

print("Left seed (row, col):", left_seed)
print("Right seed (row, col):", right_seed)

# Step 4: Apply marker-controlled watershed
watershed_labels = watershed(
    -distance,
    markers,
    mask=lung_mask,
    connectivity=np.ones((3, 3), dtype=bool)
)

# Step 5: Create two region masks
left_lung = watershed_labels == 1
right_lung = watershed_labels == 2

print("\nWatershed results:")
print("Left region voxels:", left_lung.sum())
print("Right region voxels:", right_lung.sum())
print("Total segmented voxels:", (left_lung | right_lung).sum())

# Step 6: Visualize results
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(
    ct_slice,
    cmap="gray",
    vmin=-1000,
    vmax=400,
    origin="lower"
)
axes[0].set_title("Original CT - Slice 100")

axes[1].imshow(
    lung_mask,
    cmap="gray",
    origin="lower"
)
axes[1].scatter(
    left_seed[1], left_seed[0],
    color="red", marker="+", s=120
)
axes[1].scatter(
    right_seed[1], right_seed[0],
    color="blue", marker="+", s=120
)
axes[1].set_title("Input Mask and Seeds")

axes[2].imshow(
    ct_slice,
    cmap="gray",
    vmin=-1000,
    vmax=400,
    origin="lower"
)
axes[2].contour(
    left_lung.astype(float),
    levels=[0.5],
    colors="lime",
    linewidths=1,
    origin="lower"
)
axes[2].contour(
    right_lung.astype(float),
    levels=[0.5],
    colors="cyan",
    linewidths=1,
    origin="lower"
)
axes[2].set_title("Watershed Separation")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    "outputs/watershed_slice_100.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("\nSaved: outputs/watershed_slice_100.png")