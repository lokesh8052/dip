
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed

volume = np.load(
    "data/processed/ct_volumes/patient_100012.npz"
)["volume"]

mask_volume = np.load(
    "data/processed/ct_volumes/patient_100012_lung_region_mask.npz"
)["mask"]

# Actual annotated slice
z = 125
row, col = 313, 199

ct = volume[z]
lung_mask = mask_volume[z]

# Distance transform
distance = ndimage.distance_transform_edt(lung_mask)

mid_col = lung_mask.shape[1] // 2

left_distance = distance.copy()
left_distance[:, mid_col:] = 0

right_distance = distance.copy()
right_distance[:, :mid_col] = 0

left_seed = np.unravel_index(
    np.argmax(left_distance),
    distance.shape
)

right_seed = np.unravel_index(
    np.argmax(right_distance),
    distance.shape
)

markers = np.zeros(lung_mask.shape, dtype=np.int32)
markers[left_seed] = 1
markers[right_seed] = 2

labels = watershed(
    -distance,
    markers,
    mask=lung_mask,
    connectivity=np.ones((3, 3), dtype=bool)
)

print("Actual annotation inspection - Slice 125")

print("Annotation HU:", ct[row, col])
print("Inside input mask:", bool(lung_mask[row, col]))
print("Watershed label:", labels[row, col])

print("\nLeft seed:", left_seed)
print("Right seed:", right_seed)

print("\nLeft region voxels:", np.count_nonzero(labels == 1))
print("Right region voxels:", np.count_nonzero(labels == 2))

# Visualize
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(
    ct,
    cmap="gray",
    vmin=-1000,
    vmax=400,
    origin="lower"
)
axes[0].scatter(
    col, row,
    color="red",
    marker="+",
    s=150
)
axes[0].set_title("Actual Nodule Annotation")

axes[1].imshow(
    labels,
    cmap="nipy_spectral",
    origin="lower"
)
axes[1].scatter(
    col, row,
    color="red",
    marker="+",
    s=150
)
axes[1].set_title("Watershed Labels - Slice 125")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    "outputs/watershed_annotation_slice_125.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()
