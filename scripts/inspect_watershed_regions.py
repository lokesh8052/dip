
import numpy as np

from scipy import ndimage
from skimage.segmentation import watershed

# Load CT and input mask
volume = np.load(
    "data/processed/ct_volumes/patient_100012.npz"
)["volume"]

mask = np.load(
    "data/processed/ct_volumes/patient_100012_lung_region_mask.npz"
)["mask"]

z = 100
ct = volume[z]
lung_mask = mask[z]

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

# Inspect selected anatomical locations
points = {
    "Left lung seed": left_seed,
    "Right lung seed": right_seed,
    "Image center": (256, 256),
    "Central airway": (270, 256),
    "Heart region": (340, 256),
    "Annotated nodule": (313, 199)
}

print("Watershed inspection - Slice 100")

for name, (row, col) in points.items():
    print(
        f"{name}: "
        f"HU={ct[row, col]:.1f}, "
        f"input_mask={bool(lung_mask[row, col])}, "
        f"watershed_label={labels[row, col]}"
    )

print("\nLabel voxel counts:")

for label in [0, 1, 2]:
    print(
        f"Label {label}:",
        np.count_nonzero(labels == label)
    )

print("\nInspection completed.")
