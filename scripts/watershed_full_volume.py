
import os
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed

# Step 1: Load original CT volume
volume = np.load(
    "data/processed/ct_volumes/patient_100012.npz"
)["volume"]

# Load existing lung-region mask
mask_volume = np.load(
    "data/processed/ct_volumes/patient_100012_lung_region_mask.npz"
)["mask"]

print("CT shape:", volume.shape)
print("Input mask shape:", mask_volume.shape)

# Step 2: Create output label volume
# 0 = background
# 1 = left-side region
# 2 = right-side region

watershed_volume = np.zeros(
    volume.shape,
    dtype=np.uint8
)

mid_col = volume.shape[2] // 2

structure = np.ones((3, 3), dtype=bool)

missing_left = []
missing_right = []

# Step 3: Process every axial slice
for z in range(volume.shape[0]):

    lung_mask = mask_volume[z]

    if not np.any(lung_mask):
        missing_left.append(z)
        missing_right.append(z)
        continue

    # Calculate distance transform
    distance = ndimage.distance_transform_edt(
        lung_mask
    )

    # Find seeds independently in each half
    left_distance = distance.copy()
    left_distance[:, mid_col:] = 0

    right_distance = distance.copy()
    right_distance[:, :mid_col] = 0

    # Check whether each side has foreground
    if left_distance.max() == 0:
        missing_left.append(z)
        continue

    if right_distance.max() == 0:
        missing_right.append(z)
        continue

    left_seed = np.unravel_index(
        np.argmax(left_distance),
        distance.shape
    )

    right_seed = np.unravel_index(
        np.argmax(right_distance),
        distance.shape
    )

    # Create markers
    markers = np.zeros(
        lung_mask.shape,
        dtype=np.int32
    )

    markers[left_seed] = 1
    markers[right_seed] = 2

    # Apply watershed
    labels = watershed(
        -distance,
        markers,
        mask=lung_mask,
        connectivity=structure
    )

    watershed_volume[z] = labels.astype(np.uint8)

    # Progress report
    if z % 20 == 0:
        print(
            f"Processed slice {z}/{volume.shape[0]} | "
            f"Left: {np.count_nonzero(labels == 1)} | "
            f"Right: {np.count_nonzero(labels == 2)}"
        )

# Step 4: Calculate final statistics
left_total = np.count_nonzero(watershed_volume == 1)
right_total = np.count_nonzero(watershed_volume == 2)

print("\nFULL VOLUME RESULTS")
print("Left-side voxels:", left_total)
print("Right-side voxels:", right_total)
print("Total segmented voxels:", left_total + right_total)

print("\nSlices missing left region:", missing_left)
print("Slices missing right region:", missing_right)

# Step 5: Check actual annotation
z, row, col = 125, 313, 199

annotation_label = watershed_volume[z, row, col]

print("\nANNOTATION CHECK")
print("Original HU:", volume[z, row, col])
print("Watershed label:", annotation_label)
print("Annotation included:", annotation_label in (1, 2))

# Step 6: Save output separately
output_path = (
    "data/processed/ct_volumes/"
    "patient_100012_watershed_full.npz"
)

os.makedirs("data/processed/ct_volumes", exist_ok=True)

np.savez_compressed(
    output_path,
    labels=watershed_volume
)

print("\nSaved:", output_path)
print("Output shape:", watershed_volume.shape)
print("Output dtype:", watershed_volume.dtype)
