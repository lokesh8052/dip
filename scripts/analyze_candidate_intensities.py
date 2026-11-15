
import numpy as np
import matplotlib.pyplot as plt
import os

# Load original CT volume
volume = np.load(
    "data/processed/ct_volumes/patient_100012.npz"
)["volume"]

# Load watershed labels
labels = np.load(
    "data/processed/ct_volumes/patient_100012_watershed_full.npz"
)["labels"]

# Select voxels inside either lung region
lung_mask = (labels == 1) | (labels == 2)

lung_hu = volume[lung_mask]

print("CANDIDATE INTENSITY ANALYSIS")
print("Total lung-region voxels:", lung_hu.size)

# Calculate intensity statistics
print("\nHU statistics:")
print("Minimum:", lung_hu.min())
print("Maximum:", lung_hu.max())
print("Mean:", lung_hu.mean())
print("Median:", np.median(lung_hu))
print("Standard deviation:", lung_hu.std())

# Percentiles
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

print("\nHU percentiles:")

for p in percentiles:
    print(
        f"{p}th percentile:",
        np.percentile(lung_hu, p)
    )

# Intensity ranges
ranges = [
    (-1000, -800),
    (-800, -600),
    (-600, -400),
    (-400, -200),
    (-200, 0),
    (0, 200),
    (200, 400),
    (400, 1000)
]

print("\nIntensity range distribution:")

for low, high in ranges:
    count = np.count_nonzero(
        (lung_hu >= low) & (lung_hu < high)
    )

    percentage = 100 * count / lung_hu.size

    print(
        f"{low:5d} to {high:5d} HU: "
        f"{count:10d} voxels "
        f"({percentage:.2f}%)"
    )

# Plot intensity histogram
os.makedirs("outputs", exist_ok=True)

plt.figure(figsize=(12, 6))

plt.hist(
    lung_hu,
    bins=150,
    range=(-1000, 1000)
)

plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Voxel Count")
plt.title("CT Intensity Distribution Inside Watershed Regions")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "outputs/lung_intensity_distribution.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("\nSaved: outputs/lung_intensity_distribution.png")
