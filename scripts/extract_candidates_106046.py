
import os
import csv
import numpy as np
from scipy import ndimage

VOLUME_PATH = "data/processed/ct_volumes/patient_106046.npz"
MASK_PATH = "data/processed/ct_volumes/patient_106046_lung_mask.npz"

OUTPUT_DIR = "data/processed/ct_candidates/patient_106046"

THRESHOLD = -300
MIN_SIZE = 8
MAX_SIZE = 100000

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading CT volume and lung mask...")
volume = np.load(VOLUME_PATH)["volume"]
lung_mask = np.load(MASK_PATH)["mask"].astype(bool)

if volume.shape != lung_mask.shape:
    raise ValueError("Volume and mask shapes do not match.")

print("Volume:", volume.shape)
print("Lung voxels:", int(lung_mask.sum()))

print("\nGenerating candidate voxels...")
candidate_mask = (volume > THRESHOLD) & lung_mask

print("Candidate voxels:", int(candidate_mask.sum()))

print("\nFinding 3D connected components...")
structure = np.ones((3, 3, 3), dtype=bool)

labels, count = ndimage.label(
    candidate_mask,
    structure=structure
)

print("Total components:", count)

sizes = np.bincount(labels.ravel())
sizes[0] = 0

valid_ids = np.where(
    (sizes >= MIN_SIZE) & (sizes <= MAX_SIZE)
)[0]

print("Filtered candidates:", len(valid_ids))

# Sort candidates by size, largest first
valid_ids = valid_ids[
    np.argsort(sizes[valid_ids])[::-1]
]

rows = []

for rank, component_id in enumerate(valid_ids, start=1):
    coords = np.where(labels == component_id)

    center = (
        int(np.mean(coords[0])),
        int(np.mean(coords[1])),
        int(np.mean(coords[2]))
    )

    rows.append({
        "candidate_rank": rank,
        "component_id": int(component_id),
        "component_size": int(sizes[component_id]),
        "z": center[0],
        "row": center[1],
        "col": center[2]
    })

    if rank <= 20:
        print(
            f"{rank}. ID={component_id}, "
            f"size={sizes[component_id]}, center={center}"
        )

csv_path = os.path.join(OUTPUT_DIR, "candidates.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "candidate_rank",
        "component_id",
        "component_size",
        "z",
        "row",
        "col"
    ])
    writer.writeheader()
    writer.writerows(rows)

np.save(
    os.path.join(OUTPUT_DIR, "candidate_labels.npy"),
    labels.astype(np.int32)
)

print("\nCANDIDATE EXTRACTION COMPLETED")
print("Candidate CSV:", csv_path)
print("Candidate labels saved.")
