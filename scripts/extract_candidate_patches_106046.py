
import os
import csv
import numpy as np

VOLUME_PATH = "data/processed/ct_volumes/patient_106046.npz"
CANDIDATE_CSV = "data/processed/ct_candidates/patient_106046/candidates.csv"
OUTPUT_DIR = "data/processed/ct_candidates/patient_106046/patches"

PATCH_D, PATCH_H, PATCH_W = 64, 128, 128
TOP_K = 50

# Held-out annotation, converted from physical LPS coordinates
# to resampled voxel coordinates (Z, Y, X).
ANNOTATION = np.array([93, 251, 394])

os.makedirs(OUTPUT_DIR, exist_ok=True)

volume = np.load(VOLUME_PATH)["volume"]

with open(CANDIDATE_CSV, newline="") as f:
    candidates = list(csv.DictReader(f))

candidates = candidates[:TOP_K]

def extract_patch(volume, center):
    cz, cy, cx = center

    starts = [
        cz - PATCH_D // 2,
        cy - PATCH_H // 2,
        cx - PATCH_W // 2
    ]

    ends = [
        starts[0] + PATCH_D,
        starts[1] + PATCH_H,
        starts[2] + PATCH_W
    ]

    padding = [
        (max(0, -start), max(0, end - dimension))
        for start, end, dimension in zip(
            starts, ends, volume.shape
        )
    ]

    padded = np.pad(
        volume,
        padding,
        mode="constant",
        constant_values=-1000
    )

    adjusted = [
        starts[i] + padding[i][0]
        for i in range(3)
    ]

    return padded[
        adjusted[0]:adjusted[0] + PATCH_D,
        adjusted[1]:adjusted[1] + PATCH_H,
        adjusted[2]:adjusted[2] + PATCH_W
    ].astype(np.int16)

metadata = []

print("Extracting Patient 106046 candidate patches...")

for index, candidate in enumerate(candidates):
    rank = int(candidate["candidate_rank"])
    component_id = int(candidate["component_id"])
    size = int(candidate["component_size"])

    center = (
        int(candidate["z"]),
        int(candidate["row"]),
        int(candidate["col"])
    )

    patch = extract_patch(volume, center)

    filename = f"candidate_{rank:03d}.npy"
    np.save(os.path.join(OUTPUT_DIR, filename), patch)

    # Evaluation only: check whether annotation falls inside patch.
    offset = ANNOTATION - np.array(center)

    annotation_inside = bool(
        np.all(np.abs(offset) <= np.array([
            PATCH_D // 2,
            PATCH_H // 2,
            PATCH_W // 2
        ]))
    )

    metadata.append([
        rank,
        component_id,
        size,
        *center,
        int(annotation_inside)
    ])

    print(
        f"Candidate {rank}: center={center}, "
        f"annotation_inside={annotation_inside}"
    )

metadata = np.array(metadata, dtype=np.int64)

np.save(
    os.path.join(OUTPUT_DIR, "patch_metadata.npy"),
    metadata
)

print("\nPATCH EXTRACTION COMPLETED")
print("Patches:", len(metadata))
print("Patch shape:", (PATCH_D, PATCH_H, PATCH_W))
print("Annotation-containing patches:", int(metadata[:, 6].sum()))
print("Saved to:", OUTPUT_DIR)
