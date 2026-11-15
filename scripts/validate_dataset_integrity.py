
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

MANIFEST = Path(
    "data/processed/splits/dataset_manifest.csv"
)

df = pd.read_csv(MANIFEST)

shape_counts = Counter()
dtype_counts = Counter()

invalid_images = []
invalid_metadata = []

global_min = float("inf")
global_max = float("-inf")

print("Validating dataset...")

for index, row in df.iterrows():

    image_path = Path(row["image_path"])
    metadata_path = Path(row["metadata_path"])

    try:
        image = np.load(image_path, mmap_mode="r")

        shape_counts[image.shape] += 1
        dtype_counts[str(image.dtype)] += 1

        if image.shape != (64, 128, 128):
            invalid_images.append(
                (image_path.name, "Unexpected shape")
            )

        if image.dtype != np.int16:
            invalid_images.append(
                (image_path.name, "Unexpected dtype")
            )

        if not np.isfinite(image).all():
            invalid_images.append(
                (image_path.name, "Non-finite values")
            )

        global_min = min(global_min, int(image.min()))
        global_max = max(global_max, int(image.max()))

        metadata = np.load(
            metadata_path,
            allow_pickle=True
        ).item()

        required_keys = {"origin", "spacing", "transform"}

        if not required_keys.issubset(metadata.keys()):
            invalid_metadata.append(metadata_path.name)

    except Exception as error:
        invalid_images.append(
            (image_path.name, str(error))
        )

    if (index + 1) % 500 == 0:
        print(f"Checked {index + 1}/{len(df)}")

print("\n--- DATASET INTEGRITY REPORT ---")
print("Total records:", len(df))
print("Image shapes:", shape_counts)
print("Image data types:", dtype_counts)
print("Global minimum:", global_min)
print("Global maximum:", global_max)
print("Invalid images:", len(invalid_images))
print("Invalid metadata:", len(invalid_metadata))

if invalid_images:
    print("\nImage errors:", invalid_images[:10])

if invalid_metadata:
    print("\nMetadata errors:", invalid_metadata[:10])

if not invalid_images and not invalid_metadata:
    print("\nALL DATASET CHECKS PASSED!")
