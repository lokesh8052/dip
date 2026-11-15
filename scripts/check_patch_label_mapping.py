
from pathlib import Path
import pandas as pd

CSV_PATH = Path(
    "data/raw/annotations/LUNA25_Public_Training_Development_Data.csv"
)

IMAGE_DIR = Path(
    "data/processed/nodule_blocks/luna25_nodule_blocks/image"
)

df = pd.read_csv(CSV_PATH, dtype={"AnnotationID": str})

# Get image filenames without .npy
image_ids = {
    path.stem
    for path in IMAGE_DIR.glob("*.npy")
}

# Get annotation IDs from CSV
annotation_ids = set(df["AnnotationID"].dropna())

print("--- IDENTIFIER MAPPING ---")

print("Image files:", len(image_ids))
print("CSV rows:", len(df))
print("Unique annotation IDs:", len(annotation_ids))

# Check duplicates
print(
    "Duplicate AnnotationIDs:",
    df["AnnotationID"].duplicated().sum()
)

# Compare identifiers
matched = image_ids & annotation_ids
image_without_csv = image_ids - annotation_ids
csv_without_image = annotation_ids - image_ids

print("\nMatched image IDs:", len(matched))
print("Images without CSV annotation:", len(image_without_csv))
print("CSV annotations without image:", len(csv_without_image))

print("\n--- SAMPLE MATCHED RECORDS ---")

matched_df = df[df["AnnotationID"].isin(matched)]

print(
    matched_df[
        ["AnnotationID", "PatientID", "NoduleID", "label"]
    ].head(10).to_string(index=False)
)

print("\n--- LABEL DISTRIBUTION ---")
print(matched_df["label"].value_counts(dropna=False))

print("\n--- NODULE LABEL CONSISTENCY ---")

labels_per_nodule = matched_df.groupby("NoduleID")["label"].nunique()

print(
    "Nodules with conflicting labels:",
    (labels_per_nodule > 1).sum()
)
