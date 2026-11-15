
from pathlib import Path
import pandas as pd

BASE = Path("data/processed/nodule_blocks/luna25_nodule_blocks")

CSV_PATH = Path(
    "data/processed/splits/patch_splits.csv"
)

OUTPUT_PATH = Path(
    "data/processed/splits/dataset_manifest.csv"
)

df = pd.read_csv(CSV_PATH, dtype={"AnnotationID": str})

# Construct paths from the annotation ID
df["image_path"] = df["AnnotationID"].apply(
    lambda x: str(BASE / "image" / f"{x}.npy")
)

df["metadata_path"] = df["AnnotationID"].apply(
    lambda x: str(BASE / "metadata" / f"{x}.npy")
)

# Check that every referenced file exists
df["image_exists"] = df["image_path"].apply(
    lambda x: Path(x).is_file()
)

df["metadata_exists"] = df["metadata_path"].apply(
    lambda x: Path(x).is_file()
)

print("--- MANIFEST VALIDATION ---")
print("Total records:", len(df))
print("Missing images:", (~df["image_exists"]).sum())
print("Missing metadata:", (~df["metadata_exists"]).sum())
print("Missing splits:", df["split"].isna().sum())

# Keep useful columns
columns = [
    "PatientID",
    "SeriesInstanceUID",
    "StudyDate",
    "AnnotationID",
    "NoduleID",
    "label",
    "split",
    "image_path",
    "metadata_path"
]

manifest = df[columns].copy()

# Save manifest
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
manifest.to_csv(OUTPUT_PATH, index=False)

print("\n--- SPLIT RECORD COUNTS ---")
print(manifest["split"].value_counts())

print("\n--- SAMPLE RECORDS ---")
print(manifest.head(5).to_string(index=False))

print("\nSaved:", OUTPUT_PATH)
