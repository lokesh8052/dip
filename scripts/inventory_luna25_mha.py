
import sys
from pathlib import Path

import pandas as pd
import SimpleITK as sitk


# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

MHA_DIR = Path(
    "/mnt/c/Users/balla/lunger_cancer/luna25_images"
)

ANNOTATION_FILE = (
    ROOT / "data/raw/annotations/"
    "LUNA25_Public_Training_Development_Data.csv"
)

SPLIT_FILE = (
    ROOT / "data/processed/splits/"
    "patient_splits.csv"
)

OUTPUT_DIR = ROOT / "outputs/luna25_inventory"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "mha_inventory.csv"


# =====================================================
# VALIDATE INPUTS
# =====================================================

for path in [MHA_DIR, ANNOTATION_FILE, SPLIT_FILE]:

    if not path.exists():
        raise FileNotFoundError(path)

print("MHA directory:", MHA_DIR)

annotations = pd.read_csv(
    ANNOTATION_FILE,
    dtype={"PatientID": str}
)

splits = pd.read_csv(
    SPLIT_FILE,
    dtype={"PatientID": str}
)

print("Annotation records:", len(annotations))
print("Patient split records:", len(splits))


# =====================================================
# BUILD LOOKUP TABLES
# =====================================================

annotations["SeriesInstanceUID"] = (
    annotations["SeriesInstanceUID"].astype(str)
)

splits["PatientID"] = splits["PatientID"].astype(str)

# Series UID -> patient and annotations
series_groups = {
    uid: group
    for uid, group in annotations.groupby("SeriesInstanceUID")
}

# Patient ID -> split
split_lookup = dict(
    zip(splits["PatientID"], splits["split"])
)


# =====================================================
# INSPECT MHA HEADERS
# =====================================================

mha_files = sorted(
    MHA_DIR.glob("*.mha")
)

print("\nMHA files found:", len(mha_files))

records = []

for index, path in enumerate(mha_files, start=1):

    uid = path.stem

    record = {
        "filename": path.name,
        "SeriesInstanceUID": uid,
        "file_size_MB": round(
            path.stat().st_size / (1024 ** 2), 2
        )
    }

    try:

        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))

        # Read only image information, not pixel data.
        reader.ReadImageInformation()

        size = reader.GetSize()
        spacing = reader.GetSpacing()
        origin = reader.GetOrigin()
        direction = reader.GetDirection()

        record.update({
            "status": "OK",
            "size_x": size[0],
            "size_y": size[1],
            "size_z": size[2],
            "spacing_x": spacing[0],
            "spacing_y": spacing[1],
            "spacing_z": spacing[2],
            "origin_x": origin[0],
            "origin_y": origin[1],
            "origin_z": origin[2],
            "direction": str(direction)
        })

        matched = series_groups.get(uid)

        if matched is not None:

            patient_ids = matched["PatientID"].unique()

            if len(patient_ids) != 1:
                raise ValueError(
                    "Series UID maps to multiple patients"
                )

            patient = str(patient_ids[0])

            labels = sorted(
                matched["label"].dropna().astype(int).unique()
            )

            record.update({
                "PatientID": patient,
                "annotation_count": len(matched),
                "labels": ",".join(map(str, labels)),
                "annotation_match": True,
                "split": split_lookup.get(
                    patient, "NOT_FOUND"
                )
            })

        else:

            record.update({
                "PatientID": "",
                "annotation_count": 0,
                "labels": "",
                "annotation_match": False,
                "split": "UNMATCHED"
            })

    except Exception as error:

        record.update({
            "status": "ERROR",
            "error": str(error),
            "annotation_match": False,
            "annotation_count": 0,
            "split": "UNKNOWN"
        })

    records.append(record)

    print(
        f"\rInspected {index}/{len(mha_files)}: "
        f"{path.name[:45]}",
        end="",
        flush=True
    )


# =====================================================
# SAVE INVENTORY
# =====================================================

df = pd.DataFrame(records)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\n\n" + "=" * 60)
print("LUNA25 MHA INVENTORY COMPLETED")
print("=" * 60)

print("Total MHA files:", len(df))
print("Successfully inspected:", (df["status"] == "OK").sum())
print("Annotation-matched scans:", df["annotation_match"].sum())

print("\nSplit distribution:")

print(
    df["split"].value_counts(dropna=False).to_string()
)

print("\nAnnotation matching:")

print(
    df["annotation_match"].value_counts().to_string()
)

print("\nOutput:", OUTPUT_CSV)
