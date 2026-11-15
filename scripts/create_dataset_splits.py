
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

CSV_PATH = Path(
    "data/raw/annotations/"
    "LUNA25_Public_Training_Development_Data.csv"
)

OUTPUT_DIR = Path("data/processed/splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# Load annotations
df = pd.read_csv(CSV_PATH)

# Create one record per patient
patients = (
    df.groupby("PatientID")
    .agg(label=("label", "first"))
    .reset_index()
)

# First: separate training (70%) and remaining (30%)
train_patients, remaining = train_test_split(
    patients,
    test_size=0.30,
    random_state=SEED,
    stratify=patients["label"]
)

# Second: divide remaining equally into validation and test
val_patients, test_patients = train_test_split(
    remaining,
    test_size=0.50,
    random_state=SEED,
    stratify=remaining["label"]
)

# Create patient split manifest
train_patients = train_patients.assign(split="train")
val_patients = val_patients.assign(split="validation")
test_patients = test_patients.assign(split="test")

patient_splits = pd.concat(
    [train_patients, val_patients, test_patients],
    ignore_index=True
)

# Add split information to every annotation
merged = df.merge(
    patient_splits[["PatientID", "split"]],
    on="PatientID",
    how="left",
    validate="many_to_one"
)

# Save both manifests
patient_splits.to_csv(
    OUTPUT_DIR / "patient_splits.csv",
    index=False
)

merged.to_csv(
    OUTPUT_DIR / "patch_splits.csv",
    index=False
)

# Verify patient leakage
train_ids = set(train_patients["PatientID"])
val_ids = set(val_patients["PatientID"])
test_ids = set(test_patients["PatientID"])

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

assert merged["split"].notna().all()
assert len(merged) == len(df)

print("--- DATASET SPLIT SUMMARY ---")

for name in ["train", "validation", "test"]:
    subset = patient_splits[
        patient_splits["split"] == name
    ]

    print(f"\n{name.upper()}")
    print("Patients:", len(subset))
    print("Label distribution:")
    print(subset["label"].value_counts().sort_index())

    print(
        "Class percentages:",
        (subset["label"].value_counts(normalize=True)
         .sort_index() * 100).round(2).to_dict()
    )

print("\n--- LEAKAGE VERIFICATION ---")
print("Train-validation overlap:", len(train_ids & val_ids))
print("Train-test overlap:", len(train_ids & test_ids))
print("Validation-test overlap:", len(val_ids & test_ids))

print("\n--- SAVED FILES ---")
print(OUTPUT_DIR / "patient_splits.csv")
print(OUTPUT_DIR / "patch_splits.csv")
