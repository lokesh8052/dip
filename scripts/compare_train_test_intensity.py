
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = (
    PROJECT_ROOT
    / "data/processed/nodule_blocks/luna25_nodule_blocks/image"
)

TEST_DIR = (
    PROJECT_ROOT
    / "data/processed/luna25_test_patches"
)

PREDICTIONS = (
    PROJECT_ROOT
    / "outputs/test_evaluation/test_predictions.csv"
)

# --------------------------------------------------
# Load test prediction records
# --------------------------------------------------

df = pd.read_csv(PREDICTIONS)

# Use all 25 test patches, including correctly classified ones
test_names = df["AnnotationID"].tolist()

# --------------------------------------------------
# Select a reproducible sample of training patches
# --------------------------------------------------

train_files = sorted(TRAIN_DIR.glob("*.npy"))

if not train_files:
    raise FileNotFoundError(f"No training patches found: {TRAIN_DIR}")

rng = np.random.default_rng(42)

sample_size = min(100, len(train_files))

selected_train = rng.choice(
    train_files,
    size=sample_size,
    replace=False
)

# --------------------------------------------------
# Calculate intensity statistics
# --------------------------------------------------

def get_stats(image):

    image = image.astype(np.float32)

    # Model preprocessing
    processed = np.clip(image, -1000, 400)
    processed = (processed + 1000) / 1400

    return {
        "raw_min": float(image.min()),
        "raw_max": float(image.max()),
        "raw_mean": float(image.mean()),
        "raw_std": float(image.std()),

        "below_minus_1000_pct": float(
            np.mean(image < -1000) * 100
        ),

        "above_400_pct": float(
            np.mean(image > 400) * 100
        ),

        "normalized_mean": float(processed.mean()),
        "normalized_std": float(processed.std()),
    }


# --------------------------------------------------
# Analyze training sample
# --------------------------------------------------

train_stats = []

for path in selected_train:

    image = np.load(path)

    if image.shape != (64, 128, 128):
        continue

    stats = get_stats(image)
    train_stats.append(stats)

train_df = pd.DataFrame(train_stats)

# --------------------------------------------------
# Analyze test patches
# --------------------------------------------------

test_stats = []

for name in test_names:

    path = TEST_DIR / f"{name}.npy"

    image = np.load(path)

    stats = get_stats(image)
    stats["AnnotationID"] = name

    test_stats.append(stats)

test_df = pd.DataFrame(test_stats)

# --------------------------------------------------
# Display results
# --------------------------------------------------

columns = [
    "raw_min",
    "raw_max",
    "raw_mean",
    "raw_std",
    "below_minus_1000_pct",
    "above_400_pct",
    "normalized_mean",
    "normalized_std",
]

print("\n" + "=" * 75)
print("TRAINING PATCH DISTRIBUTION")
print("=" * 75)

print("Training patches analyzed:", len(train_df))
print(train_df[columns].describe().round(4).to_string())

print("\n" + "=" * 75)
print("ALL 25 TEST PATCHES")
print("=" * 75)

print(test_df[["AnnotationID"] + columns].round(4).to_string(index=False))

print("\n" + "=" * 75)
print("TEST PATCH DISTRIBUTION")
print("=" * 75)

print(test_df[columns].describe().round(4).to_string())

print("\nComparison completed.")
