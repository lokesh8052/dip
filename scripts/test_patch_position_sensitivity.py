
import sys
from pathlib import Path

# =========================================
# 1. PROJECT PATH
# =========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import csv

from src.model import LungCNN


# =========================================
# 2. CONFIGURATION
# =========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

VOLUME_PATH = (
    PROJECT_ROOT
    / "data/processed/ct_volumes/patient_100012.npz"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints/best_model.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs/position_sensitivity"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SHAPE = (64, 128, 128)

# Annotation location in reconstructed CT volume
ANNOTATION = np.array([125, 313, 199])

print("Device:", DEVICE)


# =========================================
# 3. LOAD MODEL
# =========================================

model = LungCNN().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded successfully.")


# =========================================
# 4. LOAD CT VOLUME
# =========================================

data = np.load(VOLUME_PATH)

volume = data["volume"].astype(np.float32)

print("CT volume shape:", volume.shape)


# =========================================
# 5. PATCH EXTRACTION
# =========================================

def extract_patch(volume, center):

    center = np.array(center, dtype=int)

    depth, height, width = PATCH_SHAPE

    starts = center - np.array([
        depth // 2,
        height // 2,
        width // 2
    ])

    ends = starts + np.array(PATCH_SHAPE)

    # Reject crops that exceed volume boundaries.
    if np.any(starts < 0) or np.any(ends > volume.shape):
        return None

    patch = volume[
        starts[0]:ends[0],
        starts[1]:ends[1],
        starts[2]:ends[2]
    ]

    return patch


# =========================================
# 6. PREDICTION
# =========================================

def predict_patch(patch):

    patch = np.clip(patch, -1000, 400)

    patch = (patch + 1000.0) / 1400.0

    tensor = torch.from_numpy(
        patch.astype(np.float32)
    )

    tensor = tensor.unsqueeze(0).unsqueeze(0)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        logits, embeddings = model(tensor)

        probabilities = torch.softmax(
            logits, dim=1
        )[0]

    benign = float(probabilities[0].item())
    malignant = float(probabilities[1].item())

    return benign, malignant


# =========================================
# 7. POSITION EXPERIMENT
# =========================================

# Each tuple represents a shift in:
# (z, row, column)

shifts = [
    (0, 0, 0),

    (0, 0, 16),
    (0, 0, -16),

    (0, 16, 0),
    (0, -16, 0),

    (16, 0, 0),
    (-16, 0, 0),

    (0, 16, 16),
    (0, -16, -16),

    (16, 16, 16),
    (-16, -16, -16),

    (8, 8, 8),
    (-8, -8, -8)
]


# =========================================
# 8. RUN EXPERIMENT
# =========================================

results = []

print("\n" + "=" * 80)
print("PATCH POSITION SENSITIVITY EXPERIMENT")
print("=" * 80)

print(
    f"{'Shift (z,row,col)':<25}"
    f"{'Benign':<15}"
    f"{'Malignant':<15}"
    f"{'Prediction':<15}"
)

print("-" * 80)

for shift in shifts:

    center = ANNOTATION + np.array(shift)

    patch = extract_patch(volume, center)

    if patch is None:
        print("Skipping invalid position:", shift)
        continue

    benign, malignant = predict_patch(patch)

    prediction = (
        "Malignant" if malignant >= 0.5
        else "Benign"
    )

    results.append({
        "shift_z": shift[0],
        "shift_row": shift[1],
        "shift_col": shift[2],
        "center_z": int(center[0]),
        "center_row": int(center[1]),
        "center_col": int(center[2]),
        "benign_probability": benign,
        "malignant_probability": malignant,
        "prediction": prediction
    })

    print(
        f"{str(shift):<25}"
        f"{benign:<15.4f}"
        f"{malignant:<15.4f}"
        f"{prediction:<15}"
    )


# =========================================
# 9. SAVE RESULTS
# =========================================

csv_path = OUTPUT_DIR / "position_sensitivity.csv"

with open(csv_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)


# =========================================
# 10. SUMMARY
# =========================================

scores = [
    result["malignant_probability"]
    for result in results
]

print("\nExperiment completed.")

print("Total valid positions:", len(results))
print("Minimum malignant score:", round(min(scores), 4))
print("Maximum malignant score:", round(max(scores), 4))
print("Mean malignant score:", round(float(np.mean(scores)), 4))

print("\nResults saved to:")
print(csv_path)
