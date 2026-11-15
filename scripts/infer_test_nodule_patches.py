
import sys
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import LungCNN

PATCH_DIR = (
    PROJECT_ROOT / "data/processed/test_nodule_patches"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints/best_model.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs/test_nodule_inference"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATIENTS = ["128669", "120708", "210559"]

BATCH_SIZE = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# --------------------------------------------------
# DEVICE INFORMATION
# --------------------------------------------------

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(CHECKPOINT_PATH)

model = LungCNN().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]

elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]

else:
    state_dict = checkpoint

model.load_state_dict(state_dict)

model.eval()

print("\nDIP-MONet checkpoint loaded successfully.")

# --------------------------------------------------
# LOAD TEST PATCHES
# --------------------------------------------------

patches = []
patient_metadata = []

for patient in PATIENTS:

    patch_path = PATCH_DIR / f"patient_{patient}.npy"
    metadata_path = PATCH_DIR / f"patient_{patient}.json"

    if not patch_path.exists():
        raise FileNotFoundError(patch_path)

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    patch = np.load(patch_path).astype(np.float32)

    with open(metadata_path) as file:
        metadata = json.load(file)

    if patch.shape != (64, 128, 128):
        raise ValueError(
            f"Unexpected shape for Patient {patient}: "
            f"{patch.shape}"
        )

    if not np.isfinite(patch).all():
        raise ValueError(
            f"Non-finite values in Patient {patient} patch"
        )

    # Same preprocessing used during training.
    patch = np.clip(patch, -1000, 400)
    patch = (patch + 1000.0) / 1400.0

    patches.append(patch)

    patient_metadata.append({
        "patient_id": patient,
        "annotation_id": metadata["AnnotationID"],
        "nodule_id": metadata["NoduleID"],
        "true_label": int(metadata["label"])
    })

print("\nAll test patches loaded and normalized.")
print("Number of patches:", len(patches))

# --------------------------------------------------
# MODEL INFERENCE
# --------------------------------------------------

results = []

print("\nStarting inference...\n")

with torch.inference_mode():

    for start in range(0, len(patches), BATCH_SIZE):

        batch_patches = patches[
            start:start + BATCH_SIZE
        ]

        batch = np.stack(batch_patches)

        # [B, 64, 128, 128]
        # Convert to [B, 1, 64, 128, 128]

        batch_tensor = torch.from_numpy(
            batch
        ).unsqueeze(1).to(DEVICE)

        logits, embeddings = model(batch_tensor)

        probabilities = F.softmax(
            logits,
            dim=1
        )

        predicted_classes = torch.argmax(
            logits,
            dim=1
        )

        probabilities = probabilities.cpu().numpy()
        predicted_classes = predicted_classes.cpu().numpy()

        for i in range(len(batch_patches)):

            metadata = patient_metadata[start + i]

            result = {
                "patient_id": metadata["patient_id"],
                "annotation_id": metadata["annotation_id"],
                "nodule_id": metadata["nodule_id"],
                "true_label": metadata["true_label"],
                "predicted_class": int(predicted_classes[i]),
                "class_0_probability": float(probabilities[i][0]),
                "class_1_probability": float(probabilities[i][1]),
                "correct": bool(
                    predicted_classes[i] == metadata["true_label"]
                )
            }

            results.append(result)

        print(
            f"Processed "
            f"{min(start + BATCH_SIZE, len(patches))}"
            f"/{len(patches)} patches"
        )

# --------------------------------------------------
# SAVE CSV
# --------------------------------------------------

csv_path = OUTPUT_DIR / "test_nodule_predictions.csv"

fieldnames = [
    "patient_id",
    "annotation_id",
    "nodule_id",
    "true_label",
    "predicted_class",
    "class_0_probability",
    "class_1_probability",
    "correct"
]

with open(csv_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)

# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 65)
print("DIP-MONET TEST PATCH INFERENCE")
print("=" * 65)

for result in results:

    predicted_name = (
        "Malignant"
        if result["predicted_class"] == 1
        else "Benign"
    )

    actual_name = (
        "Malignant"
        if result["true_label"] == 1
        else "Benign"
    )

    print(
        f"\nPatient: {result['patient_id']}"
        f"\nActual: {actual_name}"
        f"\nPredicted: {predicted_name}"
        f"\nP(Benign): "
        f"{result['class_0_probability']:.4f}"
        f"\nP(Malignant): "
        f"{result['class_1_probability']:.4f}"
        f"\nCorrect: {result['correct']}"
    )

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

correct_count = sum(
    result["correct"] for result in results
)

print("\n" + "=" * 65)
print("INFERENCE SUMMARY")
print("=" * 65)

print("Total patches:", len(results))
print("Correct predictions:", correct_count)
print("Incorrect predictions:", len(results) - correct_count)

print(
    "Accuracy on these three patches:",
    f"{correct_count / len(results):.4f}"
)

print("\nPredictions saved to:")
print(csv_path)
