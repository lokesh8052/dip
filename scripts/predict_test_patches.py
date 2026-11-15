
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    balanced_accuracy_score,
)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Make project modules importable
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import LungCNN

PATCH_DIR = PROJECT_ROOT / "data/processed/luna25_test_patches"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints/best_model.pth"

OUTPUT_DIR = PROJECT_ROOT / "outputs/test_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "test_predictions.csv"
METRICS_PATH = OUTPUT_DIR / "test_metrics.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = {
    0: "Benign",
    1: "Malignant",
}


# ============================================================
# 2. VALIDATE PATHS
# ============================================================

print("=" * 65)
print("DIP-MONet: TEST PATCH PREDICTION")
print("=" * 65)

print("Project root:", PROJECT_ROOT)
print("Device:", DEVICE)

if not PATCH_DIR.exists():
    raise FileNotFoundError(f"Patch directory not found: {PATCH_DIR}")

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

model = LungCNN().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True,
)

if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.eval()

print("\nModel loaded successfully.")
print("Checkpoint:", CHECKPOINT_PATH.name)


# ============================================================
# 4. FIND PATCH FILES
# ============================================================

patch_files = sorted(PATCH_DIR.glob("*.npy"))

print("Number of patches found:", len(patch_files))

if not patch_files:
    raise RuntimeError("No .npy patches found.")

results = []
skipped = []


# ============================================================
# 5. RUN INFERENCE
# ============================================================

print("\nStarting inference...\n")

with torch.no_grad():

    for index, patch_path in enumerate(patch_files, start=1):

        metadata_path = patch_path.with_suffix(".json")

        if not metadata_path.exists():
            print("Skipping missing metadata:", patch_path.name)
            skipped.append(patch_path.name)
            continue

        # Load metadata
        with open(metadata_path, "r") as file:
            metadata = json.load(file)

        # Load patch
        image = np.load(patch_path)

        if image.shape != (64, 128, 128):
            raise ValueError(
                f"Unexpected patch shape in {patch_path.name}: {image.shape}"
            )

        # Same preprocessing used during training
        image = image.astype(np.float32)
        image = np.clip(image, -1000, 400)
        image = (image + 1000.0) / 1400.0

        # Add channel and batch dimensions
        image_tensor = torch.from_numpy(image)
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        image_tensor = image_tensor.to(DEVICE)

        # Model prediction
        logits, embeddings = model(image_tensor)

        probabilities = torch.softmax(logits, dim=1)

        benign_probability = probabilities[0, 0].item()
        malignant_probability = probabilities[0, 1].item()

        predicted_label = int(
            torch.argmax(probabilities, dim=1).item()
        )

        actual_label = int(metadata["label"])

        if actual_label not in (0, 1):
            raise ValueError(
                f"Invalid label in {metadata_path.name}: {actual_label}"
            )

        # Save prediction
        results.append({
            "AnnotationID": patch_path.stem,
            "PatientID": str(metadata["PatientID"]),
            "SeriesInstanceUID": str(metadata["SeriesInstanceUID"]),
            "Actual_Label": actual_label,
            "Actual_Class": CLASS_NAMES[actual_label],
            "Predicted_Label": predicted_label,
            "Predicted_Class": CLASS_NAMES[predicted_label],
            "Benign_Probability": benign_probability,
            "Malignant_Probability": malignant_probability,
            "Correct": predicted_label == actual_label,
        })

        print(
            f"[{index}/{len(patch_files)}] "
            f"{patch_path.stem} | "
            f"Actual: {CLASS_NAMES[actual_label]} | "
            f"Predicted: {CLASS_NAMES[predicted_label]} | "
            f"P(Malignant): {malignant_probability:.4f}"
        )


# ============================================================
# 6. SAVE PREDICTIONS
# ============================================================

df = pd.DataFrame(results)

if df.empty:
    raise RuntimeError("No predictions were generated.")

df.to_csv(CSV_PATH, index=False)

print("\nPredictions saved:", CSV_PATH)

if skipped:
    print("Skipped patches:", len(skipped))


# ============================================================
# 7. CALCULATE EVALUATION METRICS
# ============================================================

y_true = df["Actual_Label"].to_numpy()
y_pred = df["Predicted_Label"].to_numpy()
y_prob = df["Malignant_Probability"].to_numpy()

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true, y_pred, zero_division=0
)

recall = recall_score(
    y_true, y_pred, zero_division=0
)

f1 = f1_score(
    y_true, y_pred, zero_division=0
)

balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0.0
)

if len(np.unique(y_true)) == 2:
    auc = roc_auc_score(y_true, y_prob)
else:
    auc = None


# ============================================================
# 8. SAVE METRICS
# ============================================================

metrics = {
    "number_of_patches": int(len(df)),
    "correct_predictions": int(df["Correct"].sum()),
    "incorrect_predictions": int((~df["Correct"]).sum()),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall_sensitivity": float(recall),
    "specificity": float(specificity),
    "f1_score": float(f1),
    "balanced_accuracy": float(balanced_accuracy),
    "roc_auc": float(auc) if auc is not None else None,
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),
    "confusion_matrix": cm.tolist(),
}

with open(METRICS_PATH, "w") as file:
    json.dump(metrics, file, indent=4)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 65)
print("TEST EVALUATION RESULTS")
print("=" * 65)

print("Total patches:", len(df))
print("Correct predictions:", int(df["Correct"].sum()))
print("Incorrect predictions:", int((~df["Correct"]).sum()))

print(f"\nAccuracy:           {accuracy:.4f}")
print(f"Precision:          {precision:.4f}")
print(f"Recall/Sensitivity: {recall:.4f}")
print(f"Specificity:        {specificity:.4f}")
print(f"F1-score:           {f1:.4f}")
print(f"Balanced Accuracy:  {balanced_accuracy:.4f}")

if auc is not None:
    print(f"ROC-AUC:            {auc:.4f}")
else:
    print("ROC-AUC:            Not available")

print("\nConfusion Matrix:")
print("                 Predicted")
print("                 Benign  Malignant")
print(f"Actual Benign      {tn:4d}      {fp:4d}")
print(f"Actual Malignant   {fn:4d}      {tp:4d}")

print("\nOutput files:")
print("Predictions:", CSV_PATH)
print("Metrics:", METRICS_PATH)

print("\nTEST PREDICTION COMPLETED.")
