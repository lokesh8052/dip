
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    balanced_accuracy_score
)

from src.dataset import LungDataset
from src.model import LungCNN


# =========================================
# 1. CONFIGURATION
# =========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = "checkpoints/best_model.pth"
OUTPUT_DIR = "outputs/test_evaluation"

BATCH_SIZE = 2
THRESHOLD = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Device:", DEVICE)


# =========================================
# 2. LOAD TEST DATASET
# =========================================

test_dataset = LungDataset(
    split="test",
    two_views=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test samples:", len(test_dataset))


# =========================================
# 3. LOAD TRAINED MODEL
# =========================================

model = LungCNN().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Checkpoint loaded successfully.")
print("Best training epoch:", checkpoint["epoch"])
print("Best validation AUC:", checkpoint["validation_auc"])


# =========================================
# 4. RUN TEST INFERENCE
# =========================================

all_labels = []
all_probabilities = []

with torch.no_grad():

    for batch_index, (images, labels) in enumerate(test_loader):

        images = images.to(DEVICE)

        logits, embeddings = model(images)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[:, 1]

        all_labels.extend(labels.numpy().tolist())

        all_probabilities.extend(
            probabilities.cpu().numpy().tolist()
        )

        if (batch_index + 1) % 25 == 0:
            print(
                f"Processed "
                f"{(batch_index + 1) * BATCH_SIZE}/"
                f"{len(test_dataset)} patches"
            )


# =========================================
# 5. CALCULATE PREDICTIONS
# =========================================

y_true = np.array(all_labels)
y_prob = np.array(all_probabilities)

# Class 1 = malignant, Class 0 = benign
y_pred = (y_prob >= THRESHOLD).astype(int)


# =========================================
# 6. CALCULATE METRICS
# =========================================

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

balanced_accuracy = balanced_accuracy_score(
    y_true, y_pred
)

matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = matrix.ravel()

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0 else 0
)

auc = (
    roc_auc_score(y_true, y_prob)
    if len(np.unique(y_true)) == 2 else float("nan")
)


# =========================================
# 7. PRINT RESULTS
# =========================================

print("\n" + "=" * 55)
print("DIP-MONET TEST SET EVALUATION")
print("=" * 55)

print("Total samples:", len(y_true))
print("Threshold:", THRESHOLD)

print("\nPerformance Metrics")
print("-" * 40)

print(f"Accuracy:          {accuracy:.4f}")
print(f"Precision:         {precision:.4f}")
print(f"Recall/Sensitivity:{recall:.4f}")
print(f"Specificity:       {specificity:.4f}")
print(f"F1 Score:          {f1:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"ROC-AUC:           {auc:.4f}")

print("\nConfusion Matrix")
print(matrix)

print("\nClassification Report")

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["Benign (0)", "Malignant (1)"],
    zero_division=0
)

print(report)


# =========================================
# 8. SAVE PREDICTIONS
# =========================================

metadata = test_dataset.data.reset_index(drop=True)

results = metadata.copy()

results["true_label"] = y_true
results["malignant_probability"] = y_prob
results["predicted_label"] = y_pred
results["correct"] = (y_true == y_pred)

results.to_csv(
    os.path.join(OUTPUT_DIR, "predictions.csv"),
    index=False
)


# =========================================
# 9. SAVE METRICS
# =========================================

metrics_text = f"""
DIP-MONET TEST EVALUATION

Checkpoint: {CHECKPOINT_PATH}
Best epoch: {checkpoint['epoch']}
Validation AUC: {checkpoint['validation_auc']}

Threshold: {THRESHOLD}
Total samples: {len(y_true)}

Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
Specificity: {specificity:.4f}
F1 Score: {f1:.4f}
Balanced Accuracy: {balanced_accuracy:.4f}
ROC-AUC: {auc:.4f}

Confusion Matrix:
{matrix}

Classification Report:
{report}
"""

with open(
    os.path.join(OUTPUT_DIR, "metrics.txt"),
    "w"
) as f:
    f.write(metrics_text)


# =========================================
# 10. SAVE CONFUSION MATRIX
# =========================================

fig, ax = plt.subplots(figsize=(7, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=["Benign", "Malignant"]
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

ax.set_title("DIP-MONet Test Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    dpi=200
)

plt.close()


# =========================================
# 11. FINISH
# =========================================

print("\nEvaluation completed successfully.")

print("\nResults saved inside:")
print(OUTPUT_DIR)

print("\nGenerated files:")
print("- predictions.csv")
print("- metrics.txt")
print("- confusion_matrix.png")
