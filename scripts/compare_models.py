import matplotlib.pyplot as plt
import numpy as np
import os

# Results from our previous validation evaluation
models = ["Standard CNN", "DIP-MONet"]

roc_auc = [0.7720, 0.7846]
f1_score = [0.3983, 0.3745]

# Confusion matrices
baseline_cm = np.array([
    [686, 106],
    [36, 47]
])

dip_monet_cm = np.array([
    [671, 121],
    [36, 47]
])

os.makedirs("outputs/comparison", exist_ok=True)

# 1. ROC-AUC and F1 comparison
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

axes[0].bar(models, roc_auc)
axes[0].set_title("ROC-AUC Comparison")
axes[0].set_ylabel("ROC-AUC")
axes[0].set_ylim(0, 1)

for i, value in enumerate(roc_auc):
    axes[0].text(i, value + 0.015, f"{value:.4f}",
                 ha="center")

axes[1].bar(models, f1_score)
axes[1].set_title("F1-Score Comparison")
axes[1].set_ylabel("F1-score")
axes[1].set_ylim(0, 1)

for i, value in enumerate(f1_score):
    axes[1].text(i, value + 0.015, f"{value:.4f}",
                 ha="center")

plt.tight_layout()
plt.savefig("outputs/comparison/metrics_comparison.png",
            dpi=300)
plt.show()


# 2. Confusion matrix comparison
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

matrices = [baseline_cm, dip_monet_cm]

for ax, matrix, name in zip(
    axes, matrices, models
):
    ax.imshow(matrix, cmap="Blues")
    ax.set_title(name)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Malignant"])
    ax.set_yticklabels(["Benign", "Malignant"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i, j],
                    ha="center", va="center",
                    color="black", fontsize=13)

plt.tight_layout()
plt.savefig("outputs/comparison/confusion_matrices.png",
            dpi=300)
plt.show()

print("Comparison graphs saved successfully!")