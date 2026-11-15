
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PATCH_DIR = PROJECT_ROOT / "data/processed/luna25_test_patches"

PREDICTIONS_PATH = (
    PROJECT_ROOT / "outputs/test_evaluation/test_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs/misclassified_visualizations"
)

# Automatically create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD PREDICTIONS
# ============================================================

print("=" * 65)
print("DIP-MONet: MISCLASSIFIED PATCH VISUALIZATION")
print("=" * 65)

if not PREDICTIONS_PATH.exists():
    raise FileNotFoundError(PREDICTIONS_PATH)

df = pd.read_csv(PREDICTIONS_PATH)

wrong = df[df["Correct"] == False].copy()

print("Total test patches:", len(df))
print("Misclassified patches:", len(wrong))


# ============================================================
# 3. VISUALIZATION FUNCTION
# ============================================================

def visualize_patch(row):

    name = row["AnnotationID"]

    patch_path = PATCH_DIR / f"{name}.npy"
    metadata_path = PATCH_DIR / f"{name}.json"

    if not patch_path.exists():
        print("Missing patch:", name)
        return

    if not metadata_path.exists():
        print("Missing metadata:", name)
        return

    # Load patch
    image = np.load(patch_path)

    # Load metadata
    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    if image.shape != (64, 128, 128):
        raise ValueError(
            f"Unexpected image shape for {name}: {image.shape}"
        )

    # Annotation position in ZYX coordinates
    z, y, x = metadata["annotation_patch_zyx"]

    z = int(round(z))
    y = int(round(y))
    x = int(round(x))

    # Extract three orthogonal views
    axial = image[z, :, :]
    coronal = image[:, y, :]
    sagittal = image[:, :, x]

    views = [
        (axial, "Axial View"),
        (coronal, "Coronal View"),
        (sagittal, "Sagittal View"),
    ]

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1, 3,
        figsize=(16, 6)
    )

    for ax, (view, title) in zip(axes, views):

        ax.imshow(
            view,
            cmap="gray",
            vmin=-1000,
            vmax=400,
            origin="lower"
        )

        ax.set_title(title, fontsize=13)

        ax.axis("off")

    # --------------------------------------------------------
    # Mark annotation centre
    # --------------------------------------------------------

    axes[0].plot(
        x, y,
        marker="+",
        color="red",
        markersize=14,
        markeredgewidth=2
    )

    axes[1].plot(
        x, z,
        marker="+",
        color="red",
        markersize=14,
        markeredgewidth=2
    )

    axes[2].plot(
        y, z,
        marker="+",
        color="red",
        markersize=14,
        markeredgewidth=2
    )

    # --------------------------------------------------------
    # Add prediction information
    # --------------------------------------------------------

    actual = row["Actual_Class"]
    predicted = row["Predicted_Class"]
    probability = row["Malignant_Probability"]

    fig.suptitle(
        f"{name}\n"
        f"Actual: {actual} | "
        f"Predicted: {predicted} | "
        f"Malignant Probability: {probability:.4f}",
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0, 1, 0.88])

    # --------------------------------------------------------
    # Save image
    # --------------------------------------------------------

    output_path = OUTPUT_DIR / f"{name}.png"

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", output_path.name)


# ============================================================
# 4. GENERATE ALL MISCLASSIFIED VISUALIZATIONS
# ============================================================

for _, row in wrong.iterrows():

    visualize_patch(row)


# ============================================================
# 5. COMPLETION
# ============================================================

print("\n" + "=" * 65)
print("VISUALIZATION COMPLETED")
print("=" * 65)

print("Images created:", len(wrong))
print("Output directory:", OUTPUT_DIR)
