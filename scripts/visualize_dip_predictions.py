
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --------------------------------------------------
# 1. Load CT volume and preliminary lung mask
# --------------------------------------------------

volume_path = "data/processed/ct_volumes/patient_100012.npz"
mask_path = "data/processed/ct_volumes/patient_100012_preliminary_lung_mask.npz"

volume = np.load(volume_path)["volume"]
lung_mask = np.load(mask_path)["mask"]

# --------------------------------------------------
# 2. Load DIP-MONet predictions
# --------------------------------------------------

results_path = (
    "data/processed/ct_candidates/patient_100012/"
    "dip_monet_inference_results.csv"
)

df = pd.read_csv(results_path)

# Highest Class 1 probability
top = df.sort_values(
    "class_1_probability", ascending=False
).iloc[0]

pred_z = int(top["z"])
pred_row = int(top["row"])
pred_col = int(top["col"])
pred_probability = float(top["class_1_probability"])
candidate_rank = int(top["candidate_rank"])

# Ground-truth annotation coordinates
gt_z, gt_row, gt_col = 125, 313, 199

# --------------------------------------------------
# 3. Create visualization
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# Ground-truth slice
ax = axes[0]
ax.imshow(
    volume[gt_z],
    cmap="gray",
    vmin=-1000,
    vmax=400
)

ax.contour(
    lung_mask[gt_z].astype(float),
    levels=[0.5],
    colors="yellow",
    linewidths=0.6
)

ax.scatter(
    gt_col,
    gt_row,
    color="red",
    marker="x",
    s=130,
    linewidths=3,
    label="Ground-truth annotation"
)

ax.set_title(
    f"Ground Truth: Malignant Nodule\n"
    f"Voxel = ({gt_z}, {gt_row}, {gt_col})"
)
ax.legend(loc="upper right")
ax.set_xlabel("Column")
ax.set_ylabel("Row")

# Highest-scoring prediction slice
ax = axes[1]
ax.imshow(
    volume[pred_z],
    cmap="gray",
    vmin=-1000,
    vmax=400
)

ax.contour(
    lung_mask[pred_z].astype(float),
    levels=[0.5],
    colors="yellow",
    linewidths=0.6
)

# Candidate patch is 128 x 128 in-plane
patch_start_row = pred_row - 64
patch_start_col = pred_col - 64

ax.add_patch(
    Rectangle(
        (patch_start_col, patch_start_row),
        128,
        128,
        fill=False,
        edgecolor="cyan",
        linewidth=2,
        label="Candidate patch boundary"
    )
)

ax.scatter(
    pred_col,
    pred_row,
    color="red",
    marker="x",
    s=130,
    linewidths=3,
    label="Candidate center"
)

ax.set_title(
    f"Top DIP-MONet Candidate: {candidate_rank}\n"
    f"Class 1 score = {pred_probability:.4f}"
)
ax.legend(loc="upper right")
ax.set_xlabel("Column")
ax.set_ylabel("Row")

plt.suptitle(
    "DIP-MONet Full CT Prediction Visualization - Patient 100012",
    fontsize=14
)

plt.tight_layout()

# --------------------------------------------------
# 4. Save output
# --------------------------------------------------

os.makedirs("outputs", exist_ok=True)

output_path = (
    "outputs/ct_prediction_visualization_patient_100012.png"
)

plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.show()

print("\nVisualization saved successfully:")
print(output_path)

print("\nGround truth voxel:", (gt_z, gt_row, gt_col))
print("Top candidate:", candidate_rank)
print("Prediction center:", (pred_z, pred_row, pred_col))
print("Class 1 probability:", pred_probability)
