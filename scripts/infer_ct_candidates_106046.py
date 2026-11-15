
import os
import csv
import numpy as np
import torch
import torch.nn.functional as F

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import LungCNN

# ==================================================
# 1. CONFIGURATION
# ==================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PATCH_DIR = (
    "data/processed/ct_candidates/"
    "patient_106046/patches"
)

OUTPUT_DIR = (
    "data/processed/ct_candidates/"
    "patient_106046"
)

CHECKPOINT_PATH = "checkpoints/best_model.pth"

BATCH_SIZE = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# ==================================================
# 2. LOAD MODEL
# ==================================================

model = LungCNN().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

# Support either a raw state dictionary or a
# checkpoint dictionary containing model weights.

if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)

model.eval()

print("\nDIP-MONet checkpoint loaded successfully.")

# ==================================================
# 3. LOAD CANDIDATE METADATA
# ==================================================

metadata = np.load(
    os.path.join(PATCH_DIR, "patch_metadata.npy")
)

print("Number of candidate patches:", len(metadata))

# ==================================================
# 4. LOAD AND PREPROCESS PATCHES
# ==================================================

patches = []

for row in metadata:

    candidate_rank = int(row[0])

    filename = f"candidate_{candidate_rank:03d}.npy"

    patch_path = os.path.join(
        PATCH_DIR,
        filename
    )

    patch = np.load(patch_path).astype(np.float32)

    if patch.shape != (64, 128, 128):
        raise ValueError(
            f"Unexpected shape for {filename}: {patch.shape}"
        )

    # Same HU clipping and normalization
    # used by the training pipeline.
    patch = np.clip(patch, -1000, 400)

    patch = (patch + 1000.0) / 1400.0

    patches.append(patch)

print("All patches loaded and normalized.")

# ==================================================
# 5. MODEL INFERENCE
# ==================================================

results = []

print("\nStarting DIP-MONet inference...\n")

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

        positive_probabilities = (
            probabilities[:, 1]
            .cpu()
            .numpy()
        )

        predicted_classes = (
            torch.argmax(logits, dim=1)
            .cpu()
            .numpy()
        )

        for i in range(len(batch_patches)):

            metadata_row = metadata[start + i]

            results.append({
                "candidate_rank": int(metadata_row[0]),
                "component_id": int(metadata_row[1]),
                "component_size": int(metadata_row[2]),
                "z": int(metadata_row[3]),
                "row": int(metadata_row[4]),
                "col": int(metadata_row[5]),
                "predicted_class": int(predicted_classes[i]),
                "class_1_probability": float(
                    positive_probabilities[i]
                ),
                "annotation_inside_patch": bool(
                    metadata_row[6]
                )
            })

        print(
            f"Processed "
            f"{min(start + BATCH_SIZE, len(patches))}"
            f"/{len(patches)} patches"
        )

# ==================================================
# 6. RANK CANDIDATES BY MODEL PROBABILITY
# ==================================================

results.sort(
    key=lambda item: item["class_1_probability"],
    reverse=True
)

# ==================================================
# 7. SAVE RESULTS TO CSV
# ==================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "dip_monet_inference_results.csv"
)

fieldnames = [
    "candidate_rank",
    "component_id",
    "component_size",
    "z",
    "row",
    "col",
    "predicted_class",
    "class_1_probability",
    "annotation_inside_patch"
]

with open(csv_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)

# ==================================================
# 8. DISPLAY TOP 10 RESULTS
# ==================================================

print("\n======================================")
print("DIP-MONET INFERENCE COMPLETED")
print("======================================")

print("\nTop 10 candidates:")

for index, result in enumerate(results[:10], start=1):

    print(
        f"{index}. "
        f"Candidate={result['candidate_rank']}, "
        f"Class={result['predicted_class']}, "
        f"Probability={result['class_1_probability']:.4f}, "
        f"Center=({result['z']}, "
        f"{result['row']}, "
        f"{result['col']})"
    )

print("\nResults saved to:")
print(csv_path)
