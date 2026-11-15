
import sys
from pathlib import Path

# =========================================
# 1. PROJECT IMPORT PATH
# =========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from src.model import LungCNN


# =========================================
# 2. CONFIGURATION
# =========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"

print("Device:", DEVICE)


# =========================================
# 3. LOAD TRAINED MODEL
# =========================================

model = LungCNN().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("DIP-MONet checkpoint loaded successfully.")
print("Best epoch:", checkpoint["epoch"])
print("Validation AUC:", checkpoint["validation_auc"])


# =========================================
# 4. PREDICTION FUNCTION
# =========================================

def predict_patch(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Patch not found: {path}")

    patch = np.load(path).astype(np.float32)

    if patch.shape != (64, 128, 128):
        raise ValueError(
            f"Unexpected patch shape: {patch.shape}"
        )

    # Same normalization used during training
    patch = np.clip(patch, -1000, 400)
    patch = (patch + 1000.0) / 1400.0

    tensor = torch.from_numpy(patch)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        logits, embeddings = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    probabilities = probabilities.cpu().numpy()

    predicted_class = int(np.argmax(probabilities))

    return {
        "benign_probability": float(probabilities[0]),
        "malignant_probability": float(probabilities[1]),
        "predicted_class": predicted_class
    }


# =========================================
# 5. PATCH PATHS
# =========================================

patches = {
    "Original training patch": (
        "data/processed/nodule_blocks/luna25_nodule_blocks/"
        "image/100012_1_19990102.npy"
    ),

    "Candidate 3": (
        "data/processed/ct_candidates/patient_100012/"
        "patches/candidate_003.npy"
    ),

    "Candidate 11": (
        "data/processed/ct_candidates/patient_100012/"
        "patches/candidate_011.npy"
    ),

    "Candidate 22": (
        "data/processed/ct_candidates/patient_100012/"
        "patches/candidate_022.npy"
    ),

    "Candidate 23": (
        "data/processed/ct_candidates/patient_100012/"
        "patches/candidate_023.npy"
    )
}


# =========================================
# 6. RUN COMPARISON
# =========================================

print("\n" + "=" * 75)
print("DIP-MONET PATCH PREDICTION COMPARISON")
print("=" * 75)

print(
    f"{'Patch':<28}"
    f"{'Benign':<15}"
    f"{'Malignant':<15}"
    f"{'Prediction':<12}"
)

print("-" * 75)

for name, path in patches.items():

    result = predict_patch(
        PROJECT_ROOT / path
    )

    predicted_class = result["predicted_class"]

    prediction_name = (
        "Benign" if predicted_class == 0
        else "Malignant"
    )

    print(
        f"{name:<28}"
        f"{result['benign_probability']:<15.4f}"
        f"{result['malignant_probability']:<15.4f}"
        f"{prediction_name:<12}"
    )


print("\nComparison completed successfully.")
