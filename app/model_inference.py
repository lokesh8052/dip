from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.model import LungCNN

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS = {
    "DIP-MONet": PROJECT_ROOT / "checkpoints/best_model.pth",
    "Baseline": PROJECT_ROOT / "checkpoints/baseline_best_model.pth"
}


def load_model(checkpoint_path):
    model = LungCNN().to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model


def preprocess_patch(patch):
    patch = np.asarray(patch)

    if patch.shape != (64, 128, 128):
        raise ValueError(
            f"Expected patch shape (64, 128, 128), got {patch.shape}"
        )

    patch = patch.astype(np.float32)
    patch = np.clip(patch, -1000, 400)
    patch = (patch + 1000.0) / 1400.0

    return torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(DEVICE)


def predict_patch(patch, models):
    tensor = preprocess_patch(patch)
    results = {}

    with torch.inference_mode():
        for model_name, model in models.items():
            logits, embeddings = model(tensor)
            probabilities = F.softmax(logits, dim=1)[0]

            results[model_name] = {
                "predicted_class": int(logits.argmax(dim=1).item()),
                "class_0_score": float(probabilities[0].item()),
                "class_1_score": float(probabilities[1].item()),
                "embedding": embeddings[0].cpu().numpy()
            }

    return results


def load_all_models():
    models = {}

    for name, path in CHECKPOINTS.items():
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        models[name] = load_model(path)

    return models
