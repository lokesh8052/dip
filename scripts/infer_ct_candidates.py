import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import LungCNN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS = {
    "DIP-MONet": PROJECT_ROOT / "checkpoints/best_model.pth",
    "Baseline": PROJECT_ROOT / "checkpoints/baseline_best_model.pth"
}

PATCH_SHAPE = (64, 128, 128)


def load_model(checkpoint_path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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


def load_patches(patch_dir, metadata):
    patches = []

    for row in metadata:
        rank = int(row[0])
        path = patch_dir / f"candidate_{rank:03d}.npy"

        if not path.exists():
            raise FileNotFoundError(f"Patch not found: {path}")

        patch = np.load(path)

        if patch.shape != PATCH_SHAPE:
            raise ValueError(
                f"Invalid patch shape in {path}: {patch.shape}"
            )

        patch = patch.astype(np.float32)
        patch = np.clip(patch, -1000, 400)
        patch = (patch + 1000.0) / 1400.0

        patches.append(patch)

    return patches


def run_inference(model, patches, metadata, batch_size):
    results = []

    with torch.inference_mode():
        for start in range(0, len(patches), batch_size):
            batch = np.stack(patches[start:start + batch_size])

            tensor = torch.from_numpy(batch).unsqueeze(1).to(DEVICE)

            logits, embeddings = model(tensor)

            probabilities = F.softmax(logits, dim=1)
            scores = probabilities[:, 1].cpu().numpy()
            predictions = logits.argmax(dim=1).cpu().numpy()

            for i, score in enumerate(scores):
                row = metadata[start + i]

                results.append({
                    "candidate_rank": int(row[0]),
                    "component_id": int(row[1]),
                    "component_size": int(row[2]),
                    "z": int(row[3]),
                    "row": int(row[4]),
                    "col": int(row[5]),
                    "predicted_class": int(predictions[i]),
                    "class_1_score": float(score),
                    "annotation_inside_patch": (
                        bool(row[6]) if len(row) > 6 else ""
                    )
                })

            print(
                f"Processed {min(start + batch_size, len(patches))}"
                f"/{len(patches)} patches"
            )

    results.sort(
        key=lambda item: item["class_1_score"],
        reverse=True
    )

    return results


def save_results(results, output_path):
    fields = [
        "candidate_rank",
        "component_id",
        "component_size",
        "z",
        "row",
        "col",
        "predicted_class",
        "class_1_score",
        "annotation_inside_patch"
    ]

    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(
        description="Run DIP-MONet and Baseline on CT candidate patches."
    )

    parser.add_argument("--patch-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")

    patch_dir = Path(args.patch_dir)
    output_dir = Path(args.output_dir)

    metadata_path = patch_dir / "patch_metadata.npy"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    metadata = np.load(metadata_path)

    if metadata.ndim != 2 or metadata.shape[1] < 6:
        raise ValueError("Invalid patch metadata.")

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Candidate patches:", len(metadata))

    patches = load_patches(patch_dir, metadata)
    print("All patches loaded and normalized.")

    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, checkpoint_path in CHECKPOINTS.items():
        print(f"\nLoading {model_name}...")

        model = load_model(checkpoint_path)

        print(f"Starting {model_name} inference...")

        results = run_inference(
            model,
            patches,
            metadata,
            args.batch_size
        )

        output_path = output_dir / (
            model_name.lower().replace("-", "_") + "_results.csv"
        )

        save_results(results, output_path)

        print(f"\nTop 10 {model_name} candidates:")

        for rank, result in enumerate(results[:10], start=1):
            print(
                f"{rank}. Candidate={result['candidate_rank']}, "
                f"Class={result['predicted_class']}, "
                f"Score={result['class_1_score']:.4f}, "
                f"Center=({result['z']}, "
                f"{result['row']}, {result['col']})"
            )

        print("Results saved:", output_path)

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    print("\nBoth model inference runs completed.")


if __name__ == "__main__":
    main()
