import argparse
from pathlib import Path

import numpy as np


PATCH_SHAPE = (64, 128, 128)


def extract_patch(volume, center):
    starts = [
        center[0] - PATCH_SHAPE[0] // 2,
        center[1] - PATCH_SHAPE[1] // 2,
        center[2] - PATCH_SHAPE[2] // 2
    ]

    ends = [
        starts[i] + PATCH_SHAPE[i]
        for i in range(3)
    ]

    padding = [
        (max(0, -starts[i]), max(0, ends[i] - volume.shape[i]))
        for i in range(3)
    ]

    padded = np.pad(
        volume,
        padding,
        mode="constant",
        constant_values=-1000
    )

    starts = [
        starts[i] + padding[i][0]
        for i in range(3)
    ]

    patch = padded[
        starts[0]:starts[0] + PATCH_SHAPE[0],
        starts[1]:starts[1] + PATCH_SHAPE[1],
        starts[2]:starts[2] + PATCH_SHAPE[2]
    ]

    if patch.shape != PATCH_SHAPE:
        raise ValueError(f"Unexpected patch shape: {patch.shape}")

    return patch.astype(np.int16)


def extract_candidate_patches(
    ct_path,
    candidates_path,
    output_dir,
    top_k=50,
    annotation=None
):
    ct_path = Path(ct_path)
    candidates_path = Path(candidates_path)
    output_dir = Path(output_dir)

    if not ct_path.exists():
        raise FileNotFoundError(f"CT volume not found: {ct_path}")

    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates not found: {candidates_path}")

    volume = np.load(ct_path)["volume"]
    candidates = np.load(candidates_path)

    if candidates.ndim != 2 or candidates.shape[1] != 5:
        raise ValueError("Invalid candidate array. Expected 5 columns.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)

    count = min(top_k, len(candidates))
    metadata = []

    print("CT shape:", volume.shape)
    print("Available candidates:", len(candidates))
    print("Extracting patches:", count)

    for index, candidate in enumerate(candidates[:count]):
        component_id, size, z, y, x = map(int, candidate)
        center = (z, y, x)

        patch = extract_patch(volume, center)
        np.save(output_dir / f"candidate_{index + 1:03d}.npy", patch)

        annotation_inside = 0

        if annotation is not None:
            starts = [
                center[i] - PATCH_SHAPE[i] // 2
                for i in range(3)
            ]
            annotation_inside = int(
                all(
                    starts[i] <= annotation[i] < starts[i] + PATCH_SHAPE[i]
                    for i in range(3)
                )
            )

        metadata.append([
            index + 1, component_id, size, z, y, x, annotation_inside
        ])

    metadata = np.asarray(metadata, dtype=np.int64).reshape(-1, 7)
    np.save(output_dir / "patch_metadata.npy", metadata)

    print("\nPATCH EXTRACTION COMPLETED")
    print("Patches extracted:", count)
    print("Patch shape:", PATCH_SHAPE)

    if annotation is not None:
        print("Annotation-containing patches:", int(metadata[:, 6].sum()))

    print("Output directory:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--annotation", nargs=3, type=int, metavar=("Z", "Y", "X"))

    args = parser.parse_args()

    extract_candidate_patches(
        args.ct,
        args.candidates,
        args.output,
        args.top_k,
        tuple(args.annotation) if args.annotation else None
    )
