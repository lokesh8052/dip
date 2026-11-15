
import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage


def extract_candidates(
    ct_path,
    mask_path,
    output_dir,
    threshold=-300,
    min_size=8,
    max_size=100000,
    annotation=None
):
    ct_path = Path(ct_path)
    mask_path = Path(mask_path)
    output_dir = Path(output_dir)

    if not ct_path.exists():
        raise FileNotFoundError(f"CT volume not found: {ct_path}")

    if not mask_path.exists():
        raise FileNotFoundError(f"Lung mask not found: {mask_path}")

    volume = np.load(ct_path)["volume"]
    lung_mask = np.load(mask_path)["mask"].astype(bool)

    print("CT shape:", volume.shape)
    print("Lung mask shape:", lung_mask.shape)

    if volume.shape != lung_mask.shape:
        raise ValueError("CT volume and lung mask shapes do not match.")

    if not lung_mask.any():
        raise ValueError("Lung mask is empty.")

    candidate_voxels = (volume > threshold) & lung_mask

    print("Candidate voxels:", int(candidate_voxels.sum()))

    structure = np.ones((3, 3, 3), dtype=bool)

    labels, number_of_components = ndimage.label(
        candidate_voxels,
        structure=structure
    )

    component_sizes = np.bincount(labels.ravel())

    print("Total connected components:", number_of_components)

    valid_ids = np.where(
        (component_sizes >= min_size) &
        (component_sizes <= max_size)
    )[0]

    valid_ids = valid_ids[valid_ids != 0]

    component_slices = ndimage.find_objects(labels)
    candidates = []

    for component_id in valid_ids:
        bbox = component_slices[component_id - 1]

        if bbox is None:
            continue

        local_mask = labels[bbox] == component_id
        local_coords = np.where(local_mask)

        center = tuple(
            int(round(
                np.mean(local_coords[axis]) + bbox[axis].start
            ))
            for axis in range(3)
        )

        candidates.append({
            "component_id": int(component_id),
            "size": int(component_sizes[component_id]),
            "center": center
        })

    candidates.sort(
        key=lambda item: item["size"],
        reverse=True
    )

    print("Filtered candidate count:", len(candidates))

    if annotation is not None:
        if any(
            coordinate < 0 or coordinate >= volume.shape[axis]
            for axis, coordinate in enumerate(annotation)
        ):
            raise ValueError("Annotation coordinate is outside the CT volume.")

        annotation_component = int(labels[annotation])

        print("Annotation component:", annotation_component)

        if annotation_component != 0:
            print(
                "Annotation component size:",
                int(component_sizes[annotation_component])
            )
        else:
            print("Annotation is not part of the candidate mask.")

    print("\nTop 20 candidates:")

    for index, candidate in enumerate(candidates[:20], start=1):
        print(
            f"{index}. Component={candidate['component_id']}, "
            f"Size={candidate['size']}, "
            f"Center={candidate['center']}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_array = np.array(
        [
            [
                item["component_id"],
                item["size"],
                *item["center"]
            ]
            for item in candidates
        ],
        dtype=np.int64
    ).reshape(-1, 5)

    np.save(output_dir / "candidates.npy", candidate_array)

    np.savez_compressed(
        output_dir / "candidate_labels.npz",
        labels=labels
    )

    print("\nCandidate extraction completed.")
    print("Candidates:", len(candidates))
    print("Output directory:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract 3D CT candidate regions inside a lung mask."
    )

    parser.add_argument("--ct", required=True, help="CT volume NPZ path")
    parser.add_argument("--mask", required=True, help="Lung mask NPZ path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--threshold", type=float, default=-300)
    parser.add_argument("--min-size", type=int, default=8)
    parser.add_argument("--max-size", type=int, default=100000)
    parser.add_argument(
        "--annotation",
        nargs=3,
        type=int,
        metavar=("Z", "Y", "X"),
        help="Optional annotation coordinate"
    )

    args = parser.parse_args()

    extract_candidates(
        ct_path=args.ct,
        mask_path=args.mask,
        output_dir=args.output,
        threshold=args.threshold,
        min_size=args.min_size,
        max_size=args.max_size,
        annotation=tuple(args.annotation) if args.annotation else None
    )
