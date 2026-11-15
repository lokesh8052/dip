
from pathlib import Path
import json
import numpy as np

PATCH_DIR = Path("data/processed/test_nodule_patches")

for patient in ["128669", "120708", "210559"]:

    patch = np.load(
        PATCH_DIR / f"patient_{patient}.npy"
    )

    with open(PATCH_DIR / f"patient_{patient}.json") as f:
        metadata = json.load(f)

    annotation = np.array(
        metadata["annotation_patch_voxel"]
    )

    # Count voxels at or below the padding value.
    padding_count = np.sum(patch == -1000)

    total = patch.size

    # Training-equivalent intensity preprocessing.
    normalized = (
        np.clip(patch.astype(np.float32), -1000, 400)
        + 1000
    ) / 1400

    print("\n" + "=" * 55)
    print("PATIENT:", patient)
    print("=" * 55)

    print("Label:", metadata["label"])
    print("Shape:", patch.shape)
    print("Annotation position:", annotation)

    print("\nRaw HU statistics:")
    print("Minimum:", patch.min())
    print("Maximum:", patch.max())
    print("Mean:", round(float(patch.mean()), 2))
    print("Standard deviation:", round(float(patch.std()), 2))

    print("\nIntensity distribution:")
    for low, high in [
        (-1024, -500),
        (-500, -100),
        (-100, 0),
        (0, 100),
        (100, 400),
        (400, 2000)
    ]:
        count = np.sum(
            (patch >= low) & (patch < high)
        )
        print(
            f"{low:>5} to {high:>5}: "
            f"{count:>9} "
            f"({100 * count / total:.2f}%)"
        )

    print("\nPadding-value count:")
    print(
        f"{padding_count}/{total} "
        f"({100 * padding_count / total:.2f}%)"
    )

    print("\nNormalized range:")
    print(normalized.min(), normalized.max())

    print("\nPatch crop start:", metadata["patch_start_voxel"])
    print("Patch origin:", metadata["origin"])
