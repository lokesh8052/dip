
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MHA_DIR = Path(
    "/mnt/c/Users/balla/lunger_cancer/luna25_images"
)

ANNOTATION_FILE = (
    ROOT / "outputs/luna25_inventory/exact_test_annotations.csv"
)

OUTPUT_DIR = (
    ROOT / "data/processed/luna25_test_patches"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target spacing in SimpleITK order: X, Y, Z.
TARGET_SPACING_XYZ = (
    0.5859375,
    0.5859375,
    2.0
)

# Patch dimensions in NumPy order: Z, Y, X.
PATCH_SHAPE_ZYX = (64, 128, 128)

PADDING_HU = -1000.0


# ============================================================
# RESAMPLE MHA TO TRAINING SPACING
# ============================================================

def resample_to_training_spacing(image):

    original_spacing = np.asarray(
        image.GetSpacing(),
        dtype=np.float64
    )

    original_size = np.asarray(
        image.GetSize(),
        dtype=np.int64
    )

    target_spacing = np.asarray(
        TARGET_SPACING_XYZ,
        dtype=np.float64
    )

    # Preserve the physical extent and first voxel centre.
    target_size = (
        np.round(
            (original_size - 1)
            * original_spacing
            / target_spacing
        ).astype(np.int64)
        + 1
    )

    target_size = tuple(
        int(x) for x in target_size
    )

    resampler = sitk.ResampleImageFilter()

    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(TARGET_SPACING_XYZ)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())

    resampler.SetTransform(
        sitk.Transform(3, sitk.sitkIdentity)
    )

    resampler.SetInterpolator(
        sitk.sitkLinear
    )

    resampler.SetDefaultPixelValue(PADDING_HU)
    resampler.SetOutputPixelType(sitk.sitkFloat32)

    resampled = resampler.Execute(image)

    return resampled


# ============================================================
# EXTRACT FIXED-SIZE PATCH
# ============================================================

def extract_patch(volume, center_xyz):

    center_xyz = np.rint(
        center_xyz
    ).astype(int)

    # Convert XYZ centre to ZYX.
    center_zyx = center_xyz[::-1]

    patch_shape = np.asarray(
        PATCH_SHAPE_ZYX,
        dtype=int
    )

    start_zyx = (
        center_zyx - patch_shape // 2
    )

    end_zyx = start_zyx + patch_shape

    padding = []

    for start, end, dimension in zip(
        start_zyx,
        end_zyx,
        volume.shape
    ):

        padding.append(
            (
                max(0, -int(start)),
                max(0, int(end) - dimension)
            )
        )

    padded = np.pad(
        volume,
        padding,
        mode="constant",
        constant_values=PADDING_HU
    )

    adjusted_start = (
        start_zyx
        + np.array([p[0] for p in padding])
    )

    patch = padded[
        adjusted_start[0]:adjusted_start[0] + patch_shape[0],
        adjusted_start[1]:adjusted_start[1] + patch_shape[1],
        adjusted_start[2]:adjusted_start[2] + patch_shape[2]
    ]

    if patch.shape != PATCH_SHAPE_ZYX:
        raise RuntimeError(
            f"Incorrect patch shape: {patch.shape}"
        )

    return (
        patch,
        center_xyz,
        start_zyx
    )


# ============================================================
# PROCESS ONE MHA SERIES
# ============================================================

def process_series(uid, annotations):

    mha_path = MHA_DIR / f"{uid}.mha"

    if not mha_path.exists():
        raise FileNotFoundError(mha_path)

    print("\n" + "=" * 65)
    print("SERIES:", uid)
    print("Annotations:", len(annotations))
    print("=" * 65)

    image = sitk.ReadImage(str(mha_path))

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    original_origin = image.GetOrigin()
    original_direction = image.GetDirection()

    print("Original size XYZ:", original_size)
    print("Original spacing XYZ:", original_spacing)
    print("Original origin:", original_origin)

    # Check annotation physical coordinates against
    # the original image before resampling.
    for _, row in annotations.iterrows():

        point = (
            float(row["CoordX"]),
            float(row["CoordY"]),
            float(row["CoordZ"])
        )

        index = image.TransformPhysicalPointToContinuousIndex(
            point
        )

        if any(
            index[d] < -0.5
            or index[d] > original_size[d] - 0.5
            for d in range(3)
        ):
            raise RuntimeError(
                f"Annotation {row['AnnotationID']} "
                f"is outside the original MHA volume: {index}"
            )

    print("\nResampling volume...")

    resampled = resample_to_training_spacing(image)

    print("Resampled size XYZ:", resampled.GetSize())
    print("Resampled spacing XYZ:", resampled.GetSpacing())

    volume = sitk.GetArrayFromImage(
        resampled
    ).astype(np.float32)

    print("NumPy volume shape ZYX:", volume.shape)
    print("HU/intensity range:", volume.min(), volume.max())

    # Process every annotation belonging to this exact series.
    for _, row in annotations.iterrows():

        annotation_id = str(row["AnnotationID"])

        point = (
            float(row["CoordX"]),
            float(row["CoordY"]),
            float(row["CoordZ"])
        )

        # Transform physical LPS coordinates into the
        # resampled image's continuous XYZ index.
        continuous_xyz = np.asarray(
            resampled.TransformPhysicalPointToContinuousIndex(
                point
            ),
            dtype=np.float64
        )

        image_size = np.asarray(
            resampled.GetSize()
        )

        if any(
            continuous_xyz[d] < -0.5
            or continuous_xyz[d] > image_size[d] - 0.5
            for d in range(3)
        ):
            raise RuntimeError(
                f"Annotation {annotation_id} "
                "falls outside resampled image."
            )

        patch, center_xyz, start_zyx = extract_patch(
            volume,
            continuous_xyz
        )

        # Convert patch start from ZYX to XYZ.
        start_xyz = start_zyx[::-1]

        # Physical origin of the first patch voxel.
        patch_origin = resampled.TransformIndexToPhysicalPoint(
            tuple(int(x) for x in start_xyz)
        )

        # Annotation position relative to the patch.
        annotation_patch_zyx = (
            continuous_xyz[::-1] - start_zyx
        )

        # Save raw image intensities as int16 HU.
        patch_int16 = np.rint(
            patch
        ).clip(
            np.iinfo(np.int16).min,
            np.iinfo(np.int16).max
        ).astype(np.int16)

        output_patch = (
            OUTPUT_DIR / f"{annotation_id}.npy"
        )

        np.save(
            output_patch,
            patch_int16
        )

        # Calculate how much of the crop is outside
        # the original resampled volume.
        patch_shape = np.asarray(
            PATCH_SHAPE_ZYX
        )

        volume_shape = np.asarray(
            volume.shape
        )

        valid_start = np.maximum(
            start_zyx,
            0
        )

        valid_end = np.minimum(
            start_zyx + patch_shape,
            volume_shape
        )

        valid_shape = np.maximum(
            valid_end - valid_start,
            0
        )

        valid_voxels = int(
            np.prod(valid_shape)
        )

        padding_fraction = (
            1.0 - valid_voxels / patch_int16.size
        )

        metadata = {

            "PatientID": str(row["PatientID"]),

            "SeriesInstanceUID": uid,

            "AnnotationID": annotation_id,

            "NoduleID": str(row["NoduleID"]),

            "label": int(row["label"]),

            "patch_shape_zyx": list(PATCH_SHAPE_ZYX),

            "spacing_xyz": list(TARGET_SPACING_XYZ),

            "spacing_zyx": list(
                TARGET_SPACING_XYZ[::-1]
            ),

            "patch_origin_lps": list(patch_origin),

            "direction": list(
                resampled.GetDirection()
            ),

            "annotation_physical_lps": list(point),

            "annotation_resampled_xyz": (
                continuous_xyz.tolist()
            ),

            "annotation_patch_zyx": (
                annotation_patch_zyx.tolist()
            ),

            "patch_start_zyx": (
                start_zyx.tolist()
            ),

            "padding_fraction": float(
                padding_fraction
            ),

            "intensity_min": float(patch_int16.min()),

            "intensity_max": float(patch_int16.max()),

            "intensity_mean": float(patch_int16.mean()),

            "original_spacing_xyz": list(
                original_spacing
            ),

            "original_size_xyz": list(
                original_size
            ),

            "original_origin_lps": list(
                original_origin
            ),

            "resampling": "linear interpolation",

            "padding_value": PADDING_HU
        }

        output_json = (
            OUTPUT_DIR / f"{annotation_id}.json"
        )

        with open(output_json, "w") as file:
            json.dump(metadata, file, indent=4)

        print(
            f"\n{annotation_id}"
            f"\n  Label: {metadata['label']}"
            f"\n  Patch shape: {patch_int16.shape}"
            f"\n  Annotation position ZYX: "
            f"{annotation_patch_zyx}"
            f"\n  Padding: {padding_fraction:.2%}"
            f"\n  Intensity range: "
            f"{patch_int16.min()} to {patch_int16.max()}"
        )

    return len(annotations)


# ============================================================
# MAIN
# ============================================================

def main():

    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(ANNOTATION_FILE)

    annotations = pd.read_csv(
        ANNOTATION_FILE,
        dtype={
            "PatientID": str,
            "SeriesInstanceUID": str
        }
    )

    required_columns = [
        "PatientID",
        "SeriesInstanceUID",
        "AnnotationID",
        "NoduleID",
        "CoordX",
        "CoordY",
        "CoordZ",
        "label"
    ]

    missing = [
        col for col in required_columns
        if col not in annotations.columns
    ]

    if missing:
        raise ValueError(
            f"Missing annotation columns: {missing}"
        )

    print("Exact test annotations:", len(annotations))

    total = 0

    # One full-volume read and resampling per series,
    # even when that series contains multiple nodules.
    for uid, group in annotations.groupby(
        "SeriesInstanceUID",
        sort=True
    ):

        total += process_series(
            uid,
            group
        )

    print("\n" + "=" * 65)
    print("TEST PATCH EXTRACTION COMPLETED")
    print("=" * 65)

    print("Total patches:", total)
    print("Expected patch shape:", PATCH_SHAPE_ZYX)
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
