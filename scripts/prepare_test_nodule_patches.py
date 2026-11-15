
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pydicom

from scipy.ndimage import map_coordinates


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

CT_DIR = ROOT / "data/raw/ct_scans"
VOLUME_DIR = ROOT / "data/processed/ct_volumes"

ANNOTATION_FILE = (
    ROOT / "data/raw/annotations/"
    "LUNA25_Public_Training_Development_Data.csv"
)

OUTPUT_DIR = ROOT / "data/processed/test_nodule_patches"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATIENTS = ["128669", "120708", "210559"]

# Original training patch geometry.
TARGET_SPACING = np.array(
    [2.0, 0.5859375, 0.5859375],
    dtype=np.float64
)

PATCH_SHAPE = (64, 128, 128)

# Padding outside the original CT volume.
PADDING_HU = -1000.0


# ============================================================
# LOAD DICOM GEOMETRY
# ============================================================

def load_dicom_geometry(patient, expected_uid):

    dicom_dir = CT_DIR / f"patient_{patient}_dicom"

    datasets = []

    for path in dicom_dir.rglob("*"):

        if not path.is_file():
            continue

        try:
            ds = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                force=True
            )

            if not hasattr(ds, "ImagePositionPatient"):
                continue

            if not hasattr(ds, "ImageOrientationPatient"):
                continue

            if str(ds.SeriesInstanceUID) != str(expected_uid):
                continue

            datasets.append(ds)

        except Exception:
            continue

    if not datasets:
        raise RuntimeError(
            f"No DICOM geometry found for Patient {patient}"
        )

    if len({str(ds.SeriesInstanceUID) for ds in datasets}) != 1:
        raise RuntimeError("Multiple DICOM series detected.")

    first = datasets[0]

    orientation = np.asarray(
        first.ImageOrientationPatient,
        dtype=np.float64
    )

    row_direction = orientation[:3]
    column_direction = orientation[3:]

    normal = np.cross(
        row_direction,
        column_direction
    )

    def get_position(ds):

        position = np.asarray(
            ds.ImagePositionPatient,
            dtype=np.float64
        )

        return float(np.dot(position, normal))

    datasets.sort(key=get_position)

    positions = np.array(
        [get_position(ds) for ds in datasets],
        dtype=np.float64
    )

    differences = np.diff(positions)

    if len(differences) == 0:
        raise RuntimeError("Insufficient slices.")

    if np.any(differences <= 0):
        raise RuntimeError("Invalid slice ordering.")

    spacing = float(np.median(differences))

    # Check for irregular slice spacing.
    if not np.allclose(
        differences,
        spacing,
        atol=0.05,
        rtol=0
    ):
        raise RuntimeError(
            "Irregular slice spacing detected. "
            "A dedicated irregular-grid resampler is required."
        )

    pixel_spacing = np.asarray(
        first.PixelSpacing,
        dtype=np.float64
    )

    return {
        "datasets": datasets,
        "positions": positions,
        "origin": np.asarray(
            first.ImagePositionPatient,
            dtype=np.float64
        ),
        "normal": normal,
        "row_direction": row_direction,
        "column_direction": column_direction,
        "pixel_spacing": pixel_spacing,
        "slice_spacing": spacing,
        "orientation": orientation
    }


# ============================================================
# LOAD ANNOTATION
# ============================================================

def get_annotation(patient, expected_uid, annotations):

    rows = annotations[
        (annotations["PatientID"].astype(str) == patient)
        &
        (
            annotations["SeriesInstanceUID"].astype(str)
            == str(expected_uid)
        )
    ]

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one matching annotation "
            f"for Patient {patient}, found {len(rows)}"
        )

    return rows.iloc[0]


# ============================================================
# PHYSICAL COORDINATE TO SOURCE VOXEL
# ============================================================

def annotation_to_source_voxel(annotation, geometry):

    point = np.array(
        [
            annotation["CoordX"],
            annotation["CoordY"],
            annotation["CoordZ"]
        ],
        dtype=np.float64
    )

    origin = geometry["origin"]
    normal = geometry["normal"]

    row_direction = geometry["row_direction"]
    column_direction = geometry["column_direction"]

    row_spacing, column_spacing = geometry["pixel_spacing"]

    displacement = point - origin

    target_slice_position = float(
        np.dot(point, normal)
    )

    positions = geometry["positions"]

    slice_index = float(
        np.interp(
            target_slice_position,
            positions,
            np.arange(len(positions), dtype=np.float64)
        )
    )

    row = float(
        np.dot(displacement, column_direction)
        / row_spacing
    )

    column = float(
        np.dot(displacement, row_direction)
        / column_spacing
    )

    nearest_slice = int(round(slice_index))

    slice_distance = abs(
        target_slice_position - positions[nearest_slice]
    )

    return np.array(
        [slice_index, row, column],
        dtype=np.float64
    ), slice_distance


# ============================================================
# SPACING-AWARE RESAMPLING
# ============================================================

def resample_volume(volume, geometry):

    print("\nResampling CT volume...")

    source_spacing = np.array(
        [
            geometry["slice_spacing"],
            geometry["pixel_spacing"][0],
            geometry["pixel_spacing"][1]
        ],
        dtype=np.float64
    )

    target_spacing = TARGET_SPACING

    source_shape = np.array(
        volume.shape,
        dtype=np.int64
    )

    # Preserve the first voxel centre and physical extent.
    target_shape = (
        np.round(
            (source_shape - 1)
            * source_spacing
            / target_spacing
        ).astype(np.int64)
        + 1
    )

    target_shape = tuple(
        int(x) for x in target_shape
    )

    print("Original shape:", volume.shape)
    print("Original spacing:", source_spacing)
    print("Target shape:", target_shape)
    print("Target spacing:", target_spacing)

    output = np.empty(
        target_shape,
        dtype=np.float32
    )

    # Target row and column positions, in millimetres.
    target_rows = (
        np.arange(target_shape[1], dtype=np.float64)
        * target_spacing[1]
    )

    target_columns = (
        np.arange(target_shape[2], dtype=np.float64)
        * target_spacing[2]
    )

    source_rows = target_rows / source_spacing[1]
    source_columns = target_columns / source_spacing[2]

    row_grid, column_grid = np.meshgrid(
        source_rows,
        source_columns,
        indexing="ij"
    )

    positions = geometry["positions"]

    # Process one target slice at a time to limit memory use.
    for target_z in range(target_shape[0]):

        target_position = (
            positions[0]
            + target_z * target_spacing[0]
        )

        source_z = np.interp(
            target_position,
            positions,
            np.arange(len(positions), dtype=np.float64)
        )

        coordinates = np.stack(
            [
                np.full_like(row_grid, source_z),
                row_grid,
                column_grid
            ],
            axis=0
        )

        output[target_z] = map_coordinates(
            volume,
            coordinates,
            order=1,
            mode="nearest",
            prefilter=False
        )

        if (target_z + 1) % 25 == 0:
            print(
                f"\rResampled {target_z + 1}/"
                f"{target_shape[0]} slices",
                end="",
                flush=True
            )

    print("\nResampling completed.")

    return output


# ============================================================
# MAP ANNOTATION INTO RESAMPLED VOLUME
# ============================================================

def annotation_to_resampled_voxel(
    annotation,
    geometry
):

    point = np.array(
        [
            annotation["CoordX"],
            annotation["CoordY"],
            annotation["CoordZ"]
        ],
        dtype=np.float64
    )

    displacement = point - geometry["origin"]

    normal = geometry["normal"]

    row_direction = geometry["row_direction"]
    column_direction = geometry["column_direction"]

    row_spacing, column_spacing = TARGET_SPACING[1:]

    positions = geometry["positions"]

    target_position = float(
        np.dot(point, normal)
    )

    z = (
        target_position - positions[0]
    ) / TARGET_SPACING[0]

    row = (
        np.dot(displacement, column_direction)
        / row_spacing
    )

    column = (
        np.dot(displacement, row_direction)
        / column_spacing
    )

    return np.array(
        [z, row, column],
        dtype=np.float64
    )


# ============================================================
# EXTRACT ANNOTATION-CENTRED PATCH
# ============================================================

def extract_patch(volume, center):

    center_index = np.rint(center).astype(int)

    depth, height, width = PATCH_SHAPE

    starts = (
        center_index
        - np.array(
            [
                depth // 2,
                height // 2,
                width // 2
            ]
        )
    )

    ends = starts + np.array(PATCH_SHAPE)

    padding = []

    for start, end, dimension in zip(
        starts,
        ends,
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

    adjusted_starts = (
        starts + np.array(
            [p[0] for p in padding]
        )
    )

    patch = padded[
        adjusted_starts[0]:adjusted_starts[0] + depth,
        adjusted_starts[1]:adjusted_starts[1] + height,
        adjusted_starts[2]:adjusted_starts[2] + width
    ]

    if patch.shape != PATCH_SHAPE:
        raise RuntimeError(
            f"Unexpected patch shape: {patch.shape}"
        )

    return patch.astype(np.int16), center_index, starts


# ============================================================
# PROCESS ONE PATIENT
# ============================================================

def process_patient(patient, annotations):

    print("\n" + "=" * 65)
    print("PATIENT:", patient)
    print("=" * 65)

    volume_path = (
        VOLUME_DIR / f"patient_{patient}.npz"
    )

    metadata_path = (
        VOLUME_DIR / f"patient_{patient}.json"
    )

    if not volume_path.exists():
        raise FileNotFoundError(volume_path)

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    with open(metadata_path) as file:
        volume_metadata = json.load(file)

    expected_uid = volume_metadata["SeriesInstanceUID"]

    annotation = get_annotation(
        patient,
        expected_uid,
        annotations
    )

    geometry = load_dicom_geometry(
        patient,
        expected_uid
    )

    print("Series UID verified.")
    print("Annotation:", annotation["AnnotationID"])
    print("Label:", int(annotation["label"]))

    volume = np.load(volume_path)["volume"].astype(
        np.float32
    )

    if len(geometry["datasets"]) != volume.shape[0]:
        raise RuntimeError(
            "DICOM slice count does not match reconstructed volume."
        )

    # Check orientation is the expected axis-aligned orientation.
    expected_orientation = np.array(
        [1, 0, 0, 0, 1, 0],
        dtype=np.float64
    )

    if not np.allclose(
        geometry["orientation"],
        expected_orientation,
        atol=1e-5
    ):
        raise RuntimeError(
            "Unexpected orientation. "
            "This script currently expects the verified "
            "axis-aligned CT orientation."
        )

    source_center, original_slice_distance = (
        annotation_to_source_voxel(
            annotation,
            geometry
        )
    )

    print("\nSource voxel coordinates:")
    print(source_center)

    print(
        "Original nearest-slice distance:",
        round(original_slice_distance, 3),
        "mm"
    )

    # Resample to the training patch spacing.
    resampled = resample_volume(
        volume,
        geometry
    )

    # Transform annotation into the new grid.
    target_center = annotation_to_resampled_voxel(
        annotation,
        geometry
    )

    print("\nResampled annotation coordinates:")
    print(target_center)

    patch, center_index, patch_start = extract_patch(
        resampled,
        target_center
    )

    # Save raw HU patch.
    output_patch = (
        OUTPUT_DIR / f"patient_{patient}.npy"
    )

    np.save(
        output_patch,
        patch
    )

    # Physical location of the first patch voxel.
    patch_origin = (
        geometry["origin"]
        + geometry["normal"]
        * patch_start[0]
        * TARGET_SPACING[0]
        + geometry["column_direction"]
        * patch_start[1]
        * TARGET_SPACING[1]
        + geometry["row_direction"]
        * patch_start[2]
        * TARGET_SPACING[2]
    )

    metadata = {

        "PatientID": patient,

        "AnnotationID": str(
            annotation["AnnotationID"]
        ),

        "NoduleID": str(
            annotation["NoduleID"]
        ),

        "SeriesInstanceUID": expected_uid,

        "label": int(annotation["label"]),

        "patch_shape": list(PATCH_SHAPE),

        "spacing": TARGET_SPACING.tolist(),

        "origin": patch_origin.tolist(),

        "transform": np.eye(3).tolist(),

        "annotation_patient_coordinates": [
            float(annotation["CoordX"]),
            float(annotation["CoordY"]),
            float(annotation["CoordZ"])
        ],

        "annotation_resampled_voxel": (
            target_center.tolist()
        ),

        "annotation_patch_voxel": (
            (target_center - patch_start).tolist()
        ),

        "rounded_center_voxel": (
            center_index.tolist()
        ),

        "patch_start_voxel": (
            patch_start.tolist()
        ),

        "original_slice_distance_mm": float(
            original_slice_distance
        ),

        "HU_min": float(patch.min()),

        "HU_max": float(patch.max()),

        "HU_mean": float(patch.mean()),

        "data_type": str(patch.dtype),

        "resampling": "linear interpolation",

        "padding_HU": PADDING_HU
    }

    output_metadata = (
        OUTPUT_DIR / f"patient_{patient}.json"
    )

    with open(output_metadata, "w") as file:
        json.dump(metadata, file, indent=4)

    # Same intensity transformation as LungDataset.
    normalized = np.clip(
        patch.astype(np.float32),
        -1000,
        400
    )

    normalized = (
        normalized + 1000.0
    ) / 1400.0

    assert patch.shape == PATCH_SHAPE
    assert np.isfinite(patch).all()
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0

    print("\nPATCH RESULTS")
    print("Patch shape:", patch.shape)
    print("Patch dtype:", patch.dtype)
    print("HU minimum:", patch.min())
    print("HU maximum:", patch.max())
    print("HU mean:", patch.mean())
    print("Normalized range:", normalized.min(), normalized.max())

    print("\nSaved patch:", output_patch)
    print("Saved metadata:", output_metadata)

    return metadata


# ============================================================
# MAIN
# ============================================================

def main():

    annotations = pd.read_csv(
        ANNOTATION_FILE,
        dtype={"PatientID": str}
    )

    results = []

    for patient in PATIENTS:

        metadata = process_patient(
            patient,
            annotations
        )

        results.append(metadata)

    summary_path = OUTPUT_DIR / "summary.csv"

    pd.DataFrame(results).to_csv(
        summary_path,
        index=False
    )

    print("\n" + "=" * 65)
    print("ALL TEST NODULE PATCHES PREPARED")
    print("=" * 65)

    print("Output directory:", OUTPUT_DIR)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
