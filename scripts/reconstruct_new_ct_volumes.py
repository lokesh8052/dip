
from pathlib import Path
import numpy as np
import pydicom
import json

# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

CT_DIR = ROOT / "data" / "raw" / "ct_scans"
OUTPUT_DIR = ROOT / "data" / "processed" / "ct_volumes"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATIENTS = ["128669", "120708", "210559"]


# ==================================================
# LOAD AND SORT DICOM SLICES
# ==================================================

def load_dicom_series(patient):

    dicom_dir = CT_DIR / f"patient_{patient}_dicom"

    print(f"\nReading Patient {patient}...")

    datasets = []

    for path in dicom_dir.rglob("*"):

        if not path.is_file():
            continue

        try:
            ds = pydicom.dcmread(path, force=True)

            if (
                hasattr(ds, "PixelData")
                and hasattr(ds, "ImagePositionPatient")
                and hasattr(ds, "ImageOrientationPatient")
            ):
                datasets.append(ds)

        except Exception:
            continue

    if not datasets:
        raise RuntimeError(
            f"No valid DICOM slices found for Patient {patient}"
        )

    print("DICOM slices loaded:", len(datasets))

    # Verify single series.
    series_uids = {
        str(ds.SeriesInstanceUID)
        for ds in datasets
    }

    if len(series_uids) != 1:
        raise RuntimeError(
            f"Multiple series found: {series_uids}"
        )

    first = datasets[0]

    expected_shape = (
        int(first.Rows),
        int(first.Columns)
    )

    expected_orientation = np.asarray(
        first.ImageOrientationPatient,
        dtype=np.float64
    )

    # Check dimensions and orientation.
    for ds in datasets:

        current_shape = (
            int(ds.Rows),
            int(ds.Columns)
        )

        if current_shape != expected_shape:
            raise RuntimeError(
                f"Inconsistent dimensions: {current_shape}"
            )

        current_orientation = np.asarray(
            ds.ImageOrientationPatient,
            dtype=np.float64
        )

        if not np.allclose(
            current_orientation,
            expected_orientation,
            atol=1e-5
        ):
            raise RuntimeError(
                "Inconsistent slice orientation."
            )

    # Calculate slice normal.
    row_direction = expected_orientation[:3]
    column_direction = expected_orientation[3:]

    slice_normal = np.cross(
        row_direction,
        column_direction
    )

    def get_position(ds):

        position = np.asarray(
            ds.ImagePositionPatient,
            dtype=np.float64
        )

        return float(
            np.dot(position, slice_normal)
        )

    # Sort slices by physical position.
    datasets.sort(key=get_position)

    positions = np.array([
        get_position(ds)
        for ds in datasets
    ])

    differences = np.diff(positions)

    if len(differences) and np.any(differences <= 0):
        raise RuntimeError(
            "Duplicate or incorrectly ordered slice positions."
        )

    return datasets, slice_normal


# ==================================================
# DICOM PADDING MASK
# ==================================================

def get_padding_mask(ds, pixels):

    padding_value = getattr(
        ds,
        "PixelPaddingValue",
        None
    )

    padding_range = getattr(
        ds,
        "PixelPaddingRangeLimit",
        None
    )

    mask = np.zeros(
        pixels.shape,
        dtype=bool
    )

    if padding_value is None:
        return mask

    signed = int(
        getattr(ds, "PixelRepresentation", 0)
    ) == 1

    padding_value = int(padding_value)

    if signed and padding_value > 32767:
        padding_value -= 65536

    if padding_range is None:

        mask = pixels == padding_value

    else:

        padding_range = int(padding_range)

        if signed and padding_range > 32767:
            padding_range -= 65536

        lower = min(padding_value, padding_range)
        upper = max(padding_value, padding_range)

        mask = (
            (pixels >= lower) &
            (pixels <= upper)
        )

    return mask


# ==================================================
# RECONSTRUCT 3D HU VOLUME
# ==================================================

def reconstruct_volume(datasets):

    first = datasets[0]

    rows = int(first.Rows)
    columns = int(first.Columns)

    number_of_slices = len(datasets)

    volume = np.empty(
        (number_of_slices, rows, columns),
        dtype=np.float32
    )

    total_padding_pixels = 0

    for index, ds in enumerate(datasets):

        pixels = ds.pixel_array.astype(
            np.float32
        )

        slope = float(
            getattr(ds, "RescaleSlope", 1)
        )

        intercept = float(
            getattr(ds, "RescaleIntercept", 0)
        )

        padding_mask = get_padding_mask(
            ds,
            pixels
        )

        # Convert stored pixels to HU.
        hu_slice = pixels * slope + intercept

        # Replace padding with -1000 HU.
        padding_count = int(
            padding_mask.sum()
        )

        total_padding_pixels += padding_count

        hu_slice[padding_mask] = -1000.0

        volume[index] = hu_slice

        if (index + 1) % 25 == 0:

            print(
                f"\rProcessed {index + 1}/"
                f"{number_of_slices} slices",
                end="",
                flush=True
            )

    print()

    print(
        "Total padding pixels replaced:",
        total_padding_pixels
    )

    return volume, total_padding_pixels


# ==================================================
# GENERATE METADATA
# ==================================================

def create_metadata(
    patient,
    datasets,
    slice_normal,
    volume,
    padding_count
):

    first = datasets[0]

    positions = np.array([
        np.dot(
            np.asarray(
                ds.ImagePositionPatient,
                dtype=np.float64
            ),
            slice_normal
        )
        for ds in datasets
    ])

    differences = np.diff(positions)

    slice_spacing = (
        float(np.median(np.abs(differences)))
        if len(differences)
        else None
    )

    metadata = {

        "PatientID": patient,

        "SeriesInstanceUID": str(
            first.SeriesInstanceUID
        ),

        "StudyInstanceUID": str(
            first.StudyInstanceUID
        ),

        "StudyDate": str(
            getattr(first, "StudyDate", "")
        ),

        "Modality": str(
            getattr(first, "Modality", "")
        ),

        "NumberOfSlices": len(datasets),

        "VolumeShape": list(volume.shape),

        "ImageRows": int(first.Rows),

        "ImageColumns": int(first.Columns),

        "PixelSpacing": [
            float(x)
            for x in first.PixelSpacing
        ],

        "SliceThickness": float(
            getattr(first, "SliceThickness", 0)
        ),

        "SpacingBetweenSlices": slice_spacing,

        "ImageOrientationPatient": [
            float(x)
            for x in first.ImageOrientationPatient
        ],

        "FirstImagePositionPatient": [
            float(x)
            for x in first.ImagePositionPatient
        ],

        "SlicePositionRange": [
            float(positions.min()),
            float(positions.max())
        ],

        "HU_Min": float(volume.min()),

        "HU_Max": float(volume.max()),

        "HU_Mean": float(volume.mean()),

        "PaddingPixelsReplaced": padding_count,

        "DataType": str(volume.dtype)
    }

    return metadata


# ==================================================
# PROCESS PATIENT
# ==================================================

def process_patient(patient):

    print("\n" + "=" * 60)
    print(f"PATIENT {patient}")
    print("=" * 60)

    output_file = (
        OUTPUT_DIR / f"patient_{patient}.npz"
    )

    metadata_file = (
        OUTPUT_DIR / f"patient_{patient}.json"
    )

    datasets, slice_normal = load_dicom_series(
        patient
    )

    volume, padding_count = reconstruct_volume(
        datasets
    )

    metadata = create_metadata(
        patient,
        datasets,
        slice_normal,
        volume,
        padding_count
    )

    # Save compressed volume.
    np.savez_compressed(
        output_file,
        volume=volume
    )

    with open(metadata_file, "w") as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print("\nVolume saved:", output_file)
    print("Metadata saved:", metadata_file)

    print("\nVALIDATION RESULTS")

    print("Shape:", volume.shape)
    print("Dtype:", volume.dtype)
    print("HU minimum:", volume.min())
    print("HU maximum:", volume.max())
    print("HU mean:", volume.mean())

    print(
        "Pixel spacing:",
        metadata["PixelSpacing"]
    )

    print(
        "Slice spacing:",
        metadata["SpacingBetweenSlices"]
    )

    print(
        "Padding pixels replaced:",
        padding_count
    )

    # Correct sanity checks.
    assert np.isfinite(volume).all()

    assert volume.shape == (
        len(datasets),
        int(datasets[0].Rows),
        int(datasets[0].Columns)
    )

    print("\nAll sanity checks passed.")


# ==================================================
# MAIN
# ==================================================

def main():

    for patient in PATIENTS:
        process_patient(patient)

    print("\n" + "=" * 60)
    print("ALL THREE CT VOLUMES RECONSTRUCTED")
    print("=" * 60)


if __name__ == "__main__":
    main()
