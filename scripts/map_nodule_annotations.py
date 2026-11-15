
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import json

# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

CT_DIR = ROOT / "data" / "raw" / "ct_scans"
VOLUME_DIR = ROOT / "data" / "processed" / "ct_volumes"

ANNOTATION_FILE = (
    ROOT / "data" / "raw" / "annotations"
    / "LUNA25_Public_Training_Development_Data.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "nodule_localization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATIENTS = ["128669", "120708", "210559"]


# ==================================================
# LOAD DICOM SERIES
# ==================================================

def load_dicom_series(patient, expected_uid):

    folder = CT_DIR / f"patient_{patient}_dicom"

    datasets = []

    for path in folder.rglob("*"):

        if not path.is_file():
            continue

        try:
            ds = pydicom.dcmread(path, force=True)

            if (
                hasattr(ds, "PixelData")
                and hasattr(ds, "ImagePositionPatient")
                and hasattr(ds, "ImageOrientationPatient")
            ):
                if str(ds.SeriesInstanceUID) == str(expected_uid):
                    datasets.append(ds)

        except Exception:
            continue

    if not datasets:
        raise RuntimeError(
            f"No DICOM slices found for expected series {expected_uid}"
        )

    # Verify that all loaded slices belong to the same series.
    series_uids = {
        str(ds.SeriesInstanceUID) for ds in datasets
    }

    if series_uids != {str(expected_uid)}:
        raise RuntimeError("Series UID mismatch.")

    first = datasets[0]

    orientation = np.asarray(
        first.ImageOrientationPatient,
        dtype=np.float64
    )

    row_direction = orientation[:3]
    column_direction = orientation[3:]

    normal = np.cross(row_direction, column_direction)

    def slice_position(ds):
        position = np.asarray(
            ds.ImagePositionPatient,
            dtype=np.float64
        )
        return float(np.dot(position, normal))

    datasets.sort(key=slice_position)

    positions = np.array([
        slice_position(ds) for ds in datasets
    ])

    if len(positions) > 1 and np.any(np.diff(positions) <= 0):
        raise RuntimeError("Invalid or duplicate slice positions.")

    return datasets, normal, positions


# ==================================================
# PATIENT COORDINATE TO VOXEL COORDINATE
# ==================================================

def patient_to_voxel(annotation, datasets, normal, positions):

    first = datasets[0]

    origin = np.asarray(
        first.ImagePositionPatient,
        dtype=np.float64
    )

    orientation = np.asarray(
        first.ImageOrientationPatient,
        dtype=np.float64
    )

    row_direction = orientation[:3]
    column_direction = orientation[3:]

    pixel_spacing = np.asarray(
        first.PixelSpacing,
        dtype=np.float64
    )

    # Annotation coordinates are in DICOM patient space (mm).
    point = np.array([
        annotation["CoordX"],
        annotation["CoordY"],
        annotation["CoordZ"]
    ], dtype=np.float64)

    displacement = point - origin

    # DICOM PixelSpacing = [row spacing, column spacing].
    row = np.dot(displacement, column_direction) / pixel_spacing[0]

    column = np.dot(displacement, row_direction) / pixel_spacing[1]

    # Find the nearest actual slice using physical position.
    target_position = np.dot(point, normal)

    slice_index = int(
        np.argmin(np.abs(positions - target_position))
    )

    slice_distance = abs(
        positions[slice_index] - target_position
    )

    return slice_index, row, column, slice_distance


# ==================================================
# VISUALIZE ANNOTATED NODULE
# ==================================================

def visualize_annotation(patient, annotation, datasets, normal, positions):

    z, row, column, slice_distance = patient_to_voxel(
        annotation,
        datasets,
        normal,
        positions
    )

    volume_path = VOLUME_DIR / f"patient_{patient}.npz"

    volume = np.load(volume_path)["volume"]

    print("\nPatient:", patient)
    print("Annotation ID:", annotation["AnnotationID"])
    print("Series UID:", annotation["SeriesInstanceUID"])
    print("Label:", annotation["label"])

    print("\nMapped voxel coordinates:")
    print("Slice (Z):", z)
    print("Row (Y):", round(row, 2))
    print("Column (X):", round(column, 2))
    print("Distance to nearest slice:", round(slice_distance, 3), "mm")

    # Verify that the point falls inside the image.
    if not (
        0 <= row < volume.shape[1]
        and 0 <= column < volume.shape[2]
    ):
        print("WARNING: Annotation falls outside image boundaries.")
        return

    # Display axial slice with annotation marker.
    image = volume[z]

    fig, ax = plt.subplots(figsize=(9, 9))

    ax.imshow(
        image,
        cmap="gray",
        vmin=-1000,
        vmax=400,
        origin="upper"
    )

    ax.scatter(
        column,
        row,
        color="red",
        marker="x",
        s=150,
        linewidths=3,
        label="Annotated nodule"
    )

    circle = plt.Circle(
        (column, row),
        radius=10,
        color="yellow",
        fill=False,
        linewidth=1.5
    )

    ax.add_patch(circle)

    ax.set_title(
        f"Patient {patient} - Annotated Nodule\n"
        f"Slice {z}, Row {row:.1f}, Column {column:.1f}"
    )

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.legend()

    output_path = (
        OUTPUT_DIR / f"patient_{patient}_nodule_location.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    print("Saved:", output_path)

    plt.show()
    plt.close(fig)


# ==================================================
# MAIN
# ==================================================

def main():

    annotations = pd.read_csv(ANNOTATION_FILE)

    for patient in PATIENTS:

        print("\n" + "=" * 65)
        print("PATIENT", patient)
        print("=" * 65)

        rows = annotations[
            annotations["PatientID"].astype(str) == patient
        ]

        if rows.empty:
            print("No annotation found.")
            continue

        # Read the series UID from the reconstructed metadata.
        metadata_path = (
            VOLUME_DIR / f"patient_{patient}.json"
        )

        with open(metadata_path, "r") as file:
            metadata = json.load(file)

        expected_uid = metadata["SeriesInstanceUID"]

        # Use only annotations belonging to this exact series.
        rows = rows[
            rows["SeriesInstanceUID"].astype(str) == expected_uid
        ]

        if rows.empty:
            print("No annotation matches the reconstructed series UID.")
            continue

        if len(rows) > 1:
            print("Multiple matching annotations found.")

        datasets, normal, positions = load_dicom_series(
            patient,
            expected_uid
        )

        print("Verified DICOM slices:", len(datasets))
        print("Annotation matches:", len(rows))

        for _, annotation in rows.iterrows():
            visualize_annotation(
                patient,
                annotation,
                datasets,
                normal,
                positions
            )

    print("\nAll annotation mapping operations completed.")


if __name__ == "__main__":
    main()
