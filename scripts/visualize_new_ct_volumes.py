
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json

# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

VOLUME_DIR = ROOT / "data" / "processed" / "ct_volumes"

OUTPUT_DIR = ROOT / "outputs" / "ct_visualization"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATIENTS = ["128669", "120708", "210559"]


# ==================================================
# LOAD VOLUME AND METADATA
# ==================================================

def load_patient(patient):

    volume_path = VOLUME_DIR / f"patient_{patient}.npz"
    metadata_path = VOLUME_DIR / f"patient_{patient}.json"

    if not volume_path.exists():
        raise FileNotFoundError(volume_path)

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    data = np.load(volume_path)

    volume = data["volume"]

    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    return volume, metadata


# ==================================================
# VISUALIZE THREE PLANES
# ==================================================

def visualize_patient(patient):

    print("\n" + "=" * 55)
    print(f"VISUALIZING PATIENT {patient}")
    print("=" * 55)

    volume, metadata = load_patient(patient)

    depth, height, width = volume.shape

    # Select central slices.
    axial_index = depth // 2
    coronal_index = height // 2
    sagittal_index = width // 2

    axial = volume[axial_index, :, :]
    coronal = volume[:, coronal_index, :]
    sagittal = volume[:, :, sagittal_index]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 6)
    )

    # Axial view.
    axes[0].imshow(
        axial,
        cmap="gray",
        vmin=-1000,
        vmax=400,
        origin="lower"
    )

    axes[0].set_title(
        f"Axial View (Slice {axial_index})"
    )

    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")

    # Coronal view.
    axes[1].imshow(
        coronal,
        cmap="gray",
        vmin=-1000,
        vmax=400,
        aspect="auto",
        origin="lower"
    )

    axes[1].set_title(
        f"Coronal View (Slice {coronal_index})"
    )

    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Slice")

    # Sagittal view.
    axes[2].imshow(
        sagittal,
        cmap="gray",
        vmin=-1000,
        vmax=400,
        aspect="auto",
        origin="lower"
    )

    axes[2].set_title(
        f"Sagittal View (Slice {sagittal_index})"
    )

    axes[2].set_xlabel("Row")
    axes[2].set_ylabel("Slice")

    fig.suptitle(
        f"Patient {patient} - CT Volume Visualization",
        fontsize=16
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR / f"patient_{patient}_three_views.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    print("Volume shape:", volume.shape)
    print("Study date:", metadata["StudyDate"])
    print("Saved visualization:", output_path)

    plt.show()
    plt.close(fig)


# ==================================================
# MAIN
# ==================================================

def main():

    for patient in PATIENTS:
        visualize_patient(patient)

    print("\nAll CT visualizations completed.")


if __name__ == "__main__":
    main()
