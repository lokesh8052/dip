
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


TARGET_SPACING_XYZ = (0.5859375, 0.5859375, 2.0)
PADDING_HU = -1000.0


def preprocess_mha(mha_path, output_dir):
    mha_path = Path(mha_path)
    output_dir = Path(output_dir)

    if not mha_path.exists():
        raise FileNotFoundError(f"CT file not found: {mha_path}")

    if mha_path.suffix.lower() != ".mha":
        raise ValueError("Input must be an .mha file.")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading CT scan: {mha_path.name}")

    image = sitk.ReadImage(str(mha_path))

    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    original_origin = image.GetOrigin()
    original_direction = image.GetDirection()

    original_size_array = np.asarray(original_size, dtype=np.int64)
    original_spacing_array = np.asarray(
        original_spacing,
        dtype=np.float64
    )
    target_spacing_array = np.asarray(
        TARGET_SPACING_XYZ,
        dtype=np.float64
    )

    target_size = (
        np.round(
            (original_size_array - 1)
            * original_spacing_array
            / target_spacing_array
        ).astype(np.int64)
        + 1
    )

    target_size = tuple(int(x) for x in target_size)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(TARGET_SPACING_XYZ)
    resampler.SetOutputOrigin(original_origin)
    resampler.SetOutputDirection(original_direction)
    resampler.SetTransform(
        sitk.Transform(3, sitk.sitkIdentity)
    )
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(PADDING_HU)
    resampler.SetOutputPixelType(sitk.sitkFloat32)

    resampled = resampler.Execute(image)

    volume = sitk.GetArrayFromImage(resampled)

    if volume.ndim != 3 or any(dim == 0 for dim in volume.shape):
        raise ValueError(
            f"Invalid resampled volume shape: {volume.shape}"
        )

    volume = np.rint(volume).clip(
        np.iinfo(np.int16).min,
        np.iinfo(np.int16).max
    ).astype(np.int16)

    output_path = output_dir / f"{mha_path.stem}.npz"
    metadata_path = output_dir / f"{mha_path.stem}.json"

    np.savez_compressed(
        output_path,
        volume=volume
    )

    metadata = {
        "source_filename": mha_path.name,
        "original_size_xyz": list(original_size),
        "original_spacing_xyz": list(original_spacing),
        "original_origin_lps": list(original_origin),
        "original_direction": list(original_direction),
        "resampled_size_xyz": list(resampled.GetSize()),
        "resampled_spacing_xyz": list(resampled.GetSpacing()),
        "resampled_origin_lps": list(resampled.GetOrigin()),
        "resampled_direction": list(resampled.GetDirection()),
        "volume_shape_zyx": list(volume.shape),
        "intensity_min": int(volume.min()),
        "intensity_max": int(volume.max()),
        "interpolation": "linear",
        "padding_value_hu": PADDING_HU
    }

    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=4)

    return volume, metadata, output_path, metadata_path
