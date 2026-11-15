
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# ==================================================
# 1. CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

MHA_PATH = Path(
    "/mnt/c/Users/balla/lunger_cancer/luna25_images/"
    "1.2.840.113654.2.55.109935711018211239275977773631810109196.mha"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "ct_volumes"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "patient_106046.npz"
METADATA_PATH = OUTPUT_DIR / "patient_106046.json"

TARGET_SPACING_XYZ = (
    0.5859375,
    0.5859375,
    2.0
)

PADDING_HU = -1000.0

# ==================================================
# 2. LOAD ORIGINAL MHA
# ==================================================

if not MHA_PATH.exists():
    raise FileNotFoundError(MHA_PATH)

print("Reading Patient 106046 CT scan...")

image = sitk.ReadImage(str(MHA_PATH))

original_size = image.GetSize()
original_spacing = image.GetSpacing()
original_origin = image.GetOrigin()
original_direction = image.GetDirection()

print("Original size XYZ:", original_size)
print("Original spacing XYZ:", original_spacing)
print("Original origin:", original_origin)

# ==================================================
# 3. CALCULATE RESAMPLED IMAGE SIZE
# ==================================================

original_size_array = np.asarray(
    original_size,
    dtype=np.int64
)

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

target_size = tuple(
    int(x) for x in target_size
)

print("\nTarget size XYZ:", target_size)

# ==================================================
# 4. RESAMPLE CT VOLUME
# ==================================================

print("\nResampling CT volume...")

resampler = sitk.ResampleImageFilter()

resampler.SetSize(target_size)
resampler.SetOutputSpacing(TARGET_SPACING_XYZ)
resampler.SetOutputOrigin(original_origin)
resampler.SetOutputDirection(original_direction)

resampler.SetTransform(
    sitk.Transform(3, sitk.sitkIdentity)
)

resampler.SetInterpolator(
    sitk.sitkLinear
)

resampler.SetDefaultPixelValue(PADDING_HU)
resampler.SetOutputPixelType(
    sitk.sitkFloat32
)

resampled = resampler.Execute(image)

# ==================================================
# 5. CONVERT TO NUMPY VOLUME
# ==================================================

volume = sitk.GetArrayFromImage(
    resampled
)

print("Resampled volume shape ZYX:", volume.shape)
print("Intensity range:", volume.min(), volume.max())

# Convert to int16 HU, matching training patch storage.
volume = np.rint(volume).clip(
    np.iinfo(np.int16).min,
    np.iinfo(np.int16).max
).astype(np.int16)

# ==================================================
# 6. SAVE RESAMPLED VOLUME
# ==================================================

print("\nSaving resampled volume...")

np.savez_compressed(
    OUTPUT_PATH,
    volume=volume
)

# ==================================================
# 7. SAVE PHYSICAL METADATA
# ==================================================

metadata = {
    "PatientID": "106046",
    "SeriesInstanceUID": (
        "1.2.840.113654.2.55."
        "109935711018211239275977773631810109196"
    ),
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

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print("\n======================================")
print("PATIENT 106046 PREPARATION COMPLETED")
print("======================================")

print("Volume:", OUTPUT_PATH)
print("Metadata:", METADATA_PATH)
print("Volume shape:", volume.shape)
print("Data type:", volume.dtype)
print("Intensity range:", volume.min(), "to", volume.max())
