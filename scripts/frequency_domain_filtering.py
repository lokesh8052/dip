
import os
import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# 1. LOAD CT VOLUME
# ==================================================

volume_path = "data/processed/ct_volumes/patient_100012.npz"

volume = np.load(volume_path)["volume"]

print("CT volume shape:", volume.shape)

# Nodule location (Z, Row, Column)
z, row, col = 125, 313, 199

# Extract axial slice
original = volume[z, :, :].astype(np.float32)

print("Original slice shape:", original.shape)

# ==================================================
# 2. COMPUTE 2D FOURIER TRANSFORM
# ==================================================

# Transform image from spatial domain to frequency domain
fft_image = np.fft.fft2(original)

# Shift zero frequency to the center
fft_shifted = np.fft.fftshift(fft_image)

# Displayable magnitude spectrum
magnitude_spectrum = 20 * np.log(
    np.abs(fft_shifted) + 1
)

print("2D FFT completed.")

# ==================================================
# 3. CREATE GAUSSIAN FREQUENCY FILTERS
# ==================================================

height, width = original.shape

center_y = height // 2
center_x = width // 2

y, x = np.ogrid[:height, :width]

distance = np.sqrt(
    (x - center_x) ** 2 +
    (y - center_y) ** 2
)

# Gaussian cutoff frequency
D0 = 30.0

# Gaussian low-pass filter
low_pass_filter = np.exp(
    -(distance ** 2) / (2 * D0 ** 2)
)

# Gaussian high-pass filter
high_pass_filter = 1 - low_pass_filter

# ==================================================
# 4. APPLY FREQUENCY FILTERS
# ==================================================

# Low-frequency components
low_pass_fft = fft_shifted * low_pass_filter

# High-frequency components
high_pass_fft = fft_shifted * high_pass_filter

# ==================================================
# 5. CONVERT BACK TO SPATIAL DOMAIN
# ==================================================

low_pass_image = np.fft.ifft2(
    np.fft.ifftshift(low_pass_fft)
).real

high_pass_image = np.fft.ifft2(
    np.fft.ifftshift(high_pass_fft)
).real

print("Low-pass and high-pass filtering completed.")

# ==================================================
# 6. DISPLAY RESULTS
# ==================================================

fig, axes = plt.subplots(2, 2, figsize=(13, 11))

# Original image
axes[0, 0].imshow(
    original,
    cmap="gray",
    vmin=-1000,
    vmax=400,
    origin="lower"
)
axes[0, 0].scatter(
    col, row,
    color="red",
    marker="+",
    s=120
)
axes[0, 0].set_title("Original CT Image")

# Low-pass image
axes[0, 1].imshow(
    low_pass_image,
    cmap="gray",
    vmin=-1000,
    vmax=400,
    origin="lower"
)
axes[0, 1].set_title("Gaussian Low-Pass Filter")

# High-pass image
high_limit = np.percentile(
    np.abs(high_pass_image), 99
)

axes[1, 0].imshow(
    high_pass_image,
    cmap="gray",
    vmin=-high_limit,
    vmax=high_limit,
    origin="lower"
)
axes[1, 0].set_title("Gaussian High-Pass Filter")

# Frequency spectrum
axes[1, 1].imshow(
    magnitude_spectrum,
    cmap="gray",
    origin="lower"
)
axes[1, 1].set_title("Fourier Magnitude Spectrum")

for ax in axes.flat:
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

fig.suptitle(
    "DIP-MONet: Frequency-Domain CT Processing",
    fontsize=16
)

plt.tight_layout()

# ==================================================
# 7. SAVE RESULTS
# ==================================================

output_dir = "outputs/frequency_domain"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(
    output_dir,
    "frequency_domain_results.png"
)

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

print("\nFrequency-domain experiment completed.")
print("Output saved:", output_path)

plt.show()
