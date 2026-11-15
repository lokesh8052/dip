
import torch
import torch.nn.functional as F


def gaussian_filter_3d(image):
    """
    Apply 3D Gaussian smoothing to a CT image.
    """

    # One-dimensional Gaussian weights
    weights = torch.tensor(
        [1, 2, 1],
        dtype=image.dtype,
        device=image.device
    )

    # Construct a 3D Gaussian kernel
    kernel = (
        weights[:, None, None]
        * weights[None, :, None]
        * weights[None, None, :]
    )

    # Normalize kernel
    kernel = kernel / kernel.sum()

    # Shape required by Conv3D
    kernel = kernel.view(1, 1, 3, 3, 3)

    # Apply Gaussian smoothing
    filtered_image = F.conv3d(
        image,
        kernel,
        padding=1
    )

    return filtered_image
