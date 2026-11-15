
import numpy as np
from scipy import ndimage


def create_lung_mask(
    volume,
    hu_threshold=-400,
    min_component_size=500
):
    volume = np.asarray(volume)

    if volume.ndim != 3:
        raise ValueError(
            "Expected a 3D CT volume in ZYX order."
        )

    lung_mask = np.zeros(volume.shape, dtype=bool)

    connectivity = np.ones((3, 3), dtype=bool)

    for z in range(volume.shape[0]):
        binary = volume[z] < hu_threshold

        labels, count = ndimage.label(
            binary,
            structure=connectivity
        )

        if count == 0:
            continue

        sizes = np.bincount(labels.ravel())
        sizes[0] = 0

        border_labels = np.unique(np.concatenate([
            labels[0, :],
            labels[-1, :],
            labels[:, 0],
            labels[:, -1]
        ]))

        sizes[border_labels] = 0

        largest = np.argsort(sizes)[::-1][:2]

        selected = np.isin(labels, largest)
        selected &= labels != 0

        selected_labels, selected_count = ndimage.label(
            selected,
            structure=connectivity
        )

        if selected_count == 0:
            continue

        selected_sizes = np.bincount(
            selected_labels.ravel()
        )

        valid = np.where(
            selected_sizes >= min_component_size
        )[0]

        valid = valid[valid != 0]

        selected = np.isin(selected_labels, valid)

        selected = ndimage.binary_fill_holes(selected)

        lung_mask[z] = selected

    if not lung_mask.any():
        raise ValueError(
            "No lung regions detected. "
            "Check CT orientation and threshold."
        )

    return lung_mask
