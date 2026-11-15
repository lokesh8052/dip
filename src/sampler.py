
import random
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):

    def __init__(self, dataset, batch_size=8):

        if batch_size % 2 != 0:
            raise ValueError("Batch size must be even")

        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_class = batch_size // 2

        labels = dataset.data["label"].to_numpy()

        self.class_indices = {
            label: [
                i for i, value in enumerate(labels)
                if value == label
            ]
            for label in [0, 1]
        }

        for label, indices in self.class_indices.items():
            if len(indices) < self.samples_per_class:
                raise ValueError(
                    f"Not enough samples in class {label}"
                )

    def __iter__(self):

        indices_0 = self.class_indices[0].copy()
        indices_1 = self.class_indices[1].copy()

        random.shuffle(indices_0)
        random.shuffle(indices_1)

        num_batches = min(
            len(indices_0),
            len(indices_1)
        ) // self.samples_per_class

        for batch in range(num_batches):

            start = batch * self.samples_per_class

            batch_indices = (
                indices_0[start:start + self.samples_per_class]
                + indices_1[start:start + self.samples_per_class]
            )

            random.shuffle(batch_indices)

            yield batch_indices

    def __len__(self):

        return min(
            len(self.class_indices[0]),
            len(self.class_indices[1])
        ) // self.samples_per_class
