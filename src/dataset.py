
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocessing import gaussian_filter_3d


class LungDataset(Dataset):

    def __init__(self, split, two_views=True):
        self.two_views = two_views

        self.data = pd.read_csv(
            "data/processed/splits/dataset_manifest.csv"
        )

        self.data = self.data[
            self.data["split"] == split
        ].reset_index(drop=True)

        print(f"{split} samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]
        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long
        )
        # Load original CT patch
        image = np.load(
            row["image_path"]
        ).astype(np.float32)

        # Normalize intensity
        image = np.clip(image, -1000, 400)
        image = (image + 1000) / 1400

        # Add channel dimension
        image = np.expand_dims(image, axis=0)

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        # Validation: return only the original image
        if not self.two_views:
            return image, label

        # Training: create two views
        view1 = image.clone()

        view2 = gaussian_filter_3d(
            image.unsqueeze(0)
        ).squeeze(0)

        scale = torch.empty(1).uniform_(0.95, 1.05)
        shift = torch.empty(1).uniform_(-0.02, 0.02)

        view2 = torch.clamp(
            view2 * scale + shift,
            0.0,
            1.0
        )

        return view1, view2, label