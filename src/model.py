
import torch
import torch.nn as nn
import torch.nn.functional as F


class LungCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Existing 3D CNN backbone
        self.features = nn.Sequential(

            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

        # Feature representation
        self.flatten = nn.Flatten()

        self.feature_layer = nn.Sequential(
            nn.Linear(32 * 8 * 16 * 16, 64),
            nn.ReLU()
        )

        # Classification head
        self.classifier = nn.Linear(64, 2)

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.flatten(x)

        features = self.feature_layer(x)

        logits = self.classifier(features)

        embeddings = self.projection(features)

        embeddings = F.normalize(embeddings, dim=1)

        return logits, embeddings
