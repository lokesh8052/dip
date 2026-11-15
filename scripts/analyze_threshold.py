
import torch
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.dataset import LungDataset
from src.model import LungCNN


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load validation dataset
dataset = LungDataset(
    "validation",
    two_views=False
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

# Load model
model = LungCNN().to(device)

checkpoint = torch.load(
    "checkpoints/baseline_best_model.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

# Collect probabilities and labels
all_labels = []
all_probabilities = []

with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        logits, embeddings = model(images)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[:, 1]

        all_labels.extend(labels.numpy().tolist())

        all_probabilities.extend(
            probabilities.cpu().numpy().tolist()
        )


all_labels = np.array(all_labels)
all_probabilities = np.array(all_probabilities)


# Evaluate different thresholds
thresholds = [
    0.1, 0.2, 0.3, 0.4,
    0.5, 0.6, 0.7, 0.8, 0.9
]

print("\nTHRESHOLD ANALYSIS")
print("-" * 85)

print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)

best_f1 = -1
best_threshold = None

for threshold in thresholds:

    predictions = (
        all_probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        all_labels,
        predictions
    )

    precision = precision_score(
        all_labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        predictions,
        zero_division=0
    )

    print(
        f"{threshold:<12.2f}"
        f"{accuracy:<12.4f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("\nBest threshold by validation F1:", best_threshold)
print("Best validation F1:", round(best_f1, 4))

# Confusion matrix at best threshold
best_predictions = (
    all_probabilities >= best_threshold
).astype(int)

print("\nConfusion Matrix at best threshold:")
print(
    confusion_matrix(
        all_labels,
        best_predictions,
        labels=[0, 1]
    )
)
