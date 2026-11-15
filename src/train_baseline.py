
import os
import csv
import torch
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from src.dataset import LungDataset
from src.model import LungCNN
from src.sampler import BalancedBatchSampler


# =========================================
# 1. CONFIGURATION
# =========================================

EPOCHS = 20
PATIENCE = 5

BATCH_SIZE = 2
VALIDATION_BATCH_SIZE = 2

LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Device:", DEVICE)


# =========================================
# 2. LOAD DATASETS
# =========================================

# Baseline uses only original images
train_dataset = LungDataset(
    "train",
    two_views=False
)

validation_dataset = LungDataset(
    "validation",
    two_views=False
)

train_sampler = BalancedBatchSampler(
    train_dataset,
    batch_size=BATCH_SIZE
)

train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=VALIDATION_BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Training batches:", len(train_loader))
print("Validation batches:", len(validation_loader))


# =========================================
# 3. MODEL AND LOSS
# =========================================

model = LungCNN().to(DEVICE)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================================
# 4. TRAINING FUNCTION
# =========================================

def train_one_epoch():

    model.train()

    total_loss = 0.0

    for batch_index, (images, labels) in enumerate(train_loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        logits, embeddings = model(images)

        loss = criterion(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

        if (batch_index + 1) % 50 == 0:
            print(
                f"Batch [{batch_index + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    return total_loss / len(train_loader)


# =========================================
# 5. VALIDATION FUNCTION
# =========================================

def validate():

    model.eval()

    total_loss = 0.0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits, embeddings = model(images)

            loss = criterion(logits, labels)

            total_loss += loss.item()

            probabilities = torch.softmax(
                logits,
                dim=1
            )[:, 1]

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy().tolist()
            )

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        all_labels,
        all_probabilities
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1]
    )

    return {
        "loss": total_loss / len(validation_loader),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": matrix
    }


# =========================================
# 6. TRAINING HISTORY
# =========================================

history_path = os.path.join(
    OUTPUT_DIR,
    "baseline_training_history.csv"
)

history = []

best_val_auc = -1.0
epochs_without_improvement = 0


# =========================================
# 7. COMPLETE TRAINING
# =========================================

for epoch in range(EPOCHS):

    print("\n" + "=" * 50)
    print(f"BASELINE EPOCH {epoch + 1}/{EPOCHS}")
    print("=" * 50)

    train_loss = train_one_epoch()

    print("\nTraining Loss:", round(train_loss, 5))

    print("\nRunning validation...")

    val_results = validate()

    print("\nValidation Results:")
    print("Validation Loss:", round(val_results["loss"], 5))
    print("Accuracy:", round(val_results["accuracy"], 4))
    print("Precision:", round(val_results["precision"], 4))
    print("Recall:", round(val_results["recall"], 4))
    print("F1 Score:", round(val_results["f1"], 4))
    print("ROC-AUC:", round(val_results["auc"], 4))

    print("\nConfusion Matrix:")
    print(val_results["confusion_matrix"])

    # Record results
    epoch_record = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "validation_loss": val_results["loss"],
        "accuracy": val_results["accuracy"],
        "precision": val_results["precision"],
        "recall": val_results["recall"],
        "f1": val_results["f1"],
        "roc_auc": val_results["auc"]
    }

    history.append(epoch_record)

    # Save history after every epoch
    with open(history_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys()
        )

        writer.writeheader()
        writer.writerows(history)

    # Save best model
    if val_results["auc"] > best_val_auc:

        best_val_auc = val_results["auc"]
        epochs_without_improvement = 0

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            "baseline_best_model.pth"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_auc": best_val_auc
            },
            checkpoint_path
        )

        print("\nBest baseline model saved:", checkpoint_path)

    else:

        epochs_without_improvement += 1

        print(
            "No ROC-AUC improvement:",
            epochs_without_improvement,
            "/",
            PATIENCE
        )

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")
        break


# =========================================
# 8. COMPLETION
# =========================================

print("\n" + "=" * 50)
print("BASELINE TRAINING COMPLETED")
print("=" * 50)

print("Best validation ROC-AUC:", round(best_val_auc, 4))
print("History saved:", history_path)
