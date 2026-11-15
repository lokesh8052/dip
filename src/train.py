
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
from src.losses import FocalLoss, SupConLoss, ConsistencyLoss
from src.sampler import BalancedBatchSampler


# =========================================
# 1. CONFIGURATION
# =========================================

EPOCHS = 20

BATCH_SIZE = 2
VALIDATION_BATCH_SIZE = 2

LEARNING_RATE = 0.001

LAMBDA_SUPCON = 0.1
LAMBDA_CONSISTENCY = 1.0

PATIENCE = 5
epochs_without_improvement = 0

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

train_dataset = LungDataset(
    "train",
    two_views=True
)

validation_dataset = LungDataset(
    "validation",
    two_views=False
)

# Balanced training batches
train_sampler = BalancedBatchSampler(
    train_dataset,
    batch_size=BATCH_SIZE
)

train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    num_workers=0
)

# Normal validation batches
validation_loader = DataLoader(
    validation_dataset,
    batch_size=VALIDATION_BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Training batches:", len(train_loader))
print("Validation batches:", len(validation_loader))


# =========================================
# 3. INITIALIZE MODEL
# =========================================

model = LungCNN().to(DEVICE)


# =========================================
# 4. LOSS FUNCTIONS
# =========================================

criterion_focal = FocalLoss(
    gamma=2.0
)

criterion_supcon = SupConLoss(
    temperature=0.07
)

criterion_consistency = ConsistencyLoss()


# =========================================
# 5. OPTIMIZER
# =========================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================================
# 6. TRAINING FUNCTION
# =========================================

def train_one_epoch():

    model.train()

    running_total = 0.0
    running_focal = 0.0
    running_supcon = 0.0
    running_consistency = 0.0

    for batch_index, (view1, view2, labels) in enumerate(train_loader):

        view1 = view1.to(DEVICE)
        view2 = view2.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        # Combine both views
        images = torch.cat(
            [view1, view2],
            dim=0
        )

        # Forward pass
        logits, embeddings = model(images)

        batch_size = labels.size(0)

        logits_original = logits[:batch_size]
        logits_augmented = logits[batch_size:]

        embeddings_original = embeddings[:batch_size]
        embeddings_augmented = embeddings[batch_size:]

        # Focal loss
        loss_focal = (
            criterion_focal(logits_original, labels)
            + criterion_focal(logits_augmented, labels)
        ) / 2

        # Supervised contrastive loss
        combined_embeddings = torch.cat(
            [embeddings_original, embeddings_augmented],
            dim=0
        )

        combined_labels = torch.cat(
            [labels, labels],
            dim=0
        )

        loss_supcon = criterion_supcon(
            combined_embeddings,
            combined_labels
        )

        # Consistency loss
        loss_consistency = criterion_consistency(
            logits_original,
            logits_augmented
        )

        # Combined objective
        loss = (
            loss_focal
            + LAMBDA_SUPCON * loss_supcon
            + LAMBDA_CONSISTENCY * loss_consistency
        )

        # Backpropagation
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_total += loss.item()
        running_focal += loss_focal.item()
        running_supcon += loss_supcon.item()
        running_consistency += loss_consistency.item()

        if (batch_index + 1) % 50 == 0:
            print(
                f"Batch [{batch_index + 1}/{len(train_loader)}] "
                f"Total: {loss.item():.4f}"
            )

    number_of_batches = len(train_loader)

    return {
        "total": running_total / number_of_batches,
        "focal": running_focal / number_of_batches,
        "supcon": running_supcon / number_of_batches,
        "consistency": running_consistency / number_of_batches
    }


# =========================================
# 7. VALIDATION FUNCTION
# =========================================

def validate():

    model.eval()

    total_validation_loss = 0.0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits, embeddings = model(images)

            # Validation uses original images only
            loss = criterion_focal(logits, labels)

            total_validation_loss += loss.item()

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

    # Convert to arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    # Calculate metrics
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

    average_validation_loss = (
        total_validation_loss / len(validation_loader)
    )

    return {
        "loss": average_validation_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": matrix
    }


# =========================================
# 8. SAVE TRAINING HISTORY
# =========================================

history_path = os.path.join(
    OUTPUT_DIR,
    "training_history.csv"
)

history = []

best_val_auc = -1.0


# =========================================
# 9. COMPLETE TRAINING
# =========================================

for epoch in range(EPOCHS):

    print("\n" + "=" * 50)
    print(f"EPOCH {epoch + 1}/{EPOCHS}")
    print("=" * 50)

    # Training
    train_results = train_one_epoch()

    print("\nTraining Results:")
    print("Total Loss:", round(train_results["total"], 5))
    print("Focal Loss:", round(train_results["focal"], 5))
    print("SupCon Loss:", round(train_results["supcon"], 5))
    print("Consistency Loss:", round(train_results["consistency"], 7))

    # Validation
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

    # Record epoch results
    epoch_record = {
        "epoch": epoch + 1,
        "train_loss": train_results["total"],
        "focal_loss": train_results["focal"],
        "supcon_loss": train_results["supcon"],
        "consistency_loss": train_results["consistency"],
        "validation_loss": val_results["loss"],
        "accuracy": val_results["accuracy"],
        "precision": val_results["precision"],
        "recall": val_results["recall"],
        "f1": val_results["f1"],
        "roc_auc": val_results["auc"]
    }

    history.append(epoch_record)


    # Save best model and check early stopping
    if val_results["auc"] > best_val_auc:

        best_val_auc = val_results["auc"]
        epochs_without_improvement = 0

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            "best_model.pth"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_auc": best_val_auc,
                "validation_metrics": {
                    key: value
                    for key, value in val_results.items()
                    if key != "confusion_matrix"
                }
            },
            checkpoint_path
        )

        print("\nBest model saved:", checkpoint_path)

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


# Write CSV
with open(history_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=history[0].keys()
    )

    writer.writeheader()
    writer.writerows(history)


print("\n" + "=" * 50)
print("TRAINING COMPLETED")
print("=" * 50)

print("Best validation ROC-AUC:", round(best_val_auc, 4))
print("History saved:", history_path)
print("Checkpoint directory:", CHECKPOINT_DIR)
