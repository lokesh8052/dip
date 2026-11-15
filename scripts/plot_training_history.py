
import pandas as pd
import matplotlib.pyplot as plt

# Load training history
history = pd.read_csv("outputs/training_history.csv")

print(history)

# Plot 1: Training and validation loss
plt.figure(figsize=(8, 5))

plt.plot(
    history["epoch"],
    history["train_loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    history["epoch"],
    history["validation_loss"],
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    "outputs/loss_curve.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# Plot 2: Validation ROC-AUC
plt.figure(figsize=(8, 5))

plt.plot(
    history["epoch"],
    history["roc_auc"],
    marker="o",
    label="Validation ROC-AUC"
)

plt.xlabel("Epoch")
plt.ylabel("ROC-AUC")
plt.title("Validation ROC-AUC Across Epochs")
plt.legend()
plt.grid(True)

plt.savefig(
    "outputs/roc_auc_curve.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("Plots saved in outputs/")
