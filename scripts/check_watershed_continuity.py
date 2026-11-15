
import os
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load full-volume watershed labels
labels = np.load(
    "data/processed/ct_volumes/patient_100012_watershed_full.npz"
)["labels"]

print("Loaded watershed volume:", labels.shape)
print("Data type:", labels.dtype)

# Step 2: Dice similarity function
def dice_score(mask_a, mask_b):

    intersection = np.logical_and(mask_a, mask_b).sum()
    total = mask_a.sum() + mask_b.sum()

    if total == 0:
        return 1.0

    return (2.0 * intersection) / total


# Step 3: Compare consecutive slices
slice_indices = []
left_dice = []
right_dice = []

for z in range(labels.shape[0] - 1):

    left_current = labels[z] == 1
    left_next = labels[z + 1] == 1

    right_current = labels[z] == 2
    right_next = labels[z + 1] == 2

    left_score = dice_score(
        left_current,
        left_next
    )

    right_score = dice_score(
        right_current,
        right_next
    )

    slice_indices.append(z)
    left_dice.append(left_score)
    right_dice.append(right_score)


# Step 4: Print summary statistics
print("\n3D CONTINUITY ANALYSIS")

print("Mean left Dice:", np.mean(left_dice))
print("Mean right Dice:", np.mean(right_dice))

print("Minimum left Dice:", np.min(left_dice))
print("Minimum right Dice:", np.min(right_dice))

print("Median left Dice:", np.median(left_dice))
print("Median right Dice:", np.median(right_dice))


# Step 5: Find lowest Dice transitions
left_worst = np.argsort(left_dice)[:5]
right_worst = np.argsort(right_dice)[:5]

print("\nFive lowest left-side Dice scores:")

for i in left_worst:
    print(
        f"Slice {i} -> {i+1}: {left_dice[i]:.4f}"
    )

print("\nFive lowest right-side Dice scores:")

for i in right_worst:
    print(
        f"Slice {i} -> {i+1}: {right_dice[i]:.4f}"
    )


# Step 6: Plot continuity graph
plt.figure(figsize=(14, 5))

plt.plot(
    slice_indices,
    left_dice,
    label="Label 1"
)

plt.plot(
    slice_indices,
    right_dice,
    label="Label 2"
)

plt.xlabel("Axial slice transition")
plt.ylabel("Dice similarity")

plt.title(
    "Watershed 3D Slice-to-Slice Continuity"
)

plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()

plt.tight_layout()

os.makedirs("outputs", exist_ok=True)

plt.savefig(
    "outputs/watershed_continuity.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("\nSaved: outputs/watershed_continuity.png")
print("Continuity analysis completed.")
