
import pandas as pd
import numpy as np

path = "data/processed/ct_candidates/patient_100012/dip_monet_inference_results.csv"

df = pd.read_csv(path)

gt = df["annotation_inside_patch"].astype(bool)
scores = df["class_1_probability"].to_numpy()

print("\nGROUND-TRUTH NODULE EVALUATION")
print("=" * 45)

print("Total candidate patches:", len(df))
print("Patches containing annotation:", gt.sum())

print("\nThreshold analysis")
print("-" * 65)
print(f"{'Threshold':<12}{'Detected':<12}{'GT recall':<15}{'Predicted positive'}")

for threshold in [0.1, 0.2, 0.3, 0.4, 0.45, 0.49, 0.5, 0.6, 0.7]:
    predictions = scores >= threshold

    detected = (predictions & gt).sum()
    recall = detected / gt.sum() if gt.sum() else 0

    print(
        f"{threshold:<12.2f}"
        f"{detected:<12}"
        f"{recall:<15.2%}"
        f"{predictions.sum()}"
    )

print("\nGround-truth-containing patches:")
print(
    df.loc[gt, [
        "candidate_rank",
        "class_1_probability",
        "predicted_class"
    ]].sort_values("class_1_probability", ascending=False).to_string(index=False)
)
