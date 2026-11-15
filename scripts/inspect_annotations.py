
import pandas as pd

path = "data/raw/annotations/LUNA25_Public_Training_Development_Data.csv"

df = pd.read_csv(path)

print("\n========== DATASET OVERVIEW ==========")

print("Total rows:", len(df))
print("Total columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 10 rows:")
print(df.head(10).to_string(index=False))

print("\nMissing values:")
print(df.isnull().sum())

print("\nUnique labels:")
print(df["label"].value_counts(dropna=False))

print("\nUnique patients:", df["PatientID"].nunique())

print("\nUnique CT series:", df["SeriesInstanceUID"].nunique())

print("\nUnique nodules:", df["NoduleID"].nunique())

print("\nGender distribution:")
print(df["Gender"].value_counts(dropna=False))

print("\nAge statistics:")
print(df["Age_at_StudyDate"].describe())
