
import pandas as pd

# Load dataset
df = pd.read_csv("dataset/hour.csv")

# Display first 5 records
print("\n========== FIRST 5 RECORDS ==========")
print(df.head())

# Dataset size
print("\n========== DATASET SHAPE ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Column names
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# Dataset information
print("\n========== DATASET INFORMATION ==========")
df.info()

# Missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Duplicate records
print("\n========== DUPLICATES ==========")
print("Duplicate rows:", df.duplicated().sum())

# Statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# Target column information
print("\n========== RIDE REQUEST INFORMATION ==========")
print("Average hourly demand:", df["cnt"].mean())
print("Maximum hourly demand:", df["cnt"].max())
print("Minimum hourly demand:", df["cnt"].min())

print("\nDataset exploration completed!")
