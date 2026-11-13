
import pandas as pd

# Load original dataset
df = pd.read_csv("dataset/hour.csv")

print("Original dataset:", df.shape)

# Convert date column
df["dteday"] = pd.to_datetime(df["dteday"])

# Sort records by date and hour
df = df.sort_values(["dteday", "hr"])

# Remove duplicate records
df = df.drop_duplicates()

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Fill missing numerical values if present
numeric_columns = df.select_dtypes(
    include=["number"]
).columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

# Fill missing categorical values if present
categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for col in categorical_columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(
            df[col].mode()[0]
        )

# Create additional date features
df["day"] = df["dteday"].dt.day
df["month"] = df["dteday"].dt.month

# Save cleaned dataset
df.to_csv("dataset/hour_cleaned.csv", index=False)

print("\nData cleaning completed!")
print("Cleaned dataset:", df.shape)
print("Saved as hour_cleaned.csv")
