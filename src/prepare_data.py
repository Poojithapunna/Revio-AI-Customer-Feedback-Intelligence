import pandas as pd

DATA_PATH = "data/yelp_sentiment_master_dataset/yelp_sentiment_master_dataset.csv"
OUTPUT_PATH = "data/processed_reviews.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

initial_rows = len(df)

# Remove missing text or sentiment
df = df.dropna(subset=["text", "rating_review"])

# Normalize whitespace
df["text"] = df["text"].str.strip()
df["rating_review"] = df["rating_review"].str.strip()

# Remove empty review text
df = df[df["text"] != ""]

# Keep only valid sentiment labels
valid_labels = ["Positive", "Neutral", "Negative"]
df = df[df["rating_review"].isin(valid_labels)]

# Remove duplicate reviews
df = df.drop_duplicates(subset=["text"])

# Convert date to datetime
df["date"] = pd.to_datetime(
    df["date"],
    format="mixed",
    errors="coerce"
)

# Save the cleaned dataset
df.to_csv(OUTPUT_PATH, index=False)

print("=" * 50)
print("DATA PREPARATION COMPLETE")
print("=" * 50)

print("Original rows:", initial_rows)
print("Cleaned rows:", len(df))
print("Removed rows:", initial_rows - len(df))
print("Output file:", OUTPUT_PATH)

print("\nSentiment distribution after cleaning:")
print(df["rating_review"].value_counts())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("Invalid dates:", df["date"].isna().sum())
print("Date range:", df["date"].min(), "to", df["date"].max())