import pandas as pd

DATA_PATH = "data/yelp_sentiment_master_dataset/yelp_sentiment_master_dataset.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("-" * 50)

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\n" + "=" * 50)
print("SENTIMENT DISTRIBUTION")
print("=" * 50)

print(df["rating_review"].value_counts())

print("\nSentiment percentages:")
print(
    df["rating_review"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\n" + "=" * 50)
print("DUPLICATE CHECK")
print("=" * 50)

print("Duplicate rows:", df.duplicated().sum())
print("Duplicate review texts:", df["text"].duplicated().sum())

df["review_length"] = df["text"].str.len()

print("\n" + "=" * 50)
print("REVIEW LENGTH ANALYSIS")
print("=" * 50)

print(df["review_length"].describe())

print("\n" + "=" * 50)
print("SAMPLE REVIEWS")
print("=" * 50)

for sentiment in ["Positive", "Neutral", "Negative"]:
    print(f"\n--- {sentiment} ---")

    samples = df[df["rating_review"] == sentiment].sample(
        min(3, len(df[df["rating_review"] == sentiment])) ,
        random_state=42
    )

    for review in samples["text"]:
        print(f"\n{review[:500]}")
        print("-" * 30)