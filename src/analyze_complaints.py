import pandas as pd
import joblib

from aspects import extract_aspects

DATA_PATH = "data/processed_reviews.csv"
MODEL_PATH = "models/sentiment_pipeline.joblib"

print("Loading reviews...")
df = pd.read_csv(DATA_PATH)

print("Loading trained model...")
model = joblib.load(MODEL_PATH)

print("Predicting sentiments...")
df["predicted_sentiment"] = model.predict(df["text"])

print("Extracting aspects...")
df["aspects"] = df["text"].apply(extract_aspects)

# Count reviews mentioning each aspect
aspect_results = []

all_aspects = [
    "Food Quality",
    "Service",
    "Price",
    "Ambience",
    "Cleanliness",
    "Delivery"
]

for aspect in all_aspects:

    # Reviews mentioning this aspect
    aspect_reviews = df[
        df["aspects"].apply(lambda aspects: aspect in aspects)
    ]

    total_mentions = len(aspect_reviews)

    # Negative reviews mentioning this aspect
    negative_reviews = aspect_reviews[
        aspect_reviews["predicted_sentiment"] == "Negative"
    ]

    negative_count = len(negative_reviews)

    complaint_rate = (
        (negative_count / total_mentions) * 100
        if total_mentions > 0 else 0
    )

    aspect_results.append({
        "Aspect": aspect,
        "Total Mentions": total_mentions,
        "Negative Reviews": negative_count,
        "Complaint Rate (%)": round(complaint_rate, 2)
    })

results = pd.DataFrame(aspect_results)

results = results.sort_values(
    by="Negative Reviews",
    ascending=False
)

print("\n" + "=" * 60)
print("REVIO AI - RECURRING COMPLAINT ANALYSIS")
print("=" * 60)

print(results.to_string(index=False))

print("\nAnalysis completed.")