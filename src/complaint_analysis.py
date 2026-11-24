import pandas as pd

from src.aspects import extract_aspects


ASPECTS = [
    "Food Quality",
    "Service",
    "Delivery",
    "Price",
    "Ambience",
    "Cleanliness"
]


def analyze_aspects(df):

    aspect_results = []

    for aspect in ASPECTS:

        aspect_reviews = df[
            df["aspects"].apply(lambda items: aspect in items)
        ]

        total_mentions = len(aspect_reviews)

        negative_reviews = aspect_reviews[
            aspect_reviews["predicted_sentiment"] == "Negative"
        ]

        negative_count = len(negative_reviews)

        complaint_rate = (
            negative_count / total_mentions * 100
            if total_mentions > 0
            else 0
        )

        aspect_results.append({
            "Aspect": aspect,
            "Total Mentions": total_mentions,
            "Negative Reviews": negative_count,
            "Complaint Rate (%)": round(complaint_rate, 2)
        })

    results = pd.DataFrame(aspect_results)

    return results.sort_values(
        "Complaint Rate (%)",
        ascending=False
    ).reset_index(drop=True)