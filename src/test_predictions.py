import joblib

# Load the previously trained model
model = joblib.load("models/sentiment_pipeline.joblib")

test_reviews = [
    "The food was amazing and the staff were incredibly friendly.",
    "Terrible experience. The food was cold and the waiter was rude.",
    "The food was okay, nothing special. Prices were average.",
    "I loved the atmosphere but the service was extremely slow.",
    "I will never come back. Worst restaurant I have ever visited."
]

print("=" * 50)
print("REVIO AI - CUSTOM SENTIMENT TEST")
print("=" * 50)

predictions = model.predict(test_reviews)
probabilities = model.predict_proba(test_reviews)

for review, prediction, probability in zip(
    test_reviews, predictions, probabilities
):
    print("\nReview:", review)
    print("Predicted Sentiment:", prediction)

    score = max(probability) * 100
    print(f"Prediction Score: {score:.2f}%")
    print("-" * 50)