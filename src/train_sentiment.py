
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH = "data/processed_reviews.csv"

# Load cleaned data
df = pd.read_csv(DATA_PATH)

X = df["text"]
y = df["rating_review"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Build the ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words="english"
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

# Train
print("Training sentiment model...")
model.fit(X_train, y_train)

# Save the trained model
os.makedirs("models", exist_ok=True)

model_path = "models/sentiment_pipeline.joblib"
joblib.dump(model, model_path)

print(f"\nModel saved successfully at: {model_path}")
# Predict on unseen test data
y_pred = model.predict(X_test)

# Evaluation
print("\n" + "=" * 60)
print("SENTIMENT MODEL EVALUATION")
print("=" * 60)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    y_pred,
    labels=["Negative", "Neutral", "Positive"]
))

print("\nTest accuracy:", round(model.score(X_test, y_test), 4))
