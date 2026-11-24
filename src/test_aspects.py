from aspects import extract_aspects

reviews = [
    "The food was amazing but the service was slow.",
    "Very expensive, but the atmosphere was beautiful.",
    "The delivery was late and the packaging was messy.",
    "Everything was excellent.",
    "The restaurant was dirty and the food was cold."
]

for review in reviews:
    print("\nReview:", review)
    print("Detected Aspects:", extract_aspects(review))