ASPECT_KEYWORDS = {
    "Food Quality": [
        "food", "taste", "flavor", "delicious", "fresh",
        "cold", "hot", "meal", "dish", "pizza", "burger"
    ],
    "Service": [
        "service", "staff", "waiter", "waitress", "server",
        "friendly", "rude", "slow", "attentive"
    ],
    "Price": [
        "price", "prices", "expensive", "cheap", "cost",
        "value", "overpriced", "affordable"
    ],
    "Ambience": [
        "atmosphere", "ambience", "ambiance", "music",
        "environment", "interior", "comfortable"
    ],
    "Cleanliness": [
        "clean", "dirty", "hygiene", "sanitary", "messy"
    ],
    "Delivery": [
        "delivery", "delivered", "takeout", "takeaway",
        "packaging", "order"
    ]
}


def extract_aspects(review):
    review = review.lower()
    detected_aspects = []

    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(keyword in review for keyword in keywords):
            detected_aspects.append(aspect)

    return detected_aspects