def generate_recommendations(results):

    recommendation_rules = {
        "Delivery": (
            "Investigate late deliveries, order accuracy, and packaging quality. "
            "Review delivery operations."
        ),
        "Cleanliness": (
            "Review kitchen and dining-area hygiene, cleaning schedules, "
            "and sanitation practices."
        ),
        "Service": (
            "Investigate service delays, staff responsiveness, "
            "customer handling, and staff training."
        ),
        "Food Quality": (
            "Review food freshness, preparation consistency, "
            "temperature, and quality control."
        ),
        "Price": (
            "Review pricing, portion sizes, perceived value, "
            "and customer expectations."
        ),
        "Ambience": (
            "Evaluate atmosphere, seating comfort, noise levels, "
            "lighting, and dining environment."
        )
    }

    recommendations = []

    for _, row in results.iterrows():

        aspect = row["Aspect"]
        negative_count = int(row["Negative Reviews"])
        complaint_rate = float(row["Complaint Rate (%)"])

        if negative_count == 0:
            continue

        if complaint_rate >= 25:
            priority = "High"
        elif complaint_rate >= 15:
            priority = "Medium"
        else:
            priority = "Low"

        action = recommendation_rules.get(aspect)

        if action is None:
            continue

        recommendations.append({
            "Aspect": aspect,
            "Negative Reviews": negative_count,
            "Complaint Rate (%)": complaint_rate,
            "Priority": priority,
            "Recommendation": action
        })

    priority_order = {
        "High": 0,
        "Medium": 1,
        "Low": 2
    }

    recommendations.sort(
        key=lambda item: (
            priority_order[item["Priority"]],
            -item["Complaint Rate (%)"],
            -item["Negative Reviews"]
        )
    )

    return recommendations