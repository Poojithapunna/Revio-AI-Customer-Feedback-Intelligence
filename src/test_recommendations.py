import pandas as pd

from recommendations import generate_recommendations

results = pd.DataFrame([
    {
        "Aspect": "Delivery",
        "Total Mentions": 4050,
        "Negative Reviews": 1251,
        "Complaint Rate (%)": 30.89
    },
    {
        "Aspect": "Cleanliness",
        "Total Mentions": 912,
        "Negative Reviews": 219,
        "Complaint Rate (%)": 24.01
    },
    {
        "Aspect": "Service",
        "Total Mentions": 7666,
        "Negative Reviews": 1630,
        "Complaint Rate (%)": 21.26
    },
    {
        "Aspect": "Food Quality",
        "Total Mentions": 11408,
        "Negative Reviews": 2216,
        "Complaint Rate (%)": 19.42
    },
    {
        "Aspect": "Price",
        "Total Mentions": 2986,
        "Negative Reviews": 571,
        "Complaint Rate (%)": 19.12
    },
    {
        "Aspect": "Ambience",
        "Total Mentions": 2241,
        "Negative Reviews": 268,
        "Complaint Rate (%)": 11.96
    }
])

recommendations = generate_recommendations(results)

print("\n" + "=" * 75)
print("REVIO AI - BUSINESS RECOMMENDATIONS")
print("=" * 75)

for index, item in enumerate(recommendations, start=1):

    print(f"\n{index}. {item['Aspect']} | Priority: {item['Priority']}")
    print(f"   Negative Reviews: {item['Negative Reviews']}")
    print(f"   Complaint Rate: {item['Complaint Rate (%)']}%")
    print(f"   Action: {item['Recommendation']}")

print("\nAnalysis completed.")