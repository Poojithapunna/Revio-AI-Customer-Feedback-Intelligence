# Revio AI — Customer Feedback Intelligence & Action Engine

Revio AI is an AI-powered customer feedback analysis application that transforms customer reviews into actionable business insights. It helps businesses understand customer sentiment, identify recurring complaints, analyze specific service aspects, and prioritize areas for improvement.

## Overview

Customer reviews contain valuable insights, but manually analyzing large volumes of feedback can be time-consuming. Revio AI automates this process using machine learning and text analysis to uncover customer satisfaction patterns and potential operational issues.

The application supports review analysis through an interactive Streamlit dashboard.

## Features

- **Sentiment Analysis:** Classifies customer reviews into Positive, Neutral, and Negative categories using a trained machine learning model.
- **Aspect Detection:** Identifies key aspects mentioned in reviews, including Food Quality, Service, Delivery, Price, Ambience, and Cleanliness.
- **Complaint Intelligence:** Analyzes negative feedback by aspect and calculates complaint rates and complaint volumes.
- **Business Recommendations:** Generates actionable suggestions based on complaint severity and customer feedback patterns.
- **Interactive Dashboard:** Visualizes sentiment distribution, complaint metrics, and customer feedback insights.
- **Review Explorer:** Allows users to filter, search, and explore individual analyzed reviews.
- **Trend Analysis:** Supports sentiment trend exploration when date information is available in the uploaded dataset.
- **CSV Upload & Export:** Enables analysis of uploaded customer review datasets and downloading of analyzed results.

## Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Logistic Regression
- Plotly
- Joblib

## Project Structure

```text
Revio-AI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   └── sentiment_pipeline.joblib
└── src/
    ├── aspects.py
    ├── complaint_analysis.py
    ├── recommendations.py
    ├── analyze_complaints.py
    ├── inspect_data.py
    ├── prepare_data.py
    ├── train_sentiment.py
    ├── test_aspects.py
    ├── test_predictions.py
    └── test_recommendations.py
```

## Machine Learning Approach

The sentiment classification pipeline uses:

1. **Text preprocessing and feature extraction** using TF-IDF.
2. **Logistic Regression** for sentiment classification.
3. **Three sentiment classes:** Positive, Neutral, and Negative.
4. **Aspect-based analysis** to identify topics discussed in customer reviews.
5. **Complaint analysis and rule-based recommendation logic** to prioritize business improvements.

The sentiment model was trained using Yelp review data with ratings mapped to sentiment labels.

## Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 80.63% |
| Macro F1-score | 68.98% |
| Weighted F1-score | 81.17% |

*Performance metrics are based on the model's evaluation results on the available dataset. Performance may vary across different businesses, domains, and customer review datasets.*

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Poojithapunna/Revio-AI-Customer-Feedback-Intelligence.git
```

### 2. Navigate to the project directory

```bash
cd Revio-AI-Customer-Feedback-Intelligence
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## Usage

1. Launch the Revio AI Streamlit application.
2. Explore the existing dashboard or upload a customer review CSV file.
3. Select or confirm the review text column.
4. Run sentiment analysis on the uploaded reviews.
5. Explore sentiment distribution, complaint analysis, and aspect-level insights.
6. Review the generated business recommendations.
7. Filter individual reviews or explore trends when date columns are available.
8. Export the analyzed reviews as a CSV file.

## Limitations

- Sentiment predictions are model-generated and may not perfectly reflect the intended meaning of a review.
- Neutral sentiment classification is more challenging than Positive and Negative classification.
- Aspect detection is keyword-based and may not capture every context or variation.
- Business recommendations are rule-based and depend on the detected aspects and complaint metrics.
- Model performance may vary when applied to datasets outside its original training domain.

## Future Improvements

- Context-aware aspect and complaint extraction.
- Improved neutral sentiment classification.
- Multilingual customer feedback support.
- Advanced trend and anomaly detection.
- Domain adaptation for different business categories.
- More sophisticated recommendation and prioritization methods.

## Author

**Poojitha Punna**

GitHub: [Poojithapunna](https://github.com/Poojithapunna)

---

*Revio AI — Turning customer feedback into actionable intelligence.*
