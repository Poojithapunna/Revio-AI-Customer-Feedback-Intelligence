import re
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

from src.aspects import extract_aspects
from src.complaint_analysis import analyze_aspects
from src.recommendations import generate_recommendations


# ==========================================================
# CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Revio AI | Review Intelligence",
    page_icon="📊",
    layout="wide"
)

DATA_PATH = "data/processed_reviews.csv"
MODEL_PATH = "models/sentiment_pipeline.joblib"


# ==========================================================
# LOAD DATA & MODEL
# ==========================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_data()
model = load_model()


# ==========================================================
# SENTIMENT PREDICTION
# ==========================================================

@st.cache_data
def predict_sentiments(texts):
    return model.predict(texts)


df["predicted_sentiment"] = predict_sentiments(
    df["text"].fillna("").astype(str).tolist()
)

df["aspects"] = df["text"].fillna("").astype(str).apply(
    extract_aspects
)


# ==========================================================
# HEADER
# ==========================================================

st.title("Revio AI")
st.subheader("Customer Feedback Intelligence & Action Engine")

st.markdown(
    "Analyze customer sentiment, identify recurring problems, "
    "and understand customer experiences through review data."
)

st.divider()


# ==========================================================
# UPLOAD WIDGET
# ==========================================================

st.sidebar.divider()
st.sidebar.subheader("Upload Customer Reviews")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file",
    type=["csv"],
    help="Upload a CSV containing customer reviews."
)


# ==========================================================
# HELPER: NORMALIZE COLUMN NAMES
# ==========================================================

def normalize_column_name(column):
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(column).strip().lower()
    ).strip("_")


# ==========================================================
# HELPER: SENTIMENT OVERVIEW FOR UPLOADS
# ==========================================================

def render_uploaded_sentiment(valid_reviews):

    st.divider()
    st.header("📊 Sentiment Overview")

    total_reviews = len(valid_reviews)

    positive_count = (
        valid_reviews["Predicted Sentiment"] == "Positive"
    ).sum()

    neutral_count = (
        valid_reviews["Predicted Sentiment"] == "Neutral"
    ).sum()

    negative_count = (
        valid_reviews["Predicted Sentiment"] == "Negative"
    ).sum()

    positive_pct = positive_count / total_reviews * 100
    neutral_pct = neutral_count / total_reviews * 100
    negative_pct = negative_count / total_reviews * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Analyzed Reviews",
        f"{total_reviews:,}"
    )

    col2.metric(
        "Positive Reviews",
        f"{positive_count:,}",
        f"{positive_pct:.1f}%"
    )

    col3.metric(
        "Neutral Reviews",
        f"{neutral_count:,}",
        f"{neutral_pct:.1f}%"
    )

    col4.metric(
        "Negative Reviews",
        f"{negative_count:,}",
        f"{negative_pct:.1f}%"
    )

    # Sentiment distribution
    sentiment_counts = (
        valid_reviews["Predicted Sentiment"]
        .value_counts()
        .reindex(
            ["Positive", "Neutral", "Negative"],
            fill_value=0
        )
        .reset_index()
    )

    sentiment_counts.columns = [
        "Sentiment",
        "Count"
    ]

    st.subheader("Sentiment Distribution")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        fig_pie = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Count",
            hole=0.45,
            title="Sentiment Breakdown"
        )

        st.plotly_chart(
            fig_pie,
            width="stretch"
        )

    with chart_col2:

        fig_bar = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            text="Count",
            title="Reviews per Sentiment"
        )

        fig_bar.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig_bar,
            width="stretch"
        )


# ==========================================================
# HELPER: UPLOADED COMPLAINT INTELLIGENCE
# ==========================================================

def render_uploaded_complaints(valid_reviews):

    st.divider()
    st.header("🔍 Complaint Intelligence")

    try:

        aspect_results = analyze_aspects(valid_reviews)

        # Only aspects actually mentioned in the uploaded reviews
        mentioned_aspects = aspect_results[
            aspect_results["Total Mentions"] > 0
        ].copy()

        if mentioned_aspects.empty:

            st.info(
                "No supported aspects were detected in the uploaded reviews. "
                "Complaint intelligence cannot be generated for this dataset."
            )

            return aspect_results

        # Determine complaint rate and volume using only
        # aspects with actual mentions
        highest_rate = mentioned_aspects.loc[
            mentioned_aspects["Complaint Rate (%)"].idxmax()
        ]

        highest_volume = mentioned_aspects.loc[
            mentioned_aspects["Negative Reviews"].idxmax()
        ]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Highest Complaint Rate",
            highest_rate["Aspect"],
            f"{highest_rate['Complaint Rate (%)']:.2f}%"
        )

        col2.metric(
            "Highest Complaint Volume",
            highest_volume["Aspect"],
            f"{int(highest_volume['Negative Reviews'])} negative reviews"
        )

        col3.metric(
            "Aspects Detected",
            f"{len(mentioned_aspects)} / 6"
        )

        # Complaint rate chart
        st.subheader("Complaint Rate by Aspect")

        complaint_rate_chart_data = mentioned_aspects.sort_values(
            "Complaint Rate (%)",
            ascending=True
        )

        fig_rate = px.bar(
            complaint_rate_chart_data,
            x="Complaint Rate (%)",
            y="Aspect",
            orientation="h",
            text="Complaint Rate (%)",
            title="Percentage of Aspect Reviews Predicted Negative"
        )

        fig_rate.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig_rate,
            width="stretch"
        )

        # Negative review volume
        st.subheader("Negative Complaint Volume")

        complaint_volume_data = mentioned_aspects.sort_values(
            "Negative Reviews",
            ascending=True
        )

        fig_volume = px.bar(
            complaint_volume_data,
            x="Negative Reviews",
            y="Aspect",
            orientation="h",
            text="Negative Reviews",
            title="Negative Reviews per Aspect"
        )

        fig_volume.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig_volume,
            width="stretch"
        )

        st.subheader("Complaint Analysis Details")

        st.dataframe(
            mentioned_aspects,
            width="stretch",
            hide_index=True
        )

        return aspect_results

    except Exception as e:

        st.error(
            f"Complaint intelligence could not be generated: {e}"
        )

        return pd.DataFrame()


# ==========================================================
# HELPER: UPLOADED RECOMMENDATIONS
# ==========================================================

def render_uploaded_recommendations(aspect_results):

    st.divider()
    st.header("💡 Business Recommendations")

    st.markdown(
        "Actionable suggestions generated from negative customer feedback."
    )

    try:

        if aspect_results is None or aspect_results.empty:

            st.info(
                "Recommendations are unavailable because aspect analysis "
                "could not be completed."
            )

            return

        recommendations = generate_recommendations(
            aspect_results
        )

        if not recommendations:

            st.success(
                "No negative aspect feedback requiring recommendations "
                "was identified."
            )

            return

        for item in recommendations:

            with st.container(border=True):

                st.subheader(
                    f"{item['Aspect']} — {item['Priority']} Priority"
                )

                col1, col2 = st.columns(2)

                col1.metric(
                    "Negative Reviews",
                    f"{item['Negative Reviews']:,}"
                )

                col2.metric(
                    "Complaint Rate",
                    f"{item['Complaint Rate (%)']:.2f}%"
                )

                st.markdown("**Recommended Action**")

                st.write(
                    item["Recommendation"]
                )

    except Exception as e:

        st.error(
            f"Recommendations could not be generated: {e}"
        )


# ==========================================================
# HELPER: UPLOADED SENTIMENT TRENDS
# ==========================================================

def render_uploaded_trends(valid_reviews, possible_date_columns):

    st.divider()
    st.header("📈 Sentiment Trends Over Time")

    if not possible_date_columns:

        st.info(
            "No date column was detected. Upload reviews with a date column "
            "to enable time-based sentiment analysis."
        )

        return

    selected_date_column = st.selectbox(
        "Select date column for trend analysis",
        possible_date_columns,
        key="uploaded_date_column"
    )

    trend_df = valid_reviews.copy()

    trend_df["parsed_date"] = pd.to_datetime(
        trend_df[selected_date_column],
        errors="coerce"
    )

    invalid_dates = trend_df["parsed_date"].isna().sum()

    if invalid_dates > 0:

        st.warning(
            f"{invalid_dates:,} reviews have missing or invalid dates "
            "and are excluded from trend analysis."
        )

    trend_df = trend_df.dropna(
        subset=["parsed_date"]
    )

    if trend_df.empty:

        st.warning(
            "No valid dates found. Unable to generate sentiment trends."
        )

        return

    trend_df["Month"] = (
        trend_df["parsed_date"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly_sentiment = (
        trend_df.groupby(
            ["Month", "Predicted Sentiment"]
        )
        .size()
        .reset_index(name="Review Count")
    )

    fig_trend = px.line(
        monthly_sentiment,
        x="Month",
        y="Review Count",
        color="Predicted Sentiment",
        markers=True,
        title="Monthly Sentiment Trends"
    )

    st.plotly_chart(
        fig_trend,
        width="stretch"
    )


# ==========================================================
# HELPER: UPLOADED REVIEW EXPLORER
# ==========================================================

def render_uploaded_explorer(valid_reviews, selected_review_column):

    st.divider()
    st.header("📝 Uploaded Review Explorer")

    st.caption(
        "Explore individual reviews, predicted sentiment, "
        "and detected aspects."
    )

    # Sentiment filter
    sentiment_filter = st.selectbox(
        "Filter by sentiment",
        ["All", "Positive", "Neutral", "Negative"],
        key="uploaded_sentiment_filter"
    )

    # Aspect filter
    all_detected_aspects = sorted(
        {
            aspect
            for aspects in valid_reviews["aspects"]
            for aspect in aspects
        }
    )

    aspect_filter = st.selectbox(
        "Filter by aspect",
        ["All"] + all_detected_aspects,
        key="uploaded_aspect_filter"
    )

    # Text search
    search_query = st.text_input(
        "Search uploaded reviews",
        key="uploaded_review_search"
    )

    uploaded_display_data = valid_reviews.copy()

    if sentiment_filter != "All":

        uploaded_display_data = uploaded_display_data[
            uploaded_display_data["Predicted Sentiment"] == sentiment_filter
        ]

    if aspect_filter != "All":

        uploaded_display_data = uploaded_display_data[
            uploaded_display_data["aspects"].apply(
                lambda aspects: aspect_filter in aspects
            )
        ]

    if search_query.strip():

        uploaded_display_data = uploaded_display_data[
            uploaded_display_data[selected_review_column]
            .astype(str)
            .str.contains(
                search_query,
                case=False,
                na=False,
                regex=False
            )
        ]

    uploaded_display = uploaded_display_data[
        [
            selected_review_column,
            "Predicted Sentiment",
            "aspects"
        ]
    ].copy()

    uploaded_display = uploaded_display.rename(
        columns={
            selected_review_column: "Customer Review",
            "Predicted Sentiment": "Predicted Sentiment",
            "aspects": "Detected Aspects"
        }
    )

    st.caption(
        f"Showing {len(uploaded_display_data):,} of "
        f"{len(valid_reviews):,} uploaded reviews"
    )

    st.dataframe(
        uploaded_display,
        width="stretch",
        hide_index=True
    )


# ==========================================================
# HELPER: UPLOADED CSV EXPORT
# ==========================================================

def render_uploaded_export(valid_reviews, selected_review_column):

    st.divider()
    st.header("📥 Export Analysis")

    export_reviews = valid_reviews[
        [
            selected_review_column,
            "Predicted Sentiment",
            "aspects"
        ]
    ].copy()

    export_reviews = export_reviews.rename(
        columns={
            selected_review_column: "Customer Review",
            "Predicted Sentiment": "Sentiment",
            "aspects": "Detected Aspects"
        }
    )

    export_reviews["Detected Aspects"] = (
        export_reviews["Detected Aspects"].apply(
            lambda aspects: ", ".join(aspects)
        )
    )

    csv_data = export_reviews.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="⬇️ Download Analyzed Reviews (CSV)",
        data=csv_data,
        file_name="revio_analyzed_reviews.csv",
        mime="text/csv",
        type="primary",
        width="stretch"
    )


# ==========================================================
# UPLOADED CSV DASHBOARD
# ==========================================================

if uploaded_file is not None:

    st.header("📂 Uploaded Dataset Analysis")

    try:

        uploaded_reviews = pd.read_csv(uploaded_file)

        if uploaded_reviews.empty or len(uploaded_reviews.columns) == 0:

            st.warning(
                "The uploaded CSV is empty."
            )

        else:

            st.success("CSV uploaded successfully!")

            st.write(f"**Total rows:** {len(uploaded_reviews):,}")
            st.write(f"**Total columns:** {len(uploaded_reviews.columns)}")

            # --------------------------------------------------
            # COLUMN DETECTION
            # --------------------------------------------------

            normalized_columns = {
                col: normalize_column_name(col)
                for col in uploaded_reviews.columns
            }

            possible_review_columns = [
                "text",
                "review",
                "reviews",
                "review_text",
                "customer_review",
                "customer_reviews",
                "feedback",
                "comment",
                "comments"
            ]

            detected_review_column = None

            for original_column, normalized_column in normalized_columns.items():

                if normalized_column in possible_review_columns:

                    detected_review_column = original_column
                    break

            st.subheader("Review Column Detection")

            if detected_review_column is not None:

                st.success(
                    f"Review column detected: {detected_review_column}"
                )

                column_index = uploaded_reviews.columns.tolist().index(
                    detected_review_column
                )

                selected_review_column = st.selectbox(
                    "Confirm the review text column",
                    uploaded_reviews.columns.tolist(),
                    index=column_index,
                    key="uploaded_review_column"
                )

            else:

                st.warning(
                    "Review column could not be automatically identified. "
                    "Please select it manually."
                )

                selected_review_column = st.selectbox(
                    "Select the review text column",
                    uploaded_reviews.columns.tolist(),
                    key="uploaded_review_column"
                )

            # --------------------------------------------------
            # PREVIEW
            # --------------------------------------------------

            st.subheader("Dataset Preview")

            st.dataframe(
                uploaded_reviews.head(10),
                width="stretch",
                hide_index=True
            )

            st.subheader("Sample Reviews")

            st.dataframe(
                uploaded_reviews[[selected_review_column]].head(5),
                width="stretch",
                hide_index=True
            )

            # --------------------------------------------------
            # DATA QUALITY
            # --------------------------------------------------

            uploaded_texts = (
                uploaded_reviews[selected_review_column]
                .fillna("")
                .astype(str)
            )

            valid_mask = uploaded_texts.str.strip() != ""

            total_rows_uploaded = len(uploaded_reviews)
            blank_reviews_count = int((~valid_mask).sum())

            st.subheader("Data Quality")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Rows",
                f"{total_rows_uploaded:,}"
            )

            col2.metric(
                "Valid Reviews",
                f"{int(valid_mask.sum()):,}"
            )

            col3.metric(
                "Blank Reviews",
                f"{blank_reviews_count:,}"
            )

            if blank_reviews_count > 0:

                st.warning(
                    f"{blank_reviews_count:,} blank reviews will be skipped."
                )

            duplicate_count = int(
                uploaded_texts.loc[valid_mask].duplicated(keep="first").sum()
            )

            if duplicate_count > 0:

                st.warning(
                    f"Found {duplicate_count:,} duplicate reviews. "
                    "Duplicates are retained in the analysis."
                )

            else:

                st.success("No duplicate reviews detected.")

            # --------------------------------------------------
            # PROCESS REVIEWS
            # --------------------------------------------------

            if valid_mask.sum() == 0:

                st.warning(
                    "No valid review text found in the uploaded CSV."
                )

            else:

                valid_reviews = uploaded_reviews.loc[valid_mask].copy()

                valid_texts = uploaded_texts.loc[valid_mask].tolist()

                st.divider()
                st.subheader("🤖 Uploaded Review Sentiment Analysis")

                with st.spinner("Analyzing uploaded reviews..."):

                    uploaded_predictions = model.predict(
                        valid_texts
                    )

                    valid_reviews["Predicted Sentiment"] = (
                        uploaded_predictions
                    )

                    valid_reviews["predicted_sentiment"] = (
                        uploaded_predictions
                    )

                    valid_reviews["aspects"] = [
                        extract_aspects(review)
                        for review in valid_texts
                    ]

                st.success(
                    f"Successfully analyzed {len(valid_reviews):,} reviews."
                )

                # --------------------------------------------------
                # 1. SENTIMENT OVERVIEW
                # --------------------------------------------------

                render_uploaded_sentiment(
                    valid_reviews
                )

                # --------------------------------------------------
                # 2. COMPLAINT INTELLIGENCE
                # --------------------------------------------------

                uploaded_aspect_results = render_uploaded_complaints(
                    valid_reviews
                )

                # --------------------------------------------------
                # 3. BUSINESS RECOMMENDATIONS
                # --------------------------------------------------

                render_uploaded_recommendations(
                    uploaded_aspect_results
                )

                # --------------------------------------------------
                # 4. SENTIMENT TRENDS
                # --------------------------------------------------

                date_keywords = [
                    "date",
                    "time",
                    "timestamp",
                    "created_at",
                    "posted_on",
                    "published",
                    "datetime"
                ]

                possible_date_columns = [
                    col for col in uploaded_reviews.columns
                    if any(
                        keyword in normalize_column_name(col)
                        for keyword in date_keywords
                    )
                ]

                if possible_date_columns:

                    st.caption(
                        "Date columns detected: "
                        + ", ".join(map(str, possible_date_columns))
                    )

                render_uploaded_trends(
                    valid_reviews,
                    possible_date_columns
                )

                # --------------------------------------------------
                # 5. REVIEW EXPLORER
                # --------------------------------------------------

                render_uploaded_explorer(
                    valid_reviews,
                    selected_review_column
                )

                # --------------------------------------------------
                # 6. EXPORT
                # --------------------------------------------------

                render_uploaded_export(
                    valid_reviews,
                    selected_review_column
                )

    except Exception as e:

        st.error(
            f"Unable to process the uploaded CSV: {e}"
        )


# ==========================================================
# ORIGINAL YELP DASHBOARD
# ==========================================================

if uploaded_file is None:

    # ------------------------------------------------------
    # SIDEBAR FILTERS
    # ------------------------------------------------------

    st.sidebar.title("Review Filters")

    sentiment_options = [
        "All",
        "Positive",
        "Neutral",
        "Negative"
    ]

    selected_sentiment = st.sidebar.selectbox(
        "Filter by sentiment",
        sentiment_options
    )

    businesses = ["All Businesses"] + sorted(
        df["name"].dropna().unique().tolist()
    )

    selected_business = st.sidebar.selectbox(
        "Select Business",
        businesses
    )

    # ------------------------------------------------------
    # APPLY FILTERS
    # ------------------------------------------------------

    filtered_df = df.copy()

    if selected_sentiment != "All":

        filtered_df = filtered_df[
            filtered_df["predicted_sentiment"] == selected_sentiment
        ]

    if selected_business != "All Businesses":

        filtered_df = filtered_df[
            filtered_df["name"] == selected_business
        ]

    # Business-only filter for complaint analysis.
    # Sentiment filter intentionally not applied.
    business_df = df.copy()

    if selected_business != "All Businesses":

        business_df = business_df[
            business_df["name"] == selected_business
        ]

    # ------------------------------------------------------
    # KPI METRICS
    # ------------------------------------------------------

    total_reviews = len(filtered_df)

    positive_count = (
        filtered_df["predicted_sentiment"] == "Positive"
    ).sum()

    negative_count = (
        filtered_df["predicted_sentiment"] == "Negative"
    ).sum()

    neutral_count = (
        filtered_df["predicted_sentiment"] == "Neutral"
    ).sum()

    positive_percentage = (
        positive_count / total_reviews * 100
        if total_reviews else 0
    )

    negative_percentage = (
        negative_count / total_reviews * 100
        if total_reviews else 0
    )

    neutral_percentage = (
        neutral_count / total_reviews * 100
        if total_reviews else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Reviews", f"{total_reviews:,}")
    col2.metric("Positive Reviews", f"{positive_percentage:.1f}%")
    col3.metric("Negative Reviews", f"{negative_percentage:.1f}%")
    col4.metric("Neutral Reviews", f"{neutral_percentage:.1f}%")

    st.divider()

    # ------------------------------------------------------
    # SENTIMENT CHARTS
    # ------------------------------------------------------

    st.subheader("Sentiment Distribution")

    chart_col1, chart_col2 = st.columns(2)

    sentiment_counts = (
        filtered_df["predicted_sentiment"]
        .value_counts()
        .reindex(
            ["Positive", "Neutral", "Negative"],
            fill_value=0
        )
        .reset_index()
    )

    sentiment_counts.columns = [
        "Sentiment",
        "Count"
    ]

    with chart_col1:

        fig_pie = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Count",
            title="Review Sentiment Breakdown",
            hole=0.45
        )

        st.plotly_chart(
            fig_pie,
            width="stretch"
        )

    with chart_col2:

        fig_bar = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            title="Number of Reviews per Sentiment",
            text="Count"
        )

        fig_bar.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig_bar,
            width="stretch"
        )

    st.divider()

    # ------------------------------------------------------
    # REVIEW EXPLORER
    # ------------------------------------------------------

    st.subheader("Customer Review Explorer")

    st.caption(
        "Explore customer feedback and the sentiment predicted by our ML model."
    )

    display_columns = [
        column for column in [
            "text",
            "predicted_sentiment",
            "stars_review",
            "city"
        ]
        if column in filtered_df.columns
    ]

    review_display = filtered_df[display_columns].copy()

    review_display = review_display.rename(
        columns={
            "text": "Customer Review",
            "predicted_sentiment": "Predicted Sentiment",
            "stars_review": "Star Rating",
            "city": "City"
        }
    )

    st.dataframe(
        review_display,
        width="stretch",
        hide_index=True
    )

    st.caption(
        f"Showing {len(filtered_df):,} reviews"
    )

    # ------------------------------------------------------
    # ASPECT & COMPLAINT ANALYSIS
    # ------------------------------------------------------

    st.divider()

    st.title("Aspect & Complaint Intelligence")

    st.markdown(
        "Identify which parts of the customer experience receive "
        "the most negative feedback."
    )

    aspect_results = analyze_aspects(
        business_df
    )

    # Only show actual mentioned aspects
    mentioned_aspects = aspect_results[
        aspect_results["Total Mentions"] > 0
    ].copy()

    if mentioned_aspects.empty:

        st.info(
            "No aspects were detected for this business."
        )

    else:

        highest_rate_aspect = mentioned_aspects.loc[
            mentioned_aspects["Complaint Rate (%)"].idxmax()
        ]

        highest_volume_aspect = mentioned_aspects.loc[
            mentioned_aspects["Negative Reviews"].idxmax()
        ]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Highest Complaint Rate",
            highest_rate_aspect["Aspect"],
            f"{highest_rate_aspect['Complaint Rate (%)']}%"
        )

        col2.metric(
            "Most Negative Reviews",
            highest_volume_aspect["Aspect"],
            f"{highest_volume_aspect['Negative Reviews']:,} reviews"
        )

        col3.metric(
            "Aspects Analyzed",
            len(mentioned_aspects)
        )

        # Complaint rate chart
        st.subheader("Complaint Rate by Aspect")

        fig_rate = px.bar(
            mentioned_aspects.sort_values(
                "Complaint Rate (%)",
                ascending=True
            ),
            x="Complaint Rate (%)",
            y="Aspect",
            orientation="h",
            text="Complaint Rate (%)",
            title="Percentage of Aspect Reviews Predicted Negative"
        )

        fig_rate.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig_rate,
            width="stretch"
        )

        # Complaint volume chart
        st.subheader("Negative Complaint Volume")

        fig_volume = px.bar(
            mentioned_aspects.sort_values(
                "Negative Reviews",
                ascending=True
            ),
            x="Negative Reviews",
            y="Aspect",
            orientation="h",
            text="Negative Reviews",
            title="Number of Negative Reviews per Aspect"
        )

        fig_volume.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig_volume,
            width="stretch"
        )

        st.subheader("Complaint Analysis Details")

        st.dataframe(
            mentioned_aspects,
            width="stretch",
            hide_index=True
        )

    # ------------------------------------------------------
    # BUSINESS RECOMMENDATIONS
    # ------------------------------------------------------

    st.divider()

    st.header("AI-Powered Business Recommendations")

    st.markdown(
        "Actionable suggestions generated from recurring customer complaints."
    )

    recommendation_aspects = analyze_aspects(
        business_df
    )

    recommendations = generate_recommendations(
        recommendation_aspects
    )

    if recommendations:

        for item in recommendations:

            with st.container(border=True):

                st.subheader(
                    f"{item['Aspect']} — {item['Priority']} Priority"
                )

                col1, col2 = st.columns(2)

                col1.metric(
                    "Negative Reviews",
                    f"{item['Negative Reviews']:,}"
                )

                col2.metric(
                    "Complaint Rate",
                    f"{item['Complaint Rate (%)']:.2f}%"
                )

                st.markdown("**Recommended Action**")

                st.write(
                    item["Recommendation"]
                )

    else:

        st.info(
            "No complaint recommendations available."
        )

    # ------------------------------------------------------
    # SENTIMENT TRENDS
    # ------------------------------------------------------

    st.divider()

    st.header("Customer Sentiment Trends")

    st.markdown(
        "Track customer sentiment patterns over time using review dates."
    )

    trend_df = business_df.copy()

    trend_df["date"] = pd.to_datetime(
        trend_df["date"],
        errors="coerce"
    )

    trend_df = trend_df.dropna(
        subset=["date"]
    )

    if trend_df.empty:

        st.info(
            "No valid review dates are available for trend analysis."
        )

    else:

        trend_df["Month"] = (
            trend_df["date"]
            .dt.to_period("M")
            .dt.to_timestamp()
        )

        monthly_counts = (
            trend_df.groupby(
                ["Month", "predicted_sentiment"]
            )
            .size()
            .reset_index(name="Review Count")
        )

        st.subheader("Monthly Sentiment Volume")

        fig_trend = px.line(
            monthly_counts,
            x="Month",
            y="Review Count",
            color="predicted_sentiment",
            markers=True,
            title="Monthly Positive, Neutral & Negative Reviews"
        )

        st.plotly_chart(
            fig_trend,
            width="stretch"
        )

        # Monthly negative sentiment rate
        monthly_total = (
            trend_df.groupby("Month")
            .size()
            .reset_index(name="Total Reviews")
        )

        monthly_negative = (
            trend_df[
                trend_df["predicted_sentiment"] == "Negative"
            ]
            .groupby("Month")
            .size()
            .reset_index(name="Negative Reviews")
        )

        negative_trend = monthly_total.merge(
            monthly_negative,
            on="Month",
            how="left"
        )

        negative_trend["Negative Reviews"] = (
            negative_trend["Negative Reviews"].fillna(0)
        )

        negative_trend["Negative Rate (%)"] = (
            negative_trend["Negative Reviews"]
            / negative_trend["Total Reviews"]
            * 100
        )

        st.subheader("Monthly Negative Sentiment Rate")

        fig_negative = px.line(
            negative_trend,
            x="Month",
            y="Negative Rate (%)",
            markers=True,
            title="Monthly Negative Sentiment Rate"
        )

        st.plotly_chart(
            fig_negative,
            width="stretch"
        )

        st.subheader("Monthly Trend Data")

        st.dataframe(
            negative_trend.sort_values(
                "Month",
                ascending=False
            ),
            width="stretch",
            hide_index=True
        )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Revio AI | AI-powered customer feedback intelligence | Proof of Concept"
)
