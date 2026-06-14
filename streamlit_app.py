import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# ------------------------------------------------
# Page Config
# ------------------------------------------------

st.set_page_config(
    page_title="Customer Lifetime Analytics",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"

# ------------------------------------------------
# Minimal Styling
# ------------------------------------------------

st.markdown("""
<style>
body {background-color:white;}
.block-container {padding-top:2rem;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Load Data
# ------------------------------------------------

@st.cache_data
def load_data():

    clv_df = pd.read_csv("data/processed/clv_dataset.csv")
    segments_df = pd.read_csv("data/processed/customer_segments.csv")

    return clv_df, segments_df


clv_df, segments_df = load_data()

# ------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Dataset Overview",
        "CLV Distribution",
        "Customer Segmentation",
        "Feature Importance",
        "Customer Prediction",
        "Batch CLV Prediction",
        "Probabilistic CLV",
        "Churn Risk Prediction"
    ]
)

# ------------------------------------------------
# Title
# ------------------------------------------------

st.title("Customer Lifetime Analytics Dashboard")
st.caption("Customer value prediction, segmentation, and churn risk analysis")

# =================================================
# Dataset Overview
# =================================================

if page == "Dataset Overview":

    st.header("Dataset Overview")

    total_customers = clv_df.shape[0]
    total_revenue = clv_df["Monetary"].sum()
    avg_clv = clv_df["CLV"].mean()
    transactions = clv_df["Frequency"].sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Total Revenue", f"${total_revenue:,.0f}")
    col3.metric("Average CLV", f"${avg_clv:,.0f}")
    col4.metric("Transactions", f"{transactions:,}")

# =================================================
# CLV Distribution
# =================================================

elif page == "CLV Distribution":

    st.header("CLV Distribution")

    fig = px.histogram(
        clv_df,
        x="CLV",
        nbins=50,
        color_discrete_sequence=["#3498DB"]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Customers")

    top_customers = clv_df.sort_values("CLV", ascending=False).head(10)

    fig2 = px.bar(
        top_customers,
        x=top_customers.index.astype(str),
        y="CLV",
        color_discrete_sequence=["#2C3E50"]
    )

    st.plotly_chart(fig2, use_container_width=True)

# =================================================
# Customer Segmentation
# =================================================

elif page == "Customer Segmentation":

    st.header("Customer Segmentation")

    segment_counts = segments_df["SegmentLabel"].value_counts()

    fig = px.bar(
        segment_counts,
        color_discrete_sequence=["#27AE60"]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Revenue Contribution by Segment")

    revenue_segment = segments_df.groupby(
        "SegmentLabel"
    )["CLV"].sum()

    fig2 = px.pie(
        revenue_segment,
        names=revenue_segment.index,
        values=revenue_segment.values
    )

    st.plotly_chart(fig2, use_container_width=True)

# =================================================
# Feature Importance
# =================================================

elif page == "Feature Importance":

    st.header("Feature Importance")

    try:

        response = requests.get(f"{API_URL}/model/features")

        if response.status_code != 200:
            st.error(response.text)
        else:

            feature_data = pd.DataFrame(response.json())

            fig = px.bar(
                feature_data,
                x="Importance",
                y="Feature",
                orientation="h",
                color_discrete_sequence=["#2C3E50"]
            )

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:

        st.error("FastAPI server not reachable")
        st.write(e)

# =================================================
# Single CLV Prediction
# =================================================

elif page == "Customer Prediction":

    st.header("Customer CLV Prediction")

    recency = st.number_input("Recency", 0.0)
    frequency = st.number_input("Frequency", 0.0)
    monetary = st.number_input("Monetary", 0.0)
    purchase_freq = st.number_input("PurchaseFrequency", 0.0)
    basket = st.number_input("AvgBasketSize", 0.0)
    purchase_value = st.number_input("AvgPurchaseValue", 0.0)
    lifetime = st.number_input("CustomerLifetime", 0.0)

    if st.button("Predict CLV"):

        payload = {
            "Recency": recency,
            "Frequency": frequency,
            "Monetary": monetary,
            "PurchaseFrequency": purchase_freq,
            "AvgBasketSize": basket,
            "AvgPurchaseValue": purchase_value,
            "CustomerLifetime": lifetime
        }

        try:

            response = requests.post(
                f"{API_URL}/predict_clv",
                json=payload
            )

            if response.status_code != 200:
                st.error(response.text)
            else:

                result = response.json()

                st.success(
                    f"Predicted CLV: ${result['Predicted_CLV']:.2f}"
                )

        except Exception as e:

            st.error("Prediction API error")
            st.write(e)

# =================================================
# Batch CLV Prediction
# =================================================

elif page == "Batch CLV Prediction":

    st.header("Batch CLV Prediction")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        st.dataframe(df.head())

        if st.button("Run Batch Prediction"):

            payload = {
                "customers": df.to_dict(orient="records")
            }

            try:

                response = requests.post(
                    f"{API_URL}/predict_clv_batch",
                    json=payload
                )

                if response.status_code != 200:
                    st.error(response.text)
                else:

                    result = response.json()

                    df["Predicted_CLV"] = result["Predicted_CLV"]

                    st.success("Batch prediction completed")

                    st.dataframe(df)

                    st.download_button(
                        "Download Results",
                        df.to_csv(index=False),
                        "clv_predictions.csv"
                    )

            except Exception as e:

                st.error("Batch prediction API error")
                st.write(e)

# =================================================
# Probabilistic CLV
# =================================================

elif page == "Probabilistic CLV":

    st.header("Probabilistic CLV Prediction")
    st.caption("BG/NBD + Gamma-Gamma Model")

    frequency = st.number_input("Frequency", 0.0)
    recency = st.number_input("Recency", 0.0)
    T = st.number_input("Customer Age (T)", 0.0)
    monetary = st.number_input("Monetary Value", 0.0)

    if st.button("Predict Probabilistic CLV"):

        payload = {
            "Frequency": frequency,
            "Recency": recency,
            "T": T,
            "Monetary": monetary
        }

        try:

            response = requests.post(
                f"{API_URL}/predict_clv_probabilistic",
                json=payload
            )

            if response.status_code != 200:
                st.error(response.text)
            else:

                result = response.json()

                st.success(
                    f"Expected CLV (12 months): ${result['Expected_CLV_12_months']:.2f}"
                )

        except Exception as e:

            st.error("Probabilistic CLV API error")
            st.write(e)

# =================================================
# Churn Risk Prediction
# =================================================

elif page == "Churn Risk Prediction":

    st.header("Churn Risk Prediction")

    duration = st.number_input("Duration", value=200.0)
    event = st.number_input("Event Observed (0 or 1)", 0, 1)

    st.subheader("Customer Features")

    recency = st.number_input("Recency", 0.0)
    frequency = st.number_input("Frequency", 0.0)
    monetary = st.number_input("Monetary", 0.0)
    purchase_freq = st.number_input("PurchaseFrequency", 0.0)
    basket = st.number_input("AvgBasketSize", 0.0)
    purchase_value = st.number_input("AvgPurchaseValue", 0.0)
    lifetime = st.number_input("CustomerLifetime", 0.0)

    if st.button("Predict Churn Risk"):

        payload = {
            "duration": duration,
            "event_observed": event,

            "Recency": recency,
            "Frequency": frequency,
            "Monetary": monetary,
            "PurchaseFrequency": purchase_freq,
            "AvgBasketSize": basket,
            "AvgPurchaseValue": purchase_value,
            "CustomerLifetime": lifetime
        }

        try:

            response = requests.post(
                f"{API_URL}/predict_churn_risk",
                json=payload
            )

            if response.status_code != 200:
                st.error(response.text)

            else:

                result = response.json()

                st.success(
                    f"Churn Risk Score: {result['Churn_Risk_Score']:.3f}"
                )

        except Exception as e:

            st.error("Survival API error")
            st.write(e)