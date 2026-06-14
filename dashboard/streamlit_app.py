import streamlit as st
import requests
import pandas as pd


# ---------------------------------
# Configuration
# ---------------------------------

API_URL = "http://localhost:8000/predict_clv"


# ---------------------------------
# Page Config
# ---------------------------------

st.set_page_config(
    page_title="Customer Lifetime Value Dashboard",
    page_icon="📈",
    layout="wide"
)


# ---------------------------------
# Title
# ---------------------------------

st.title("Customer Lifetime Value Prediction System")

st.markdown(
"""
This dashboard allows you to:

- Predict **Customer Lifetime Value**
- Analyze **customer segments**
- View **survival analysis**
"""
)


# ---------------------------------
# Sidebar Navigation
# ---------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "CLV Prediction",
        "Dataset Preview"
    ]
)


# ---------------------------------
# CLV Prediction Page
# ---------------------------------

if page == "CLV Prediction":

    st.header("Predict Customer Lifetime Value")

    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.number_input("Recency", value=30)
        frequency = st.number_input("Frequency", value=5)

    with col2:
        monetary = st.number_input("Monetary", value=250.0)
        purchase_freq = st.number_input("Purchase Frequency", value=5)

    with col3:
        basket_size = st.number_input("Average Basket Size", value=3.0)
        avg_purchase = st.number_input("Average Purchase Value", value=120.0)

    lifetime = st.number_input("Customer Lifetime", value=180)

    if st.button("Predict CLV"):

        payload = {
            "Recency": recency,
            "Frequency": frequency,
            "Monetary": monetary,
            "PurchaseFrequency": purchase_freq,
            "AvgBasketSize": basket_size,
            "AvgPurchaseValue": avg_purchase,
            "CustomerLifetime": lifetime
        }

        try:

            response = requests.post(API_URL, json=payload)

            result = response.json()

            clv_value = result["Predicted_CLV"]

            st.success(f"Predicted CLV: {round(clv_value,2)}")

        except Exception as e:

            st.error("API connection failed")

            st.write(e)


# ---------------------------------
# Dataset Preview
# ---------------------------------

elif page == "Dataset Preview":

    st.header("Processed Dataset")

    try:

        df = pd.read_csv("data/processed/clv_dataset.csv")

        st.dataframe(df.head(50))

        st.write("Dataset Shape:", df.shape)

    except:

        st.warning("Processed dataset not found.")