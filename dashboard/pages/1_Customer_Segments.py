import streamlit as st
import pandas as pd
import plotly.express as px


st.title("Customer Segmentation")


# Load dataset
try:

    df = pd.read_csv("data/processed/customer_segments.csv")

    st.subheader("Segmented Customers Dataset")

    st.dataframe(df.head(50))


    st.subheader("Segment Distribution")

    segment_counts = df["Segment"].value_counts().reset_index()
    segment_counts.columns = ["Segment", "Count"]

    fig = px.bar(
        segment_counts,
        x="Segment",
        y="Count",
        color="Segment",
        title="Customer Segments Distribution"
    )

    st.plotly_chart(fig)


except:
    st.warning("Customer segmentation dataset not found.")