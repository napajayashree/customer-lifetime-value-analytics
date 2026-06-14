import streamlit as st
import pandas as pd
import plotly.express as px


st.title("Customer Lifetime Value Predictions")


try:

    df = pd.read_csv("data/processed/clv_predictions.csv")

    st.subheader("CLV Predictions Dataset")

    st.dataframe(df.head(50))


    st.subheader("CLV Distribution")

    fig = px.histogram(
        df,
        x="Predicted_CLV",
        nbins=50,
        title="Distribution of Predicted CLV"
    )

    st.plotly_chart(fig)


    st.subheader("Top High Value Customers")

    top_customers = df.sort_values(
        by="Predicted_CLV",
        ascending=False
    ).head(20)

    st.dataframe(top_customers)


except:
    st.warning("CLV predictions dataset not found.")