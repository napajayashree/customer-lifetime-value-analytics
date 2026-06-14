import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.title("Customer Survival Analysis")


try:

    df = pd.read_csv("data/processed/survival_dataset.csv")

    st.subheader("Survival Dataset")

    st.dataframe(df.head(50))


    st.subheader("Duration Distribution")

    fig, ax = plt.subplots()

    ax.hist(df["duration"], bins=40)

    ax.set_xlabel("Customer Lifetime (Days)")
    ax.set_ylabel("Customers")
    ax.set_title("Customer Lifetime Distribution")

    st.pyplot(fig)


    st.subheader("Churn Events")

    churn_counts = df["event_observed"].value_counts()

    st.write("Churned vs Active Customers")

    st.bar_chart(churn_counts)


except:
    st.warning("Survival dataset not found.")