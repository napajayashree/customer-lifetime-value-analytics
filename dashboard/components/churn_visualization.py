import plotly.express as px
import pandas as pd


# ---------------------------------
# Churn Distribution
# ---------------------------------

def plot_churn_distribution(df: pd.DataFrame):

    churn_counts = df["event_observed"].value_counts().reset_index()

    churn_counts.columns = ["Churn", "Count"]

    fig = px.bar(
        churn_counts,
        x="Churn",
        y="Count",
        title="Customer Churn Distribution"
    )

    return fig


# ---------------------------------
# Recency vs Churn
# ---------------------------------

def plot_recency_vs_churn(df: pd.DataFrame):

    fig = px.scatter(
        df,
        x="Recency",
        y="Monetary",
        color="event_observed",
        title="Recency vs Monetary Value (Churn Analysis)"
    )

    return fig