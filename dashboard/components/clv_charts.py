import plotly.express as px
import pandas as pd


# ---------------------------------
# CLV Distribution
# ---------------------------------

def plot_clv_distribution(df: pd.DataFrame):

    fig = px.histogram(
        df,
        x="Predicted_CLV",
        nbins=50,
        title="Customer Lifetime Value Distribution"
    )

    return fig


# ---------------------------------
# Top Customers Chart
# ---------------------------------

def plot_top_customers(df: pd.DataFrame, n=20):

    top_customers = df.sort_values(
        by="Predicted_CLV",
        ascending=False
    ).head(n)

    fig = px.bar(
        top_customers,
        x="CustomerID",
        y="Predicted_CLV",
        title=f"Top {n} High Value Customers"
    )

    return fig


# ---------------------------------
# CLV Segment Chart
# ---------------------------------

def plot_clv_segments(df: pd.DataFrame):

    segment_counts = df["CLV_Segment"].value_counts().reset_index()

    segment_counts.columns = ["Segment", "Count"]

    fig = px.pie(
        segment_counts,
        names="Segment",
        values="Count",
        title="CLV Segmentation"
    )

    return fig