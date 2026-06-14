import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------
# Survival Duration Distribution
# ---------------------------------

def plot_duration_distribution(df: pd.DataFrame):

    fig, ax = plt.subplots()

    ax.hist(df["duration"], bins=40)

    ax.set_title("Customer Lifetime Distribution")

    ax.set_xlabel("Days")

    ax.set_ylabel("Customers")

    return fig


# ---------------------------------
# Survival Event Chart
# ---------------------------------

def plot_event_distribution(df: pd.DataFrame):

    fig, ax = plt.subplots()

    counts = df["event_observed"].value_counts()

    ax.bar(
        ["Active", "Churned"],
        counts.values
    )

    ax.set_title("Customer Survival Events")

    return fig