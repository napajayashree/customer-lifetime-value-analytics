import pandas as pd
import os


# ------------------------------------------------
# Customer activity window
# ------------------------------------------------

def customer_activity_window(df: pd.DataFrame):

    print("\nCalculating customer activity window...")

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    activity = df.groupby("CustomerID")["InvoiceDate"].agg(["min", "max"])

    activity.columns = ["FirstPurchase", "LastPurchase"]

    return activity


# ------------------------------------------------
# Create survival dataset
# ------------------------------------------------

def create_survival_dataset(df: pd.DataFrame, churn_threshold=180):

    print("\nCreating survival dataset...")

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    snapshot_date = df["InvoiceDate"].max()

    activity = customer_activity_window(df)

    # duration customer observed
    activity["duration"] = (
        activity["LastPurchase"] - activity["FirstPurchase"]
    ).dt.days

    # inactivity
    activity["days_since_last_purchase"] = (
        snapshot_date - activity["LastPurchase"]
    ).dt.days

    # churn event
    activity["event_observed"] = (
        activity["days_since_last_purchase"] > churn_threshold
    ).astype(int)

    survival_df = activity.reset_index()

    survival_df = survival_df[
        ["CustomerID", "duration", "event_observed"]
    ]

    print("Survival dataset created")
    print("Shape:", survival_df.shape)

    return survival_df


# ------------------------------------------------
# Merge survival with customer features
# ------------------------------------------------

def merge_survival_features(features_df, survival_df):

    print("\nMerging survival data with customer features...")

    merged = survival_df.merge(
        features_df,
        on="CustomerID",
        how="inner"
    )

    print("Merged survival dataset shape:", merged.shape)

    return merged


# ------------------------------------------------
# Save survival dataset
# ------------------------------------------------

def save_survival_dataset(df):

    path = "data/processed/survival_dataset.csv"

    os.makedirs(os.path.dirname(path), exist_ok=True)

    df.to_csv(path, index=False)

    print(f"Survival dataset saved to {path}")