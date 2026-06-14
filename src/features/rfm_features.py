import pandas as pd
import os


# ---------------------------------
# Create RFM Features
# ---------------------------------

def create_rfm_features(df: pd.DataFrame):

    print("\nCreating RFM features...")

    df = df.copy()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg({

        "InvoiceDate": [
            lambda x: (snapshot_date - x.max()).days,   # Recency
            lambda x: (snapshot_date - x.min()).days    # T (customer age)
        ],

        "InvoiceNo": "nunique",

        "Revenue": "sum"

    })

    rfm.columns = ["Recency", "T", "Frequency", "TotalRevenue"]

    # Monetary value must be average order value
    rfm["Monetary"] = rfm["TotalRevenue"] / rfm["Frequency"]

    rfm = rfm.drop(columns=["TotalRevenue"])

    # BG/NBD requires Frequency > 0
    rfm = rfm[rfm["Frequency"] > 0]

    print("RFM features created")
    print(rfm.head())

    return rfm


# ---------------------------------
# Save RFM Dataset
# ---------------------------------

def save_rfm_features(rfm):

    output_path = "data/processed/rfm_features.csv"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rfm.to_csv(output_path)

    print(f"\nRFM dataset saved to {output_path}")