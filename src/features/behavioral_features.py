import pandas as pd
import os


def purchase_frequency(df):

    freq = df.groupby("CustomerID")["InvoiceNo"].nunique()
    return freq.rename("PurchaseFrequency")


def average_basket_size(df):

    basket = df.groupby(["CustomerID", "InvoiceNo"])["Quantity"].sum()
    basket_size = basket.groupby("CustomerID").mean()

    return basket_size.rename("AvgBasketSize")


def average_purchase_value(df):

    order_value = df.groupby(["CustomerID", "InvoiceNo"])["Revenue"].sum()
    avg_value = order_value.groupby("CustomerID").mean()

    return avg_value.rename("AvgPurchaseValue")


def customer_lifetime(df):

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    lifetime = df.groupby("CustomerID")["InvoiceDate"].agg(["min", "max"])

    lifetime_days = (lifetime["max"] - lifetime["min"]).dt.days

    return lifetime_days.rename("CustomerLifetime")


def create_behavioral_features(df):

    print("\nCreating behavioral features...")

    freq = purchase_frequency(df)
    basket = average_basket_size(df)
    value = average_purchase_value(df)
    lifetime = customer_lifetime(df)

    features = pd.concat([freq, basket, value, lifetime], axis=1)

    features = features.dropna()

    print("Behavioral features created:", features.shape)

    if features.shape[0] == 0:
        raise ValueError("Behavioral features dataframe is empty.")

    return features


def save_behavioral_features(features):

    os.makedirs("data/processed", exist_ok=True)

    output_path = "data/processed/behavioral_features.csv"

    features.to_csv(output_path)

    print("Behavioral features saved to", output_path)

    print("Behavioral features preview:")
    
    print(features.head())