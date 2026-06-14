import pandas as pd
import os


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:

    print("\nStarting data cleaning...")

    original_rows = df.shape[0]

    df.columns = df.columns.str.strip()

    column_mapping = {
        "Invoice": "InvoiceNo",
        "Price": "UnitPrice",
        "Customer ID": "CustomerID"
    }

    df = df.rename(columns=column_mapping)

    # Remove missing customers
    df = df.dropna(subset=["CustomerID"])

    # Remove returns
    df = df[df["Quantity"] > 0]

    # Remove invalid prices
    df = df[df["UnitPrice"] > 0]

    # Remove duplicates
    df = df.drop_duplicates()

    # Fix types
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # Revenue
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    cleaned_rows = df.shape[0]

    print("\nCleaning completed")
    print("Rows before cleaning:", original_rows)
    print("Rows after cleaning:", cleaned_rows)

    if cleaned_rows == 0:
        raise ValueError("Cleaning removed all rows.")

    return df


def save_clean_data(df: pd.DataFrame):

    os.makedirs("data/interim", exist_ok=True)

    path = "data/interim/cleaned_transactions.csv"

    df.to_csv(path, index=False)

    print("Clean dataset saved to:", path)