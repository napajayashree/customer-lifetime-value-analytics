import pandas as pd


REQUIRED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country"
]


def validate_columns(df: pd.DataFrame):

    print("\nValidating required columns...")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print("All required columns present")


def validate_data_types(df: pd.DataFrame):

    print("\nChecking column data types...")

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    print("Data types validated")

    return df


def check_missing_values(df: pd.DataFrame):

    print("\nMissing values summary:")

    missing = df.isnull().sum()

    print(missing)

    return missing


def check_duplicates(df: pd.DataFrame):

    duplicates = df.duplicated().sum()

    print("\nDuplicate rows:", duplicates)

    return duplicates


def validate_value_ranges(df: pd.DataFrame):

    negative_quantity = (df["Quantity"] < 0).sum()
    negative_price = (df["UnitPrice"] < 0).sum()

    print("\nNegative quantity rows:", negative_quantity)
    print("Negative price rows:", negative_price)

    return {
        "negative_quantity": negative_quantity,
        "negative_price": negative_price
    }


def validate_dataset(df: pd.DataFrame):

    print("\nRunning dataset validation...")

    validate_columns(df)

    df = validate_data_types(df)

    check_missing_values(df)

    check_duplicates(df)

    validate_value_ranges(df)

    print("\nDataset validation completed successfully")

    return df