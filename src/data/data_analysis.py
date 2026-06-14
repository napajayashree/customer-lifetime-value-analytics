import pandas as pd


def dataset_overview(df: pd.DataFrame):

    print("\n===== DATASET OVERVIEW =====")

    print("\nDataset Shape")
    print(df.shape)

    print("\nColumn Names")
    print(df.columns.tolist())

    print("\nFirst 5 Rows")
    print(df.head())


def analyze_missing_values(df: pd.DataFrame):

    print("\n===== MISSING VALUE ANALYSIS =====")

    missing = df.isnull().sum()

    print(missing)


def basic_statistics(df: pd.DataFrame):

    print("\n===== BASIC DATASET STATISTICS =====")

    total_transactions = df.shape[0]
    unique_customers = df["CustomerID"].nunique()
    unique_products = df["StockCode"].nunique()

    print("Total Transactions:", total_transactions)
    print("Unique Customers:", unique_customers)
    print("Unique Products:", unique_products)


def revenue_statistics(df: pd.DataFrame):

    print("\n===== REVENUE STATISTICS =====")

    temp = df.copy()

    if "Revenue" not in temp.columns:
        temp["Revenue"] = temp["Quantity"] * temp["UnitPrice"]

    total_revenue = temp["Revenue"].sum()
    avg_order_value = temp["Revenue"].mean()

    print("Total Revenue:", round(total_revenue, 2))
    print("Average Transaction Value:", round(avg_order_value, 2))


def country_analysis(df: pd.DataFrame):

    print("\n===== COUNTRY ANALYSIS =====")

    temp = df.copy()

    if "Revenue" not in temp.columns:
        temp["Revenue"] = temp["Quantity"] * temp["UnitPrice"]

    country_sales = (
        temp.groupby("Country")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    print(country_sales.head(10))


def monthly_sales_analysis(df: pd.DataFrame):

    print("\n===== MONTHLY SALES ANALYSIS =====")

    temp = df.copy()

    if "Revenue" not in temp.columns:
        temp["Revenue"] = temp["Quantity"] * temp["UnitPrice"]

    temp["InvoiceDate"] = pd.to_datetime(temp["InvoiceDate"])

    temp["Month"] = temp["InvoiceDate"].dt.to_period("M")

    monthly_sales = temp.groupby("Month")["Revenue"].sum()

    print(monthly_sales.head(12))


def dataset_statistics(df: pd.DataFrame):

    dataset_overview(df)

    analyze_missing_values(df)

    basic_statistics(df)

    revenue_statistics(df)

    print("\nDataset statistics completed.")


def run_data_analysis(df: pd.DataFrame):

    dataset_overview(df)

    analyze_missing_values(df)

    basic_statistics(df)

    revenue_statistics(df)

    country_analysis(df)

    monthly_sales_analysis(df)

    print("\nData analysis completed.")