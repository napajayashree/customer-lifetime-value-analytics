import pandas as pd
import os


# ---------------------------------
# Load Dataset
# ---------------------------------

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV or Excel.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    print("\nLoading dataset from:", file_path)

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".csv":

        df = pd.read_csv(
            file_path,
            encoding="ISO-8859-1",
            engine="python",
            on_bad_lines="skip",
            low_memory=False
        )

    elif extension in [".xlsx", ".xls"]:

        df = pd.read_excel(
            file_path,
            engine="openpyxl"
        )

    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")

    print("Dataset loaded successfully")
    print("Dataset shape:", df.shape)

    # ---------------------------------
    # Standardize Column Names
    # ---------------------------------

    df.columns = df.columns.str.strip()

    column_mapping = {
        "Invoice": "InvoiceNo",
        "Price": "UnitPrice",
        "Customer ID": "CustomerID"
    }

    df = df.rename(columns=column_mapping)

    print("\nStandardized Columns:")
    print(df.columns.tolist())

    return df


# ---------------------------------
# Preview Dataset
# ---------------------------------

def preview_dataset(df: pd.DataFrame, rows=5):

    print("\n===== DATASET PREVIEW =====")
    print(df.head(rows))


# ---------------------------------
# Dataset Info
# ---------------------------------

def dataset_info(df: pd.DataFrame):

    print("\n===== DATASET INFO =====")

    df.info()

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDataset Shape:")
    print(df.shape)