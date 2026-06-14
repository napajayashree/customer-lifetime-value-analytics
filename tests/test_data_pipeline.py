import pandas as pd

from src.data.data_cleaning import clean_transactions
from src.features.rfm_features import create_rfm_features


# ---------------------------------
# Sample Test Dataset
# ---------------------------------

def sample_dataset():

    data = {
        "CustomerID": [1, 1, 2, 2],
        "InvoiceNo": ["A1", "A2", "B1", "B2"],
        "InvoiceDate": [
            "2023-01-01",
            "2023-02-01",
            "2023-01-10",
            "2023-03-01"
        ],
        "Quantity": [2, 3, 1, 5],
        "UnitPrice": [10, 15, 20, 5]
    }

    df = pd.DataFrame(data)

    return df


# ---------------------------------
# Data Cleaning Test
# ---------------------------------

def test_data_cleaning():

    df = sample_dataset()

    cleaned = clean_transactions(df)

    assert "Revenue" in cleaned.columns

    assert cleaned["Quantity"].min() > 0


# ---------------------------------
# RFM Feature Generation Test
# ---------------------------------

def test_rfm_features():

    df = sample_dataset()

    cleaned = clean_transactions(df)

    rfm = create_rfm_features(cleaned)

    assert "Recency" in rfm.columns
    assert "Frequency" in rfm.columns
    assert "Monetary" in rfm.columns

    assert len(rfm) > 0