import pandas as pd
import numpy as np

from src.models.ml_models.random_forest_clv import train_random_forest
from src.models.ml_models.xgboost_clv import train_xgboost


# ---------------------------------
# Sample Dataset Generator
# ---------------------------------

def sample_dataset():

    data = {
        "Recency": [10, 20, 30, 40, 50],
        "Frequency": [5, 3, 6, 2, 1],
        "Monetary": [200, 150, 400, 120, 80],
        "PurchaseFrequency": [5, 3, 6, 2, 1],
        "AvgBasketSize": [3, 2, 4, 1, 2],
        "AvgPurchaseValue": [120, 90, 200, 70, 50],
        "CustomerLifetime": [365, 240, 400, 120, 90],
        "CLV": [500, 300, 900, 150, 80]
    }

    df = pd.DataFrame(data)

    return df


# ---------------------------------
# Random Forest Test
# ---------------------------------

def test_random_forest_model():

    df = sample_dataset()

    X = df.drop(columns=["CLV"])
    y = df["CLV"]

    model = train_random_forest(X, y)

    preds = model.predict(X)

    assert len(preds) == len(X)

    assert isinstance(preds, np.ndarray)


# ---------------------------------
# XGBoost Test
# ---------------------------------

def test_xgboost_model():

    df = sample_dataset()

    X = df.drop(columns=["CLV"])
    y = df["CLV"]

    model = train_xgboost(X, y)

    preds = model.predict(X)

    assert len(preds) == len(X)

    assert isinstance(preds, np.ndarray)