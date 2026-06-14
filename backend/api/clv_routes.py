import pickle
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException

from src.utils.constants import RANDOM_FOREST_MODEL
from src.models.clv_models.bg_nbd_model import load_bg_nbd_model
from src.models.clv_models.gamma_gamma_model import load_gamma_gamma_model

from backend.schemas.request_schema import (
    CLVRequest,
    CLVBatchRequest,
    ProbabilisticCLVRequest,
    ProbabilisticCLVBatchRequest
)

router = APIRouter()


# ------------------------------------------------
# Load ML Model
# ------------------------------------------------

def load_ml_model():

    try:
        with open(RANDOM_FOREST_MODEL, "rb") as f:
            model = pickle.load(f)

        return model

    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Random Forest model not found"
        )


# ------------------------------------------------
# Feature order used during training
# ------------------------------------------------

FEATURE_COLUMNS = [
    "Recency",
    "Frequency",
    "Monetary",
    "PurchaseFrequency",
    "AvgBasketSize",
    "AvgPurchaseValue",
    "CustomerLifetime"
]


# ------------------------------------------------
# ML CLV Prediction (Single)
# ------------------------------------------------

@router.post("/predict_clv")
def predict_clv(customer: CLVRequest):

    model = load_ml_model()

    df = pd.DataFrame([customer.dict()])

    # Ensure correct feature order
    df = df[FEATURE_COLUMNS]

    pred_log = model.predict(df)

    pred = np.expm1(pred_log)

    return {
        "Predicted_CLV": float(pred[0])
    }


# ------------------------------------------------
# ML CLV Prediction (Batch)
# ------------------------------------------------

@router.post("/predict_clv_batch")
def predict_clv_batch(request: CLVBatchRequest):

    model = load_ml_model()

    df = pd.DataFrame([c.dict() for c in request.customers])

    df = df[FEATURE_COLUMNS]

    pred_log = model.predict(df)

    preds = np.expm1(pred_log)

    return {
        "Predicted_CLV": preds.tolist()
    }


# ------------------------------------------------
# Feature Importance
# ------------------------------------------------

@router.get("/model/features")
def get_feature_importance():

    model = load_ml_model()

    importance = model.feature_importances_

    df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": importance
    }).sort_values("Importance", ascending=False)

    return df.to_dict(orient="records")


# ------------------------------------------------
# Probabilistic CLV (Single)
# ------------------------------------------------

@router.post("/predict_clv_probabilistic")
def predict_clv_probabilistic(data: ProbabilisticCLVRequest):

    try:

        bgf = load_bg_nbd_model()
        ggf = load_gamma_gamma_model()

    except:
        raise HTTPException(
            status_code=500,
            detail="Probabilistic CLV models not found"
        )

    clv = ggf.customer_lifetime_value(
        bgf,
        pd.Series([data.Frequency]),
        pd.Series([data.Recency]),
        pd.Series([data.T]),
        pd.Series([data.Monetary]),
        time=12,
        discount_rate=0.01
    )

    return {
        "Expected_CLV_12_months": float(clv.iloc[0])
    }


# ------------------------------------------------
# Probabilistic CLV (Batch)
# ------------------------------------------------

@router.post("/predict_clv_probabilistic_batch")
def predict_clv_probabilistic_batch(request: ProbabilisticCLVBatchRequest):

    try:

        bgf = load_bg_nbd_model()
        ggf = load_gamma_gamma_model()

    except:
        raise HTTPException(
            status_code=500,
            detail="Probabilistic CLV models not found"
        )

    df = pd.DataFrame([c.dict() for c in request.customers])

    clv = ggf.customer_lifetime_value(
        bgf,
        df["Frequency"],
        df["Recency"],
        df["T"],
        df["Monetary"],
        time=12,
        discount_rate=0.01
    )

    return {
        "Expected_CLV_12_months": clv.tolist()
    }