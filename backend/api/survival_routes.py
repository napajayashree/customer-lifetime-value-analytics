from fastapi import APIRouter, HTTPException
import pandas as pd
import pickle
import os

from backend.schemas.request_schema import SurvivalRequest

router = APIRouter()

MODEL_PATH = "models/survival_models/cox_model.pkl"


def load_survival_model():

    if not os.path.exists(MODEL_PATH):

        raise HTTPException(
            status_code=500,
            detail="Survival model file not found"
        )

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    return model

@router.post("/predict_churn_risk")
def predict_churn_risk(data: SurvivalRequest):

    model = load_survival_model()

    try:

        df = pd.DataFrame([{
            "duration": data.duration,
            "event_observed": data.event_observed,
            "Recency": data.Recency,
            "Frequency": data.Frequency,
            "Monetary": data.Monetary,
            "PurchaseFrequency": data.PurchaseFrequency,
            "AvgBasketSize": data.AvgBasketSize,
            "AvgPurchaseValue": data.AvgPurchaseValue
        }])

        risk = model.predict_partial_hazard(df)

        return {
            "Churn_Risk_Score": float(risk.iloc[0])
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
