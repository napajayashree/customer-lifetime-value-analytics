from pydantic import BaseModel
from typing import List

# ------------------------------------------------
# ML CLV Prediction
# ------------------------------------------------

class CLVRequest(BaseModel):

    Recency: float
    Frequency: float
    Monetary: float
    PurchaseFrequency: float
    AvgBasketSize: float
    AvgPurchaseValue: float
    CustomerLifetime: float


class CLVBatchRequest(BaseModel):

    customers: List[CLVRequest]


# ------------------------------------------------
# Probabilistic CLV
# ------------------------------------------------

class ProbabilisticCLVRequest(BaseModel):

    Frequency: float
    Recency: float
    T: float
    Monetary: float


class ProbabilisticCLVBatchRequest(BaseModel):

    customers: List[ProbabilisticCLVRequest]


# ------------------------------------------------
# Survival / Churn Prediction
# ------------------------------------------------

class SurvivalRequest(BaseModel):

    duration: float
    event_observed: int

    Recency: float
    Frequency: float
    Monetary: float
    PurchaseFrequency: float
    AvgBasketSize: float
    AvgPurchaseValue: float
    CustomerLifetime: float