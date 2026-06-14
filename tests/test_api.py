import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


# ---------------------------------
# Health Endpoint Test
# ---------------------------------

def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert "status" in data


# ---------------------------------
# CLV Prediction Endpoint
# ---------------------------------

def test_clv_prediction():

    sample_customer = {
        "Recency": 30,
        "Frequency": 5,
        "Monetary": 250,
        "PurchaseFrequency": 5,
        "AvgBasketSize": 3,
        "AvgPurchaseValue": 120,
        "CustomerLifetime": 180
    }

    response = client.post("/predict_clv", json=sample_customer)

    assert response.status_code == 200

    data = response.json()

    assert "Predicted_CLV" in data