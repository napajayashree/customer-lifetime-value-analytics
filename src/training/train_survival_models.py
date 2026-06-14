import os
import pickle
import pandas as pd
from lifelines import CoxPHFitter


def train_survival_model(dataset_path: str):

    print("\nTraining survival model...")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)

    # Remove identifier column
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])

    # Required columns
    required_columns = [
        "duration",
        "event_observed",
        "Recency",
        "Frequency",
        "Monetary",
        "PurchaseFrequency",
        "AvgBasketSize",
        "AvgPurchaseValue",
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(f"Missing survival columns: {missing}")

    training_df = df[required_columns]

    cph = CoxPHFitter(penalizer=0.1)

    cph.fit(
        training_df,
        duration_col="duration",
        event_col="event_observed"
    )

    print("Cox model trained successfully")

    os.makedirs("models/survival_models", exist_ok=True)

    model_path = "models/survival_models/cox_model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(cph, f)

    print(f"Survival model saved to {model_path}")

    return cph