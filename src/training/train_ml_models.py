import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.models.ml_models.random_forest_clv import train_random_forest
from src.models.ml_models.xgboost_clv import train_xgboost

from src.evaluation.model_comparison import compare_models


# ---------------------------------
# Load Dataset
# ---------------------------------

def load_dataset(path):

    print("\nLoading ML training dataset...")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    print("Dataset loaded successfully")
    print("Shape:", df.shape)

    return df


# ---------------------------------
# Train/Test Split
# ---------------------------------

def split_data(df, target="CLV"):

    print("\nPreparing training data...")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")

    FEATURE_COLUMNS = [
        "Recency",
        "Frequency",
        "Monetary",
        "PurchaseFrequency",
        "AvgBasketSize",
        "AvgPurchaseValue",
        "CustomerLifetime"
    ]

    X = df[FEATURE_COLUMNS]
    # Log transform target
    y = np.log1p(df[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


# ---------------------------------
# Train Models
# ---------------------------------

def train_models(X_train, y_train):

    print("\nTraining ML models...")

    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    return rf_model, xgb_model


# ---------------------------------
# Evaluate Models
# ---------------------------------

def evaluate_models(models, X_test, y_test):

    print("\nEvaluating models...")

    predictions = {}

    for name, model in models.items():
        predictions[name] = model.predict(X_test)

    results = compare_models(y_test, predictions)

    print("\nModel comparison:")
    print(results)

    return results


# ---------------------------------
# Save Models
# ---------------------------------

def save_models(models):

    os.makedirs("models/ml_models", exist_ok=True)

    for name, model in models.items():

        path = f"models/ml_models/{name.lower()}.pkl"

        with open(path, "wb") as f:
            pickle.dump(model, f)

        print(f"Saved model: {path}")


# ---------------------------------
# Training Pipeline
# ---------------------------------

def run_training_pipeline(dataset_path):

    df = load_dataset(dataset_path)

    X_train, X_test, y_train, y_test = split_data(df)

    rf_model, xgb_model = train_models(X_train, y_train)

    models = {
        "RandomForest": rf_model,
        "XGBoost": xgb_model
    }

    results = evaluate_models(models, X_test, y_test)

    save_models(models)

    return models, results


if __name__ == "__main__":

    dataset_path = "data/processed/clv_dataset.csv"

    run_training_pipeline(dataset_path)