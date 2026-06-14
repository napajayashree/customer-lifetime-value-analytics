import pandas as pd

from sklearn.model_selection import train_test_split

from src.models.ml_models.random_forest_clv import train_random_forest
from src.models.ml_models.xgboost_clv import train_xgboost

from src.evaluation.metrics import regression_metrics
from src.evaluation.model_comparison import compare_models


# ---------------------------------
# Load Feature Dataset
# ---------------------------------

def load_training_data(path):

    print("\nLoading training dataset...")

    df = pd.read_csv(path)

    return df


# ---------------------------------
# Prepare Train/Test Data
# ---------------------------------

def prepare_data(df, target="CLV"):

    print("\nPreparing training data...")

    X = df.drop(columns=[target])

    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


# ---------------------------------
# Train All Models
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

    print(results)

    return results


# ---------------------------------
# Main Training Pipeline
# ---------------------------------

def run_training_pipeline(dataset_path):

    df = load_training_data(dataset_path)

    X_train, X_test, y_train, y_test = prepare_data(df)

    rf_model, xgb_model = train_models(X_train, y_train)

    models = {
        "RandomForest": rf_model,
        "XGBoost": xgb_model
    }

    results = evaluate_models(models, X_test, y_test)

    return models, results


# ---------------------------------
# Run Script
# ---------------------------------

if __name__ == "__main__":

    dataset_path = "data/processed/clv_dataset.csv"

    models, results = run_training_pipeline(dataset_path)

    print("\nTraining pipeline completed.")