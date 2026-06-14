import pandas as pd
from src.evaluation.metrics import regression_metrics


# ---------------------------------
# Compare Multiple Models
# ---------------------------------

def compare_models(y_true, predictions_dict):

    """
    predictions_dict format:
    {
        "RandomForest": y_pred_rf,
        "XGBoost": y_pred_xgb,
        "NeuralNetwork": y_pred_nn
    }
    """

    results = []

    for model_name, y_pred in predictions_dict.items():

        metrics = regression_metrics(y_true, y_pred)

        metrics["Model"] = model_name

        results.append(metrics)

    results_df = pd.DataFrame(results)

    results_df = results_df[
        ["Model", "MAE", "RMSE", "R2", "MAPE"]
    ]

    results_df = results_df.sort_values(by="RMSE")

    return results_df


# ---------------------------------
# Display Comparison Table
# ---------------------------------

def display_comparison(results_df):

    print("\n===== MODEL COMPARISON =====")

    print(results_df.to_string(index=False))