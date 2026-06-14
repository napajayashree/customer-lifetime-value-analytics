import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ---------------------------------
# Mean Absolute Percentage Error
# ---------------------------------

def mean_absolute_percentage_error(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# ---------------------------------
# Compute All Metrics
# ---------------------------------

def regression_metrics(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    r2 = r2_score(y_true, y_pred)

    mape = mean_absolute_percentage_error(y_true, y_pred)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }

    return metrics


# ---------------------------------
# Print Metrics
# ---------------------------------

def print_metrics(metrics_dict):

    print("\n===== MODEL PERFORMANCE =====")

    for key, value in metrics_dict.items():

        print(f"{key}: {round(value,4)}")