import pandas as pd
import pickle


# ---------------------------------
# Load Model
# ---------------------------------

def load_model(path):

    print(f"\nLoading model from {path}")

    with open(path, "rb") as f:
        model = pickle.load(f)

    return model


# ---------------------------------
# Predict CLV
# ---------------------------------

def predict_clv(model, features_df):

    print("\nGenerating CLV predictions...")

    predictions = model.predict(features_df)

    result = features_df.copy()

    result["Predicted_CLV"] = predictions

    return result


# ---------------------------------
# Predict Single Customer
# ---------------------------------

def predict_single_customer(model, customer_features):

    print("\nPredicting CLV for single customer...")

    df = pd.DataFrame([customer_features])

    prediction = model.predict(df)

    return prediction[0]


# ---------------------------------
# Save Predictions
# ---------------------------------

def save_predictions(df):

    path = "data/processed/clv_predictions.csv"

    df.to_csv(path, index=False)

    print(f"\nPredictions saved to {path}")