import os

# Data modules
from src.data.load_data import load_dataset
from src.data.data_validation import validate_dataset
from src.data.data_cleaning import clean_transactions
from src.data.data_analysis import dataset_statistics

# Feature modules
from src.features.rfm_features import create_rfm_features
from src.features.behavioral_features import create_behavioral_features
from src.features.survival_features import (
    create_survival_dataset,
    merge_survival_features
)

# ML training
from src.training.train_ml_models import run_training_pipeline
from src.training.train_survival_models import train_survival_model

# Probabilistic CLV models
from src.models.clv_models.bg_nbd_model import (
    train_bg_nbd_model,
    save_bg_nbd_model
)

from src.models.clv_models.gamma_gamma_model import (
    train_gamma_gamma_model,
    save_gamma_gamma_model
)

# Segmentation
from src.segmentation.customer_segmentation import (
    prepare_features,
    train_kmeans,
    assign_segments,
    label_segments,
    segment_statistics,
    segment_summary,
    save_segments
)

# Utilities
from src.utils.helpers import save_dataframe


# ------------------------------------------------
# Ensure required directories exist
# ------------------------------------------------

def ensure_directories():

    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/interim", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    os.makedirs("models/ml_models", exist_ok=True)
    os.makedirs("models/clv_models", exist_ok=True)
    os.makedirs("models/survival_models", exist_ok=True)


# ------------------------------------------------
# Main pipeline
# ------------------------------------------------

def run_pipeline():

    print("\n========== CLV PIPELINE STARTED ==========\n")

    ensure_directories()

    # ------------------------------------------------
    # 1 Load dataset
    # ------------------------------------------------

    dataset_path = "data/raw/online_retail.xlsx"

    print("Loading dataset...")

    df = load_dataset(dataset_path)

    print("Dataset shape:", df.shape)

    # ------------------------------------------------
    # 2 Validation
    # ------------------------------------------------

    print("\nRunning validation...")

    df = validate_dataset(df)

    # ------------------------------------------------
    # 3 Basic analysis
    # ------------------------------------------------

    dataset_statistics(df)

    # ------------------------------------------------
    # 4 Cleaning
    # ------------------------------------------------

    print("\nCleaning dataset...")

    df_clean = clean_transactions(df)

    print("Clean dataset shape:", df_clean.shape)

    save_dataframe(
        df_clean,
        "data/interim/cleaned_transactions.csv"
    )

    # ------------------------------------------------
    # 5 Feature Engineering
    # ------------------------------------------------

    print("\nCreating RFM features...")

    rfm = create_rfm_features(df_clean)

    print("RFM shape:", rfm.shape)

    print("\nCreating behavioral features...")

    behavioral = create_behavioral_features(df_clean)

    print("Behavioral shape:", behavioral.shape)

    print("\nCombining features...")

    features = rfm.join(behavioral, how="inner")

    print("Final feature dataset:", features.shape)

    if features.empty:
        raise ValueError("Feature dataset is empty")

    save_dataframe(
        features,
        "data/processed/rfm_features.csv"
    )

    # ------------------------------------------------
    # 6 Survival dataset
    # ------------------------------------------------

    print("\nCreating survival dataset...")

    survival_df = create_survival_dataset(df_clean)

    print("\nMerging survival data with customer features...")

    survival_training_df = merge_survival_features(
        features,
        survival_df
    )

    save_dataframe(
        survival_training_df,
        "data/processed/survival_dataset.csv"
    )

    # ------------------------------------------------
    # 7 Train survival model
    # ------------------------------------------------

    print("\nTraining survival model...")

    train_survival_model(
        "data/processed/survival_dataset.csv"
    )

    # ------------------------------------------------
    # 8 CLV calculation
    # ------------------------------------------------

    print("\nPreparing CLV dataset...")

    features["CLV"] = (
        features["AvgPurchaseValue"]
        * features["PurchaseFrequency"]
        * (features["CustomerLifetime"] / 365)
    )

    save_dataframe(
        features,
        "data/processed/clv_dataset.csv"
    )

    # ------------------------------------------------
    # 9 Train ML models
    # ------------------------------------------------

    print("\nTraining ML models...")

    run_training_pipeline(
        "data/processed/clv_dataset.csv"
    )

    # ------------------------------------------------
    # 10 Train probabilistic CLV models
    # ------------------------------------------------

    print("\nTraining probabilistic CLV models...")

    bgf = train_bg_nbd_model(rfm)
    save_bg_nbd_model(bgf)

    ggf = train_gamma_gamma_model(rfm)
    save_gamma_gamma_model(ggf)

    # ------------------------------------------------
    # 11 Customer segmentation
    # ------------------------------------------------

    print("\nRunning customer segmentation...")

    X_scaled, scaler = prepare_features(
        features,
        ["Recency", "Frequency", "Monetary"]
    )

    kmeans = train_kmeans(X_scaled)

    segmented_df = assign_segments(features, kmeans, X_scaled)

    segmented_df = label_segments(segmented_df)

    segment_statistics(segmented_df)

    segment_summary(segmented_df)

    save_segments(segmented_df)

    print("\n========== PIPELINE COMPLETED ==========\n")


# ------------------------------------------------
# Entry point
# ------------------------------------------------

if __name__ == "__main__":

    print("RUNNING PIPELINE")

    run_pipeline()