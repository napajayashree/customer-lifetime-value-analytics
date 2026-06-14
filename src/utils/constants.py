# ---------------------------------
# Project Paths
# ---------------------------------

RAW_DATA_PATH = "data/raw/"
INTERIM_DATA_PATH = "data/interim/"
PROCESSED_DATA_PATH = "data/processed/"
MODEL_PATH = "models/"


# ---------------------------------
# Dataset Files
# ---------------------------------

TRANSACTIONS_FILE = "online_retail.xlsx"
RFM_FILE = "rfm_features.csv"
SURVIVAL_DATASET = "survival_dataset.csv"
CLV_DATASET = "clv_dataset.csv"


# ---------------------------------
# Model Files
# ---------------------------------

BG_NBD_MODEL = "models/clv_models/bg_nbd.pkl"
GAMMA_GAMMA_MODEL = "models/clv_models/gamma_gamma.pkl"
RANDOM_FOREST_MODEL = "models/ml_models/randomforest.pkl"
XGBOOST_MODEL = "models/ml_models/xgboost.pkl"
COX_MODEL = "models/survival_models/cox_model.pkl"


# ---------------------------------
# Survival Analysis
# ---------------------------------

CHURN_THRESHOLD_DAYS = 180


# ---------------------------------
# CLV Settings
# ---------------------------------

CLV_PREDICTION_PERIOD = 12
DISCOUNT_RATE = 0.01


# ---------------------------------
# Feature Columns
# ---------------------------------

RFM_FEATURES = [
    "Recency",
    "Frequency",
    "Monetary"
]

BEHAVIORAL_FEATURES = [
    "PurchaseFrequency",
    "AvgBasketSize",
    "AvgPurchaseValue",
    "CustomerLifetime"
]