# Customer Lifetime Analytics Platform

A full-stack machine learning platform that predicts Customer Lifetime Value (CLV), estimates churn risk, segments customers, and provides future revenue insights using machine learning, survival analysis, FastAPI, and Streamlit.

## Features

- Predicts Customer Lifetime Value using Random Forest and XGBoost
- Estimates future purchases using BG-NBD and Gamma-Gamma models
- Predicts churn risk using survival analysis
- Segments customers using K-Means clustering
- Provides REST API endpoints with FastAPI
- Includes an interactive Streamlit dashboard for business insights

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn, XGBoost
- Lifetimes, Lifelines
- FastAPI, Uvicorn
- Streamlit, Plotly
- TensorFlow/Keras

## Dataset

This project uses the Online Retail Dataset from the UCI Machine Learning Repository.  
The dataset contains customer transaction records including invoice details, quantity, price, customer ID, country, and purchase date.

## Project Workflow

Raw Transaction Data  
→ Data Cleaning & Feature Engineering  
→ Model Training  
→ CLV Prediction & Churn Risk Analysis  
→ FastAPI Prediction Service  
→ Streamlit Dashboard

## Machine Learning Models

- Random Forest Regressor for CLV prediction
- XGBoost Regressor for boosted CLV prediction
- BG-NBD model for future transaction prediction
- Gamma-Gamma model for expected monetary value
- Cox Survival Model for churn risk analysis
- K-Means clustering for customer segmentation

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/predict_clv` | Predicts CLV for a single customer |
| POST | `/predict_clv_batch` | Predicts CLV for multiple customers |
| POST | `/predict_clv_probabilistic` | Estimates future CLV using probabilistic models |
| POST | `/predict_churn_risk` | Predicts customer churn risk |
| GET | `/model/features` | Returns model feature importance |

## How to Run

```bash
pip install -r requirements.txt
python run_pipeline.py
uvicorn main:app --reload
streamlit run dashboard.py
```
## Author
JAYASHREE NAPA

