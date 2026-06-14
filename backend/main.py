from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import api_router


# ---------------------------------
# Create FastAPI Application
# ---------------------------------

app = FastAPI(
    title="Customer Lifetime Value Prediction API",
    description="API for predicting CLV using ML models",
    version="1.0.0"
)


# ---------------------------------
# Enable CORS (for Streamlit / frontend)
# ---------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------
# Root Endpoint
# ---------------------------------

@app.get("/")
def root():

    return {
        "message": "CLV Prediction API is running"
    }


# ---------------------------------
# Register API Routes
# ---------------------------------

app.include_router(api_router)