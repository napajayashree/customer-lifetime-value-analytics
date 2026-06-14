from fastapi import APIRouter

# Create router
router = APIRouter()


# ---------------------------------
# Health Check Endpoint
# ---------------------------------

@router.get("/health")
def health_check():

    return {
        "status": "ok",
        "message": "CLV Prediction API is running"
    }