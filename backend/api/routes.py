from fastapi import APIRouter

from backend.api.clv_routes import router as clv_router
from backend.api.health_routes import router as health_router
from backend.api.survival_routes import router as survival_router

api_router = APIRouter()

# Register routes

api_router.include_router(health_router)
api_router.include_router(clv_router)
api_router.include_router(survival_router)