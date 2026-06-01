"""System monitoring endpoints."""

from fastapi import APIRouter
from backend.models.schemas import SystemStats
from backend.services.hardware import get_system_stats
from backend.services.llm import get_inference_stats

router = APIRouter()


@router.get("/stats", response_model=SystemStats)
async def system_stats():
    return get_system_stats()


@router.get("/inference")
async def inference_stats():
    return get_inference_stats()
