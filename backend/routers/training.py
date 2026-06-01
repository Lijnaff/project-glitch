"""RLHF training endpoints."""

import uuid
from datetime import datetime, timezone
from fastapi import APIRouter
from backend.models.schemas import TrainingRequest, TrainingStatus

router = APIRouter()

_training_jobs: dict[str, TrainingStatus] = {}


@router.post("/start", response_model=TrainingStatus)
async def start_training(req: TrainingRequest):
    job_id = str(uuid.uuid4())[:8]
    status = TrainingStatus(
        job_id=job_id,
        status="running",
        progress_pct=0.0,
        current_epoch=0,
        total_epochs=req.epochs,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    _training_jobs[job_id] = status
    return status


@router.get("/status/{job_id}", response_model=TrainingStatus)
async def training_status(job_id: str):
    if job_id not in _training_jobs:
        return TrainingStatus(job_id=job_id, status="not_found")
    return _training_jobs[job_id]


@router.get("/jobs", response_model=list[TrainingStatus])
async def list_training_jobs():
    return list(_training_jobs.values())


@router.post("/stop/{job_id}")
async def stop_training(job_id: str):
    if job_id in _training_jobs:
        _training_jobs[job_id].status = "stopped"
    return {"ok": True}
